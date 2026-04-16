from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from controlgen.transfer_function import tf_from_node
from controlgen.types import GraphNodeSpec, ParameterizedGraph, ParameterizedNode, PortSpec


LAYER_ORDER = {
    "reference": 0,
    "controller": 1,
    "actuator": 2,
    "disturbance": 2,
    "process": 3,
    "noise": 4,
    "sensor": 5,
    "output": 6,
}


@dataclass(frozen=True)
class SignalSpec:
    kind: str
    amplitude: float = 1.0
    frequency: float = 1.0
    phase: float = 0.0
    offset: float = 0.0
    start_time: float = 0.0
    duration: float = 1.0
    levels: tuple[float, ...] = field(default_factory=tuple)
    bandwidth: float = 1.0
    seed: int = 0


@dataclass(frozen=True)
class SimulationConfig:
    duration: float = 12.0
    dt: float = 0.01


@dataclass(frozen=True)
class ScenarioSpec:
    external_inputs: dict[str, tuple[SignalSpec, ...]]
    disturbances: dict[str, tuple[SignalSpec, ...]] = field(default_factory=dict)
    noises: dict[str, tuple[SignalSpec, ...]] = field(default_factory=dict)
    sim_config: SimulationConfig = field(default_factory=SimulationConfig)

    def as_serializable(self) -> dict[str, object]:
        return {
            "external_inputs": {
                node_id: [vars(spec) for spec in specs]
                for node_id, specs in self.external_inputs.items()
            },
            "disturbances": {
                node_id: [vars(spec) for spec in specs]
                for node_id, specs in self.disturbances.items()
            },
            "noises": {
                node_id: [vars(spec) for spec in specs]
                for node_id, specs in self.noises.items()
            },
            "sim_config": vars(self.sim_config),
        }


@dataclass(frozen=True)
class SignalBundle:
    t: np.ndarray
    external_inputs: dict[str, np.ndarray]
    primary_outputs: dict[str, np.ndarray]
    tap_signals: dict[str, np.ndarray]
    disturbance_signals: dict[str, np.ndarray]
    noise_signals: dict[str, np.ndarray]
    teacher_signals: dict[str, np.ndarray]
    scenario: ScenarioSpec

    def as_serializable(self) -> dict[str, object]:
        return {
            "t": self.t.tolist(),
            "external_inputs": {name: values.tolist() for name, values in self.external_inputs.items()},
            "primary_outputs": {name: values.tolist() for name, values in self.primary_outputs.items()},
            "tap_signals": {name: values.tolist() for name, values in self.tap_signals.items()},
            "disturbance_signals": {name: values.tolist() for name, values in self.disturbance_signals.items()},
            "noise_signals": {name: values.tolist() for name, values in self.noise_signals.items()},
            "teacher_signals": {name: values.tolist() for name, values in self.teacher_signals.items()},
            "scenario": self.scenario.as_serializable(),
        }


@dataclass
class _DiscreteLinearBlock:
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    d: np.ndarray
    x: np.ndarray

    @classmethod
    def from_parameterized_node(
        cls,
        node: ParameterizedNode,
        dt: float,
    ) -> "_DiscreteLinearBlock | None":
        if node.dynamics is None:
            return None
        system = tf_from_node(node.dynamics)
        if system.state_dimension == 0:
            d = np.asarray(system.d, dtype=float)
            return cls(
                a=np.zeros((0, 0), dtype=float),
                b=np.zeros((0, system.input_channels), dtype=float),
                c=np.zeros((system.output_channels, 0), dtype=float),
                d=d,
                x=np.zeros(0, dtype=float),
            )
        ad, bd, cd, dd, _ = signal.cont2discrete((system.a, system.b, system.c, system.d), dt=dt)
        return cls(
            a=np.asarray(ad, dtype=float),
            b=np.asarray(bd, dtype=float),
            c=np.asarray(cd, dtype=float),
            d=np.asarray(dd, dtype=float),
            x=np.zeros(system.state_dimension, dtype=float),
        )

    def step(self, u: np.ndarray) -> np.ndarray:
        u_vec = np.asarray(u, dtype=float).reshape(-1, 1)
        if self.x.size:
            x_vec = self.x.reshape(-1, 1)
            y = self.c @ x_vec + self.d @ u_vec
            self.x = (self.a @ x_vec + self.b @ u_vec).reshape(-1)
            return y.reshape(-1)
        return (self.d @ u_vec).reshape(-1)


def simulate(parameterized_graph: ParameterizedGraph, scenario: ScenarioSpec) -> SignalBundle:
    return simulate_graph(parameterized_graph, scenario)


def simulate_graph(parameterized_graph: ParameterizedGraph, scenario: ScenarioSpec) -> SignalBundle:
    dt = scenario.sim_config.dt
    t = np.arange(0.0, scenario.sim_config.duration + dt, dt, dtype=float)
    node_map = {node.spec.node_id: node for node in parameterized_graph.nodes}
    graph = parameterized_graph.graph
    order = sorted(graph.nodes, key=lambda node: (LAYER_ORDER[node.layer], node.node_id))
    runtimes = {
        node.spec.node_id: _DiscreteLinearBlock.from_parameterized_node(node, dt)
        for node in parameterized_graph.nodes
    }

    external_inputs = {
        node_id: _build_signal_matrix(t, specs)
        for node_id, specs in scenario.external_inputs.items()
    }
    disturbance_signals = {
        node_id: _build_signal_matrix(t, specs)
        for node_id, specs in scenario.disturbances.items()
    }
    noise_signals = {
        node_id: _build_signal_matrix(t, specs)
        for node_id, specs in scenario.noises.items()
    }

    tap_signals = {
        tap.tap_id: np.zeros((len(t), tap.dimension), dtype=float)
        for tap in graph.taps
    }
    teacher_signals = {
        f"{node.spec.node_id}:state": np.zeros((len(t), runtimes[node.spec.node_id].x.size), dtype=float)
        for node in parameterized_graph.nodes
        if runtimes[node.spec.node_id] is not None and runtimes[node.spec.node_id].x.size
    }

    output_node_id = str(graph.metadata["primary_output_node"])
    output_port_name = str(graph.metadata["primary_output_port"])
    primary_output_dim = _find_port(node_map[output_node_id].spec.output_ports, output_port_name).dimension
    primary_outputs = {
        "system_output": np.zeros((len(t), primary_output_dim), dtype=float),
    }

    previous_outputs = {
        node.spec.node_id: np.zeros(sum(port.dimension for port in node.spec.output_ports), dtype=float)
        for node in parameterized_graph.nodes
    }

    incoming_edges: dict[str, list] = {}
    for edge in graph.edges:
        incoming_edges.setdefault(edge.target_node, []).append(edge)

    for time_index, _ in enumerate(t):
        current_outputs: dict[str, np.ndarray] = {}
        for graph_node in order:
            parameterized_node = node_map[graph_node.node_id]
            output = _step_graph_node(
                graph_node,
                parameterized_node,
                runtimes[graph_node.node_id],
                incoming_edges.get(graph_node.node_id, []),
                current_outputs,
                previous_outputs,
                time_index,
                external_inputs,
                disturbance_signals,
                noise_signals,
                dt,
                parameterized_graph,
            )
            current_outputs[graph_node.node_id] = output
            runtime = runtimes[graph_node.node_id]
            if runtime is not None and runtime.x.size:
                teacher_signals[f"{graph_node.node_id}:state"][time_index] = runtime.x

        previous_outputs = {node_id: values.copy() for node_id, values in current_outputs.items()}
        primary_outputs["system_output"][time_index] = _port_view(
            node_map[output_node_id].spec.output_ports,
            current_outputs[output_node_id],
            output_port_name,
        )
        for tap in graph.taps:
            tap_signals[tap.tap_id][time_index] = _port_view(
                node_map[tap.source_node].spec.output_ports,
                current_outputs[tap.source_node],
                tap.source_port,
            )

    return SignalBundle(
        t=t,
        external_inputs=external_inputs,
        primary_outputs=primary_outputs,
        tap_signals=tap_signals,
        disturbance_signals=disturbance_signals,
        noise_signals=noise_signals,
        teacher_signals=teacher_signals,
        scenario=scenario,
    )


def build_signal(t: np.ndarray, spec: SignalSpec | None) -> np.ndarray:
    if spec is None or spec.kind == "none":
        return np.zeros_like(t, dtype=float)
    kind = spec.kind.lower()
    if kind == "step":
        signal_values = np.full(len(t), spec.offset, dtype=float)
        signal_values[t >= spec.start_time] += spec.amplitude
        return signal_values
    if kind == "ramp":
        ramp = np.clip(t - spec.start_time, 0.0, None)
        return spec.offset + spec.amplitude * ramp
    if kind == "chirp":
        active = np.clip(t - spec.start_time, 0.0, None)
        f1 = max(spec.frequency * 4.0, spec.frequency + 0.1)
        chirp_values = signal.chirp(active, f0=spec.frequency, t1=max(active[-1], 1e-6), f1=f1)
        chirp_values[t < spec.start_time] = 0.0
        return spec.offset + spec.amplitude * chirp_values
    if kind == "multistep":
        levels = spec.levels or (0.4, 1.0, -0.4, 0.8)
        base = np.full(len(t), spec.offset, dtype=float)
        segments = np.array_split(np.where(t >= spec.start_time)[0], len(levels))
        for level, indices in zip(levels, segments, strict=False):
            base[indices] += float(level)
        return base
    if kind == "prbs":
        rng = np.random.default_rng(spec.seed)
        values = np.zeros(len(t), dtype=float)
        step = max(int(round(spec.duration / max(t[1] - t[0], 1e-6))), 1) if len(t) > 1 else 1
        current = spec.offset
        for idx in range(len(t)):
            if idx % step == 0 and t[idx] >= spec.start_time:
                current = spec.offset + spec.amplitude * (1.0 if rng.random() > 0.5 else -1.0)
            values[idx] = current if t[idx] >= spec.start_time else spec.offset
        return values
    if kind == "load_step":
        values = np.zeros(len(t), dtype=float)
        values[t >= spec.start_time] = spec.amplitude
        return values
    if kind == "load_pulse":
        values = np.zeros(len(t), dtype=float)
        end_time = spec.start_time + spec.duration
        active = (t >= spec.start_time) & (t <= end_time)
        values[active] = spec.amplitude
        return values
    if kind == "filtered_noise":
        rng = np.random.default_rng(spec.seed)
        white = rng.normal(0.0, spec.amplitude, size=len(t))
        alpha = np.exp(-2.0 * np.pi * max(spec.bandwidth, 1e-3) * (t[1] - t[0] if len(t) > 1 else 0.01))
        values = np.zeros(len(t), dtype=float)
        for idx, value in enumerate(white):
            values[idx] = alpha * (values[idx - 1] if idx > 0 else 0.0) + (1.0 - alpha) * value
        values[t < spec.start_time] = 0.0
        return values
    if kind == "white_noise":
        rng = np.random.default_rng(spec.seed)
        return rng.normal(spec.offset, spec.amplitude, size=len(t))
    if kind == "lowpass_noise":
        rng = np.random.default_rng(spec.seed)
        white = rng.normal(0.0, spec.amplitude, size=len(t))
        cutoff = max(spec.bandwidth, 1e-3)
        alpha = np.exp(-2.0 * np.pi * cutoff * (t[1] - t[0] if len(t) > 1 else 0.01))
        values = np.zeros(len(t), dtype=float)
        for idx, value in enumerate(white):
            values[idx] = alpha * (values[idx - 1] if idx > 0 else 0.0) + (1.0 - alpha) * value
        return values + spec.offset
    raise ValueError(f"Unsupported signal kind: {spec.kind}")


def _build_signal_matrix(t: np.ndarray, specs: tuple[SignalSpec, ...]) -> np.ndarray:
    if not specs:
        return np.zeros((len(t), 0), dtype=float)
    return np.column_stack([build_signal(t, spec) for spec in specs])


def _step_graph_node(
    graph_node: GraphNodeSpec,
    parameterized_node: ParameterizedNode,
    runtime: _DiscreteLinearBlock | None,
    edges: list,
    current_outputs: dict[str, np.ndarray],
    previous_outputs: dict[str, np.ndarray],
    time_index: int,
    external_inputs: dict[str, np.ndarray],
    disturbance_signals: dict[str, np.ndarray],
    noise_signals: dict[str, np.ndarray],
    dt: float,
    parameterized_graph: ParameterizedGraph,
) -> np.ndarray:
    if graph_node.kind == "reference_source":
        return external_inputs.get(graph_node.node_id, np.zeros((1, graph_node.output_ports[0].dimension), dtype=float))[time_index]
    if graph_node.kind == "disturbance_injector":
        return disturbance_signals.get(graph_node.node_id, np.zeros((1, graph_node.output_ports[0].dimension), dtype=float))[time_index]
    if graph_node.kind == "noise_injector":
        return noise_signals.get(graph_node.node_id, np.zeros((1, graph_node.output_ports[0].dimension), dtype=float))[time_index]

    input_vector = _collect_node_inputs(graph_node, edges, current_outputs, previous_outputs, parameterized_graph)
    if runtime is not None:
        output = runtime.step(input_vector)
    else:
        output = input_vector
    if parameterized_node.nonlinearity == "saturation" and parameterized_node.saturation_limit is not None:
        output = np.clip(output, -parameterized_node.saturation_limit, parameterized_node.saturation_limit)
    if parameterized_node.nonlinearity == "rate_limit" and parameterized_node.rate_limit is not None:
        prev = previous_outputs.get(graph_node.node_id, np.zeros_like(output))
        max_delta = parameterized_node.rate_limit * dt
        output = prev + np.clip(output - prev, -max_delta, max_delta)
    if abs(parameterized_node.bias) > 0.0:
        output = output + parameterized_node.bias
    return np.asarray(output, dtype=float).reshape(-1)


def _collect_node_inputs(
    graph_node: GraphNodeSpec,
    edges: list,
    current_outputs: dict[str, np.ndarray],
    previous_outputs: dict[str, np.ndarray],
    parameterized_graph: ParameterizedGraph,
) -> np.ndarray:
    node_lookup = {node.spec.node_id: node.spec for node in parameterized_graph.nodes}
    port_buffers = {
        port.name: np.zeros(port.dimension, dtype=float)
        for port in graph_node.input_ports
    }
    for edge in edges:
        source_spec = node_lookup[edge.source_node]
        source_value = _select_source_value(graph_node, source_spec, current_outputs, previous_outputs, edge.source_node)
        source_port_value = _port_view(source_spec.output_ports, source_value, edge.source_port)
        target_port = _find_port(graph_node.input_ports, edge.target_port)
        matrix = np.asarray(edge.matrix, dtype=float) if edge.matrix is not None else np.eye(target_port.dimension, source_port_value.size, dtype=float)
        port_buffers[edge.target_port] += matrix @ source_port_value
    if not graph_node.input_ports:
        return np.zeros(0, dtype=float)
    return np.concatenate([port_buffers[port.name] for port in graph_node.input_ports], axis=0)


def _select_source_value(
    target_node: GraphNodeSpec,
    source_node: GraphNodeSpec,
    current_outputs: dict[str, np.ndarray],
    previous_outputs: dict[str, np.ndarray],
    source_node_id: str,
) -> np.ndarray:
    if LAYER_ORDER[source_node.layer] < LAYER_ORDER[target_node.layer] and source_node_id in current_outputs:
        return current_outputs[source_node_id]
    return previous_outputs[source_node_id]


def _port_view(ports: tuple[PortSpec, ...], values: np.ndarray, port_name: str) -> np.ndarray:
    start = 0
    for port in ports:
        end = start + port.dimension
        if port.name == port_name:
            return np.asarray(values[start:end], dtype=float)
        start = end
    raise KeyError(port_name)


def _find_port(ports: tuple[PortSpec, ...], name: str) -> PortSpec:
    for port in ports:
        if port.name == name:
            return port
    raise KeyError(name)
