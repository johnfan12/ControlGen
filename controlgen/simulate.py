from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import signal

from controlgen.transfer_function import tf_from_node
from controlgen.types import ControlNode, ParameterizedGraph


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
    reference: SignalSpec
    disturbance: SignalSpec | None = None
    noise: SignalSpec | None = None
    sim_config: SimulationConfig = field(default_factory=SimulationConfig)

    def as_serializable(self) -> dict[str, object]:
        return {
            "reference": asdict(self.reference),
            "disturbance": asdict(self.disturbance) if self.disturbance else None,
            "noise": asdict(self.noise) if self.noise else None,
            "sim_config": asdict(self.sim_config),
        }


@dataclass(frozen=True)
class SignalBundle:
    t: np.ndarray
    signals: dict[str, np.ndarray]
    scenario: ScenarioSpec

    def as_serializable(self) -> dict[str, object]:
        return {
            "t": self.t.tolist(),
            "signals": {name: values.tolist() for name, values in self.signals.items()},
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
    def from_control_node(cls, node: ControlNode | None, dt: float) -> "_DiscreteLinearBlock | None":
        if node is None:
            return None
        system = tf_from_node(node)
        if system.state_dimension == 0:
            d = np.asarray(system.d, dtype=float)
            return cls(
                a=np.zeros((0, 0), dtype=float),
                b=np.zeros((0, 1), dtype=float),
                c=np.zeros((1, 0), dtype=float),
                d=d.reshape(1, 1),
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

    def step(self, u: float) -> float:
        u_vec = np.array([[float(u)]], dtype=float)
        if self.x.size:
            y = self.c @ self.x.reshape(-1, 1) + self.d @ u_vec
            self.x = (self.a @ self.x.reshape(-1, 1) + self.b @ u_vec).reshape(-1)
            return float(y.reshape(-1)[0])
        return float((self.d @ u_vec).reshape(-1)[0])


def simulate(parameterized_graph: ParameterizedGraph, scenario: ScenarioSpec) -> SignalBundle:
    t = np.arange(
        0.0,
        scenario.sim_config.duration + scenario.sim_config.dt,
        scenario.sim_config.dt,
        dtype=float,
    )
    reference = build_signal(t, scenario.reference)
    disturbance = build_signal(t, scenario.disturbance) if scenario.disturbance else np.zeros_like(reference)
    noise = build_signal(t, scenario.noise) if scenario.noise else np.zeros_like(reference)

    controller = _DiscreteLinearBlock.from_control_node(parameterized_graph.controller_dynamics, scenario.sim_config.dt)
    actuator = _DiscreteLinearBlock.from_control_node(parameterized_graph.actuator_dynamics, scenario.sim_config.dt)
    plant_control = _DiscreteLinearBlock.from_control_node(
        parameterized_graph.plant_control_dynamics,
        scenario.sim_config.dt,
    )
    plant_dist = _DiscreteLinearBlock.from_control_node(
        parameterized_graph.plant_disturbance_dynamics,
        scenario.sim_config.dt,
    )
    sensor = _DiscreteLinearBlock.from_control_node(parameterized_graph.sensor_dynamics, scenario.sim_config.dt)

    signals = {
        "r": np.zeros_like(reference),
        "d": np.zeros_like(reference),
        "n": np.zeros_like(reference),
        "e": np.zeros_like(reference),
        "u_cmd": np.zeros_like(reference),
        "u_act": np.zeros_like(reference),
        "y": np.zeros_like(reference),
        "y_m": np.zeros_like(reference),
    }

    y_measured_prev = 0.0
    disturbance_injection = parameterized_graph.graph.disturbance.injection if parameterized_graph.graph.disturbance else "output"

    for idx, _ in enumerate(t):
        r_value = float(reference[idx])
        d_value = float(disturbance[idx])
        n_value = float(noise[idx])
        e_value = r_value - y_measured_prev
        u_cmd = controller.step(e_value) if controller else e_value
        u_linear = actuator.step(u_cmd) if actuator else u_cmd
        if parameterized_graph.actuator_saturation_limit is not None:
            u_act = float(
                np.clip(
                    u_linear,
                    -parameterized_graph.actuator_saturation_limit,
                    parameterized_graph.actuator_saturation_limit,
                )
            )
        else:
            u_act = float(u_linear)

        process_input = u_act
        process_output = 0.0
        if disturbance_injection == "input":
            process_input += d_value
        process_output += plant_control.step(process_input) if plant_control else process_input
        if disturbance_injection == "output":
            process_output += plant_dist.step(d_value) if plant_dist else d_value
        y_value = float(process_output)
        sensed = sensor.step(y_value) if sensor else y_value
        y_measured = float(sensed + parameterized_graph.sensor_bias + n_value)
        y_measured_prev = y_measured

        signals["r"][idx] = r_value
        signals["d"][idx] = d_value
        signals["n"][idx] = n_value
        signals["e"][idx] = e_value
        signals["u_cmd"][idx] = u_cmd
        signals["u_act"][idx] = u_act
        signals["y"][idx] = y_value
        signals["y_m"][idx] = y_measured

    return SignalBundle(t=t, signals=signals, scenario=scenario)


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
