from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from controlgen.dataset import DatasetSample, build_sample
from controlgen.simulate import ScenarioSpec, SignalSpec, SimulationConfig, simulate_graph
from controlgen.transfer_function import matrix_to_tuple
from controlgen.types import (
    ControlGraph,
    ControlNode,
    GraphEdgeSpec,
    GraphNodeSpec,
    MatrixGainNode,
    ParameterizedGraph,
    ParameterizedNode,
    PortSpec,
    SSNode,
    TapSpec,
)


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
class GrammarConfig:
    graph_size_range: tuple[int, int] = (8, 24)
    io_channels_range: tuple[int, int] = (3, 5)
    controller_nodes_range: tuple[int, int] = (2, 4)
    sensor_nodes_range: tuple[int, int] = (2, 4)
    reference_family_weights: dict[str, float] = field(
        default_factory=lambda: {
            "step": 0.30,
            "multistep": 0.25,
            "ramp": 0.10,
            "chirp": 0.20,
            "prbs": 0.15,
        }
    )
    controller_family_weights: dict[str, float] = field(
        default_factory=lambda: {
            "p": 0.20,
            "pi": 0.35,
            "pid": 0.30,
            "mimo_filter": 0.15,
        }
    )
    actuator_family_weights: dict[str, float] = field(
        default_factory=lambda: {
            "ideal": 0.25,
            "lag": 0.35,
            "lag_saturation": 0.25,
            "rate_limited": 0.15,
        }
    )
    process_family_weights: dict[str, float] = field(
        default_factory=lambda: {
            "first_order_bank": 0.30,
            "second_order_bank": 0.25,
            "coupled_process": 0.25,
            "oscillatory_bank": 0.20,
        }
    )
    sensor_family_weights: dict[str, float] = field(
        default_factory=lambda: {
            "ideal": 0.25,
            "lag_filter": 0.45,
            "filtered": 0.30,
        }
    )
    disturbance_family_weights: dict[str, float] = field(
        default_factory=lambda: {
            "load_step": 0.40,
            "load_pulse": 0.25,
            "filtered_noise": 0.35,
        }
    )
    noise_family_weights: dict[str, float] = field(
        default_factory=lambda: {
            "white_noise": 0.55,
            "lowpass_noise": 0.45,
        }
    )
    coupling_family_weights: dict[str, float] = field(
        default_factory=lambda: {"weak": 0.35, "mixed": 0.40, "strong": 0.25}
    )


@dataclass(frozen=True)
class SamplingConfig:
    controller_gain_range: tuple[float, float] = (0.3, 2.4)
    controller_integral_range: tuple[float, float] = (0.05, 1.2)
    controller_derivative_range: tuple[float, float] = (0.02, 0.4)
    actuator_gain_range: tuple[float, float] = (0.8, 1.3)
    actuator_tau_range: tuple[float, float] = (0.05, 0.5)
    actuator_saturation_range: tuple[float, float] = (0.7, 1.8)
    actuator_rate_limit_range: tuple[float, float] = (0.5, 3.0)
    process_gain_range: tuple[float, float] = (0.2, 1.5)
    process_tau_range: tuple[float, float] = (0.4, 3.0)
    process_wn_range: tuple[float, float] = (0.5, 2.4)
    process_zeta_range: tuple[float, float] = (0.2, 0.9)
    sensor_gain_range: tuple[float, float] = (0.9, 1.1)
    sensor_tau_range: tuple[float, float] = (0.02, 0.4)
    sensor_bias_range: tuple[float, float] = (-0.03, 0.03)
    disturbance_amplitude_range: tuple[float, float] = (0.15, 1.0)
    noise_std_range: tuple[float, float] = (0.01, 0.06)
    tap_extra_probability: float = 0.15


@dataclass(frozen=True)
class DatasetConfig:
    count: int = 10
    seed: int = 0
    sim_config: SimulationConfig = field(default_factory=SimulationConfig)


def generate_graph(
    config: GrammarConfig,
    rng: np.random.Generator,
    structure_family: str | None = None,
) -> tuple[ControlGraph, dict[str, object]]:
    del structure_family
    topology_family = "layered_mimo_graph"
    input_channels = int(rng.integers(config.io_channels_range[0], config.io_channels_range[1] + 1))
    output_channels = int(rng.integers(config.io_channels_range[0], config.io_channels_range[1] + 1))
    target_nodes = int(rng.integers(config.graph_size_range[0], config.graph_size_range[1] + 1))

    controller_count = int(rng.integers(config.controller_nodes_range[0], min(config.controller_nodes_range[1], input_channels) + 1))
    actuator_count = controller_count
    sensor_count = int(rng.integers(config.sensor_nodes_range[0], min(config.sensor_nodes_range[1], output_channels) + 1))
    disturbance_count = int(rng.integers(1, 3))
    noise_count = int(rng.integers(1, 3))
    fixed_nodes = 1 + controller_count + actuator_count + sensor_count + disturbance_count + noise_count + 1
    process_count = max(3, min(8, target_nodes - fixed_nodes))
    target_nodes = fixed_nodes + process_count

    reference_family = _choose_weighted(config.reference_family_weights, rng)
    controller_family = _choose_weighted(config.controller_family_weights, rng)
    actuator_family = _choose_weighted(config.actuator_family_weights, rng)
    process_family = _choose_weighted(config.process_family_weights, rng)
    sensor_family = _choose_weighted(config.sensor_family_weights, rng)
    disturbance_family = _choose_weighted(config.disturbance_family_weights, rng)
    noise_family = _choose_weighted(config.noise_family_weights, rng)
    coupling_family = _choose_weighted(config.coupling_family_weights, rng)

    controller_dims = _partition_total(input_channels, controller_count, rng)
    sensor_output_dims = _partition_total(output_channels, sensor_count, rng)
    internal_process_channels = output_channels + process_count + int(rng.integers(1, output_channels + 2))
    process_output_dims = _partition_total(internal_process_channels, process_count, rng)
    process_input_dims = [int(rng.integers(1, input_channels + 1)) for _ in range(process_count)]
    sensor_input_dims = [int(rng.integers(1, internal_process_channels + 1)) for _ in range(sensor_count)]

    input_ports = (PortSpec(name="reference", direction="input", dimension=input_channels, role="external_input"),)
    output_ports = (PortSpec(name="system_output", direction="output", dimension=output_channels, role="primary_output"),)

    nodes: list[GraphNodeSpec] = []
    edges: list[GraphEdgeSpec] = []
    taps: list[TapSpec] = []

    ref_node = GraphNodeSpec(
        node_id="ref_0",
        kind="reference_source",
        family=reference_family,
        layer="reference",
        input_ports=(),
        output_ports=(PortSpec(name="y", direction="output", dimension=input_channels, role="reference"),),
    )
    nodes.append(ref_node)
    taps.append(TapSpec("tap_ref_0", "ref_0", "y", "reference", "primary", input_channels))

    controller_nodes = []
    actuator_nodes = []
    process_nodes = []
    sensor_nodes = []
    disturbance_nodes = []
    noise_nodes = []

    ref_offset = 0
    sensor_output_offsets = _offsets(sensor_output_dims)
    controller_ref_slices: list[tuple[int, int]] = []

    for index, dim in enumerate(controller_dims):
        ctrl_id = f"ctrl_{index}"
        act_id = f"act_{index}"
        ctrl_spec = GraphNodeSpec(
            node_id=ctrl_id,
            kind="controller_bank",
            family=controller_family,
            layer="controller",
            input_ports=(PortSpec(name="u", direction="input", dimension=dim, role="controller_input"),),
            output_ports=(PortSpec(name="y", direction="output", dimension=dim, role="controller_output"),),
        )
        act_spec = GraphNodeSpec(
            node_id=act_id,
            kind="actuator_bank",
            family=actuator_family,
            layer="actuator",
            input_ports=(PortSpec(name="u", direction="input", dimension=dim, role="actuator_input"),),
            output_ports=(PortSpec(name="y", direction="output", dimension=dim, role="actuator_output"),),
        )
        controller_nodes.append(ctrl_spec)
        actuator_nodes.append(act_spec)
        nodes.extend((ctrl_spec, act_spec))
        controller_ref_slices.append((ref_offset, dim))
        edges.append(
            GraphEdgeSpec(
                edge_id=f"edge_ref_{ctrl_id}",
                source_node="ref_0",
                source_port="y",
                target_node=ctrl_id,
                target_port="u",
                matrix=matrix_to_tuple(_selection_matrix(input_channels, ref_offset, dim)),
                signal_role="reference",
            )
        )
        edges.append(
            GraphEdgeSpec(
                edge_id=f"edge_{ctrl_id}_{act_id}",
                source_node=ctrl_id,
                source_port="y",
                target_node=act_id,
                target_port="u",
                matrix=matrix_to_tuple(np.eye(dim, dtype=float)),
                signal_role="control",
            )
        )
        taps.append(TapSpec(f"tap_{ctrl_id}", ctrl_id, "y", "controller_output", "primary", dim))
        taps.append(TapSpec(f"tap_{act_id}", act_id, "y", "actuator_output", "primary", dim))
        ref_offset += dim

    process_output_offset = 0
    for index, (input_dim, output_dim) in enumerate(zip(process_input_dims, process_output_dims, strict=False)):
        node_id = f"proc_{index}"
        spec = GraphNodeSpec(
            node_id=node_id,
            kind="process_unit",
            family=process_family,
            layer="process",
            input_ports=(PortSpec(name="u", direction="input", dimension=input_dim, role="process_input"),),
            output_ports=(PortSpec(name="y", direction="output", dimension=output_dim, role="process_output"),),
            metadata={"output_offset": process_output_offset},
        )
        process_output_offset += output_dim
        process_nodes.append(spec)
        nodes.append(spec)
        taps.append(TapSpec(f"tap_{node_id}", node_id, "y", "process_output", "primary", output_dim))

    for index, input_dim in enumerate(process_input_dims[:disturbance_count]):
        node_id = f"dist_{index}"
        spec = GraphNodeSpec(
            node_id=node_id,
            kind="disturbance_injector",
            family=disturbance_family,
            layer="disturbance",
            input_ports=(),
            output_ports=(PortSpec(name="y", direction="output", dimension=input_dim, role="disturbance"),),
        )
        disturbance_nodes.append(spec)
        nodes.append(spec)
        taps.append(TapSpec(f"tap_{node_id}", node_id, "y", "disturbance", "auxiliary", input_dim))
        edges.append(
            GraphEdgeSpec(
                edge_id=f"edge_{node_id}_{process_nodes[index].node_id}",
                source_node=node_id,
                source_port="y",
                target_node=process_nodes[index].node_id,
                target_port="u",
                matrix=matrix_to_tuple(np.eye(input_dim, dtype=float)),
                signal_role="disturbance",
            )
        )

    for index, (input_dim, output_dim) in enumerate(zip(sensor_input_dims, sensor_output_dims, strict=False)):
        node_id = f"sense_{index}"
        spec = GraphNodeSpec(
            node_id=node_id,
            kind="sensor_bank",
            family=sensor_family,
            layer="sensor",
            input_ports=(PortSpec(name="u", direction="input", dimension=input_dim, role="sensor_input"),),
            output_ports=(PortSpec(name="y", direction="output", dimension=output_dim, role="sensor_output"),),
            metadata={"output_offset": sensor_output_offsets[index]},
        )
        sensor_nodes.append(spec)
        nodes.append(spec)
        taps.append(TapSpec(f"tap_{node_id}", node_id, "y", "sensor_output", "primary", output_dim))

    for index, input_dim in enumerate(sensor_input_dims[:noise_count]):
        node_id = f"noise_{index}"
        spec = GraphNodeSpec(
            node_id=node_id,
            kind="noise_injector",
            family=noise_family,
            layer="noise",
            input_ports=(),
            output_ports=(PortSpec(name="y", direction="output", dimension=input_dim, role="noise"),),
        )
        noise_nodes.append(spec)
        nodes.append(spec)
        taps.append(TapSpec(f"tap_{node_id}", node_id, "y", "noise", "auxiliary", input_dim))
        edges.append(
            GraphEdgeSpec(
                edge_id=f"edge_{node_id}_{sensor_nodes[index].node_id}",
                source_node=node_id,
                source_port="y",
                target_node=sensor_nodes[index].node_id,
                target_port="u",
                matrix=matrix_to_tuple(np.eye(input_dim, dtype=float)),
                signal_role="noise",
            )
        )

    out_node = GraphNodeSpec(
        node_id="out_0",
        kind="output_sink",
        family="identity",
        layer="output",
        input_ports=(PortSpec(name="u", direction="input", dimension=output_channels, role="output_input"),),
        output_ports=(PortSpec(name="y", direction="output", dimension=output_channels, role="primary_output"),),
    )
    nodes.append(out_node)
    taps.append(TapSpec("tap_out_0", "out_0", "y", "primary_output", "primary", output_channels))

    for act_spec in actuator_nodes:
        act_dim = act_spec.output_ports[0].dimension
        for proc_spec, proc_input_dim in zip(process_nodes, process_input_dims, strict=False):
            edges.append(
                GraphEdgeSpec(
                    edge_id=f"edge_{act_spec.node_id}_{proc_spec.node_id}",
                    source_node=act_spec.node_id,
                    source_port="y",
                    target_node=proc_spec.node_id,
                    target_port="u",
                    matrix=matrix_to_tuple(_random_mix_matrix(proc_input_dim, act_dim, coupling_family, rng)),
                    signal_role="actuation",
                )
            )

    for proc_spec, proc_output_dim in zip(process_nodes, process_output_dims, strict=False):
        for sensor_spec, sensor_input_dim in zip(sensor_nodes, sensor_input_dims, strict=False):
            edges.append(
                GraphEdgeSpec(
                    edge_id=f"edge_{proc_spec.node_id}_{sensor_spec.node_id}",
                    source_node=proc_spec.node_id,
                    source_port="y",
                    target_node=sensor_spec.node_id,
                    target_port="u",
                    matrix=matrix_to_tuple(_random_mix_matrix(sensor_input_dim, proc_output_dim, coupling_family, rng)),
                    signal_role="measurement",
                )
            )

    for sensor_spec, output_dim, output_offset in zip(sensor_nodes, sensor_output_dims, sensor_output_offsets, strict=False):
        edges.append(
            GraphEdgeSpec(
                edge_id=f"edge_{sensor_spec.node_id}_out_0",
                source_node=sensor_spec.node_id,
                source_port="y",
                target_node="out_0",
                target_port="u",
                matrix=matrix_to_tuple(_placement_matrix(output_channels, output_offset, output_dim)),
                signal_role="primary_output",
            )
        )

    for ctrl_spec in controller_nodes:
        ctrl_dim = ctrl_spec.input_ports[0].dimension
        edges.append(
            GraphEdgeSpec(
                edge_id=f"edge_out_0_{ctrl_spec.node_id}",
                source_node="out_0",
                source_port="y",
                target_node=ctrl_spec.node_id,
                target_port="u",
                matrix=matrix_to_tuple(-_random_mix_matrix(ctrl_dim, output_channels, coupling_family, rng)),
                signal_role="feedback",
            )
        )

    metadata = {
        "topology_family": topology_family,
        "reference_family": reference_family,
        "controller_family": controller_family,
        "actuator_family": actuator_family,
        "process_family": process_family,
        "sensor_family": sensor_family,
        "disturbance_family": disturbance_family,
        "noise_family": noise_family,
        "coupling_family": coupling_family,
        "graph_size_target": target_nodes,
        "input_channels": input_channels,
        "output_channels": output_channels,
        "primary_input_node": "ref_0",
        "primary_output_node": "out_0",
        "primary_output_port": "y",
        "layer_order": LAYER_ORDER,
        "hidden_state_policy": "optional_teacher",
    }

    graph = ControlGraph(
        topology_family=topology_family,
        nodes=tuple(nodes),
        edges=tuple(edges),
        input_ports=input_ports,
        output_ports=output_ports,
        taps=tuple(taps),
        metadata=metadata,
    )
    _validate_graph(graph)
    return graph, metadata


def generate_structure(
    config: GrammarConfig,
    rng: np.random.Generator,
    difficulty: str | None = None,
    system_type: str | None = None,
    structure_family: str | None = None,
    controller_family: str | None = None,
) -> ControlGraph:
    del difficulty, system_type, structure_family, controller_family
    graph, _ = generate_graph(config, rng)
    return graph


def sample_parameters(
    graph: ControlGraph,
    config: SamplingConfig,
    rng: np.random.Generator,
    parameter_family: str | None = None,
) -> ParameterizedGraph:
    del parameter_family
    node_models: list[ParameterizedNode] = []
    module_params: dict[str, dict[str, object]] = {}

    for node in graph.nodes:
        input_dim = sum(port.dimension for port in node.input_ports)
        output_dim = sum(port.dimension for port in node.output_ports)
        dynamics: ControlNode | None = None
        nonlinearity = "none"
        saturation_limit = None
        rate_limit = None
        bias = 0.0
        params: dict[str, object] = {"kind": node.kind, "family": node.family}

        if node.kind == "controller_bank":
            dynamics, params = _sample_controller_node(node.family, input_dim, output_dim, config, rng)
        elif node.kind == "actuator_bank":
            dynamics, params, nonlinearity, saturation_limit, rate_limit = _sample_actuator_node(
                node.family,
                input_dim,
                output_dim,
                config,
                rng,
            )
        elif node.kind == "process_unit":
            dynamics, params = _sample_process_node(node.family, input_dim, output_dim, config, rng)
        elif node.kind == "sensor_bank":
            dynamics, params, bias = _sample_sensor_node(node.family, input_dim, output_dim, config, rng)
        elif node.kind == "output_sink":
            dynamics = MatrixGainNode(
                k=matrix_to_tuple(np.eye(output_dim, input_dim, dtype=float)),
                rows=output_dim,
                cols=input_dim,
                name=f"{node.node_id}_identity",
            )
            params = {"kind": node.kind, "family": "identity"}
        else:
            params["dimension"] = output_dim

        node_models.append(
            ParameterizedNode(
                spec=node,
                dynamics=dynamics,
                parameters=params,
                nonlinearity=nonlinearity,
                saturation_limit=saturation_limit,
                rate_limit=rate_limit,
                bias=bias,
            )
        )
        module_params[node.node_id] = params

    graph_metrics = _graph_metrics(graph)
    metadata = dict(graph.metadata)
    metadata["state_dimension_total"] = int(sum(_node_state_dimension(node.dynamics) for node in node_models))
    return ParameterizedGraph(
        graph=graph,
        nodes=tuple(node_models),
        module_params=module_params,
        graph_metrics=graph_metrics,
        metadata=metadata,
    )


def sample_scenario(
    graph: ControlGraph,
    dataset_config: DatasetConfig,
    rng: np.random.Generator,
    sample_seed: int,
) -> ScenarioSpec:
    reference_specs: dict[str, tuple[SignalSpec, ...]] = {}
    disturbance_specs: dict[str, tuple[SignalSpec, ...]] = {}
    noise_specs: dict[str, tuple[SignalSpec, ...]] = {}
    duration = dataset_config.sim_config.duration

    for node in graph.nodes:
        output_dim = sum(port.dimension for port in node.output_ports)
        if node.kind == "reference_source":
            reference_specs[node.node_id] = tuple(
                _sample_signal_spec(
                    node.family,
                    duration,
                    rng,
                    seed=sample_seed + 17 + index,
                    channel_index=index,
                )
                for index in range(output_dim)
            )
        elif node.kind == "disturbance_injector":
            disturbance_specs[node.node_id] = tuple(
                _sample_signal_spec(
                    node.family,
                    duration,
                    rng,
                    seed=sample_seed + 101 + index + 11 * len(disturbance_specs),
                    channel_index=index,
                )
                for index in range(output_dim)
            )
        elif node.kind == "noise_injector":
            noise_specs[node.node_id] = tuple(
                _sample_signal_spec(
                    node.family,
                    duration,
                    rng,
                    seed=sample_seed + 211 + index + 17 * len(noise_specs),
                    channel_index=index,
                )
                for index in range(output_dim)
            )

    return ScenarioSpec(
        external_inputs=reference_specs,
        disturbances=disturbance_specs,
        noises=noise_specs,
        sim_config=dataset_config.sim_config,
    )


def generate_sample(
    dataset_config: DatasetConfig,
    grammar_config: GrammarConfig | None = None,
    sampling_config: SamplingConfig | None = None,
    sample_index: int = 0,
) -> DatasetSample:
    grammar = grammar_config or GrammarConfig()
    sampling = sampling_config or SamplingConfig()
    sample_seed = int(dataset_config.seed + sample_index)
    rng = np.random.default_rng(sample_seed)
    graph, metadata = generate_graph(grammar, rng)
    parameterized_graph = sample_parameters(graph, sampling, rng)
    scenario = sample_scenario(graph, dataset_config, rng, sample_seed)
    signal_bundle = simulate_graph(parameterized_graph, scenario)
    metadata = {
        **metadata,
        "seed": sample_seed,
        "sample_id": _sample_id(graph, sample_seed),
    }
    return build_sample(parameterized_graph, signal_bundle, scenario, metadata)


def generate_dataset(
    dataset_config: DatasetConfig,
    grammar_config: GrammarConfig | None = None,
    sampling_config: SamplingConfig | None = None,
) -> list[DatasetSample]:
    return [
        generate_sample(dataset_config, grammar_config=grammar_config, sampling_config=sampling_config, sample_index=index)
        for index in range(dataset_config.count)
    ]


def iter_dataset(
    dataset_config: DatasetConfig,
    grammar_config: GrammarConfig | None = None,
    sampling_config: SamplingConfig | None = None,
) -> Iterator[DatasetSample]:
    for index in range(dataset_config.count):
        yield generate_sample(
            dataset_config,
            grammar_config=grammar_config,
            sampling_config=sampling_config,
            sample_index=index,
        )


def _validate_graph(graph: ControlGraph) -> None:
    node_map = {node.node_id: node for node in graph.nodes}
    if len(node_map) != len(graph.nodes):
        raise ValueError("Node ids must be unique")
    if not graph.output_ports:
        raise ValueError("Graph must declare at least one output port")
    for node in graph.nodes:
        if node.layer not in LAYER_ORDER:
            raise ValueError(f"Unsupported layer: {node.layer}")
        port_names = {port.name for port in (*node.input_ports, *node.output_ports)}
        if len(port_names) != len(node.input_ports) + len(node.output_ports):
            raise ValueError(f"Duplicate port names on node {node.node_id}")
        for port in (*node.input_ports, *node.output_ports):
            if port.dimension <= 0:
                raise ValueError(f"Port dimension must be positive on {node.node_id}.{port.name}")
    for edge in graph.edges:
        source_node = node_map.get(edge.source_node)
        target_node = node_map.get(edge.target_node)
        if source_node is None or target_node is None:
            raise ValueError(f"Unknown node in edge {edge.edge_id}")
        source_port = _find_port(source_node.output_ports, edge.source_port)
        target_port = _find_port(target_node.input_ports, edge.target_port)
        if source_port is None or target_port is None:
            raise ValueError(f"Unknown port in edge {edge.edge_id}")
        if edge.matrix is not None:
            rows = len(edge.matrix)
            cols = len(edge.matrix[0]) if rows else 0
            if rows != target_port.dimension or cols != source_port.dimension:
                raise ValueError(f"Edge matrix shape mismatch in {edge.edge_id}")
        source_layer = LAYER_ORDER[source_node.layer]
        target_layer = LAYER_ORDER[target_node.layer]
        if source_layer == target_layer:
            raise ValueError(f"Same-layer edges are not allowed: {edge.edge_id}")
        if source_layer > target_layer and target_node.layer != "controller":
            raise ValueError(f"Backward edges must terminate at controller nodes: {edge.edge_id}")
    tap_ids = {tap.tap_id for tap in graph.taps}
    if len(tap_ids) != len(graph.taps):
        raise ValueError("Tap ids must be unique")
    for tap in graph.taps:
        source_node = node_map.get(tap.source_node)
        if source_node is None:
            raise ValueError(f"Tap references unknown node: {tap.tap_id}")
        source_port = _find_port(source_node.output_ports, tap.source_port)
        if source_port is None:
            raise ValueError(f"Tap references unknown port: {tap.tap_id}")
        if source_port.dimension != tap.dimension:
            raise ValueError(f"Tap dimension mismatch: {tap.tap_id}")


def _graph_metrics(graph: ControlGraph) -> dict[str, object]:
    node_count = len(graph.nodes)
    edge_count = len(graph.edges)
    input_channels = sum(port.dimension for port in graph.input_ports)
    output_channels = sum(port.dimension for port in graph.output_ports)
    forward_edges = sum(
        1
        for edge in graph.edges
        if LAYER_ORDER[_node_layer(graph, edge.source_node)] < LAYER_ORDER[_node_layer(graph, edge.target_node)]
    )
    possible_forward = max(node_count * (node_count - 1) // 2, 1)
    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "input_channels": input_channels,
        "output_channels": output_channels,
        "tap_count": len(graph.taps),
        "feedback_edge_count": edge_count - forward_edges,
        "coupling_density": float(edge_count / possible_forward),
        "graph_size_bucket": _size_bucket(node_count),
    }


def _sample_controller_node(
    family: str,
    input_dim: int,
    output_dim: int,
    config: SamplingConfig,
    rng: np.random.Generator,
) -> tuple[ControlNode, dict[str, object]]:
    base_gain = _diag_dominant_matrix(output_dim, input_dim, config.controller_gain_range, rng)
    if family == "p":
        return (
            MatrixGainNode(k=matrix_to_tuple(base_gain), rows=output_dim, cols=input_dim, name="controller_gain"),
            {"family": family, "gain": matrix_to_tuple(base_gain)},
        )
    if family == "pi":
        ki = np.diag(_sample_vector(config.controller_integral_range, output_dim, rng))
        a = np.zeros((output_dim, output_dim), dtype=float)
        b = np.eye(output_dim, input_dim, dtype=float)
        c = ki
        d = base_gain
        return (
            SSNode(
                a=matrix_to_tuple(a),
                b=matrix_to_tuple(b),
                c=matrix_to_tuple(c),
                d=matrix_to_tuple(d),
                states=output_dim,
                inputs=input_dim,
                outputs=output_dim,
                name="controller_pi",
            ),
            {"family": family, "kp": matrix_to_tuple(base_gain), "ki": matrix_to_tuple(ki)},
        )
    if family == "pid":
        kp = base_gain
        ki = np.diag(_sample_vector(config.controller_integral_range, output_dim, rng))
        kd = np.diag(_sample_vector(config.controller_derivative_range, output_dim, rng))
        tau = float(_sample_range((0.03, 0.15), rng))
        a = np.block(
            [
                [np.zeros((output_dim, output_dim), dtype=float), np.zeros((output_dim, output_dim), dtype=float)],
                [np.zeros((output_dim, output_dim), dtype=float), -(1.0 / tau) * np.eye(output_dim, dtype=float)],
            ]
        )
        b = np.vstack([np.eye(output_dim, input_dim, dtype=float), (1.0 / tau) * np.eye(output_dim, input_dim, dtype=float)])
        c = np.hstack([ki, -kd])
        d = kp + kd / tau
        return (
            SSNode(
                a=matrix_to_tuple(a),
                b=matrix_to_tuple(b),
                c=matrix_to_tuple(c),
                d=matrix_to_tuple(d),
                states=2 * output_dim,
                inputs=input_dim,
                outputs=output_dim,
                name="controller_pid",
            ),
            {
                "family": family,
                "kp": matrix_to_tuple(kp),
                "ki": matrix_to_tuple(ki),
                "kd": matrix_to_tuple(kd),
                "tau": tau,
            },
        )
    return _stable_dynamic_node(
        input_dim=input_dim,
        output_dim=output_dim,
        state_dim=max(output_dim, 2),
        gain_range=config.controller_gain_range,
        tau_range=(0.05, 0.6),
        rng=rng,
        name="controller_mimo_filter",
        family=family,
    )


def _sample_actuator_node(
    family: str,
    input_dim: int,
    output_dim: int,
    config: SamplingConfig,
    rng: np.random.Generator,
) -> tuple[ControlNode, dict[str, object], str, float | None, float | None]:
    gain = _diag_dominant_matrix(output_dim, input_dim, config.actuator_gain_range, rng)
    if family == "ideal":
        return (
            MatrixGainNode(k=matrix_to_tuple(gain), rows=output_dim, cols=input_dim, name="actuator_ideal"),
            {"family": family, "gain": matrix_to_tuple(gain)},
            "none",
            None,
            None,
        )
    tau = _sample_vector(config.actuator_tau_range, output_dim, rng)
    a = -np.diag(1.0 / np.maximum(tau, 1e-6))
    b = gain / np.maximum(tau[:, None], 1e-6)
    c = np.eye(output_dim, dtype=float)
    d = np.zeros((output_dim, input_dim), dtype=float)
    nonlinearity = "none"
    saturation_limit = None
    rate_limit = None
    if family == "lag_saturation":
        nonlinearity = "saturation"
        saturation_limit = float(_sample_range(config.actuator_saturation_range, rng))
    elif family == "rate_limited":
        nonlinearity = "rate_limit"
        rate_limit = float(_sample_range(config.actuator_rate_limit_range, rng))
    return (
        SSNode(
            a=matrix_to_tuple(a),
            b=matrix_to_tuple(b),
            c=matrix_to_tuple(c),
            d=matrix_to_tuple(d),
            states=output_dim,
            inputs=input_dim,
            outputs=output_dim,
            name="actuator_lag",
        ),
        {
            "family": family,
            "gain": matrix_to_tuple(gain),
            "tau": tuple(float(value) for value in tau),
            "saturation_limit": saturation_limit,
            "rate_limit": rate_limit,
        },
        nonlinearity,
        saturation_limit,
        rate_limit,
    )


def _sample_process_node(
    family: str,
    input_dim: int,
    output_dim: int,
    config: SamplingConfig,
    rng: np.random.Generator,
) -> tuple[ControlNode, dict[str, object]]:
    if family == "first_order_bank":
        return _stable_dynamic_node(
            input_dim=input_dim,
            output_dim=output_dim,
            state_dim=output_dim,
            gain_range=config.process_gain_range,
            tau_range=config.process_tau_range,
            rng=rng,
            name="process_first_order",
            family=family,
        )
    if family == "second_order_bank":
        return _second_order_process_node(input_dim, output_dim, config, rng, family, oscillatory=False)
    if family == "oscillatory_bank":
        return _second_order_process_node(input_dim, output_dim, config, rng, family, oscillatory=True)
    return _stable_dynamic_node(
        input_dim=input_dim,
        output_dim=output_dim,
        state_dim=max(2 * output_dim, input_dim + output_dim),
        gain_range=config.process_gain_range,
        tau_range=config.process_tau_range,
        rng=rng,
        name="process_coupled",
        family=family,
        coupling_scale=0.8,
    )


def _sample_sensor_node(
    family: str,
    input_dim: int,
    output_dim: int,
    config: SamplingConfig,
    rng: np.random.Generator,
) -> tuple[ControlNode, dict[str, object], float]:
    gain = _diag_dominant_matrix(output_dim, input_dim, config.sensor_gain_range, rng)
    bias = float(_sample_range(config.sensor_bias_range, rng))
    if family == "ideal":
        return (
            MatrixGainNode(k=matrix_to_tuple(gain), rows=output_dim, cols=input_dim, name="sensor_ideal"),
            {"family": family, "gain": matrix_to_tuple(gain), "bias": bias},
            bias,
        )
    tau = _sample_vector(config.sensor_tau_range, output_dim, rng)
    a = -np.diag(1.0 / np.maximum(tau, 1e-6))
    b = gain / np.maximum(tau[:, None], 1e-6)
    c = np.eye(output_dim, dtype=float)
    d = np.zeros((output_dim, input_dim), dtype=float)
    return (
        SSNode(
            a=matrix_to_tuple(a),
            b=matrix_to_tuple(b),
            c=matrix_to_tuple(c),
            d=matrix_to_tuple(d),
            states=output_dim,
            inputs=input_dim,
            outputs=output_dim,
            name="sensor_filter",
        ),
        {"family": family, "gain": matrix_to_tuple(gain), "tau": tuple(float(value) for value in tau), "bias": bias},
        bias,
    )


def _stable_dynamic_node(
    input_dim: int,
    output_dim: int,
    state_dim: int,
    gain_range: tuple[float, float],
    tau_range: tuple[float, float],
    rng: np.random.Generator,
    name: str,
    family: str,
    coupling_scale: float = 0.45,
) -> tuple[ControlNode, dict[str, object]]:
    decay = _sample_vector(tau_range, state_dim, rng)
    a = -np.diag(1.0 / np.maximum(decay, 1e-6))
    a += rng.normal(0.0, coupling_scale / max(state_dim, 1), size=a.shape)
    eig_shift = max(np.max(np.real(np.linalg.eigvals(a))), -0.2)
    if eig_shift >= -0.05:
        a -= np.eye(state_dim, dtype=float) * (eig_shift + 0.2)
    b = rng.normal(0.0, 0.7, size=(state_dim, input_dim))
    c = rng.normal(0.0, 0.7, size=(output_dim, state_dim))
    c *= _sample_range(gain_range, rng)
    d = rng.normal(0.0, 0.05, size=(output_dim, input_dim))
    return (
        SSNode(
            a=matrix_to_tuple(a),
            b=matrix_to_tuple(b),
            c=matrix_to_tuple(c),
            d=matrix_to_tuple(d),
            states=state_dim,
            inputs=input_dim,
            outputs=output_dim,
            name=name,
        ),
        {"family": family, "state_dim": state_dim},
    )


def _second_order_process_node(
    input_dim: int,
    output_dim: int,
    config: SamplingConfig,
    rng: np.random.Generator,
    family: str,
    oscillatory: bool,
) -> tuple[ControlNode, dict[str, object]]:
    wn = _sample_vector(config.process_wn_range, output_dim, rng)
    zeta = _sample_vector(config.process_zeta_range, output_dim, rng)
    blocks = []
    for omega, damping in zip(wn, zeta, strict=False):
        if oscillatory:
            damping = min(damping, 0.45)
        blocks.append(np.array([[0.0, 1.0], [-(omega**2), -2.0 * damping * omega]], dtype=float))
    a = _block_diag(blocks)
    b = np.zeros((2 * output_dim, input_dim), dtype=float)
    for index in range(output_dim):
        b[2 * index + 1, :] = rng.normal(0.0, 0.8, size=input_dim)
    c = np.zeros((output_dim, 2 * output_dim), dtype=float)
    for index in range(output_dim):
        c[index, 2 * index] = _sample_range(config.process_gain_range, rng)
    d = rng.normal(0.0, 0.02, size=(output_dim, input_dim))
    return (
        SSNode(
            a=matrix_to_tuple(a),
            b=matrix_to_tuple(b),
            c=matrix_to_tuple(c),
            d=matrix_to_tuple(d),
            states=2 * output_dim,
            inputs=input_dim,
            outputs=output_dim,
            name="process_second_order",
        ),
        {
            "family": family,
            "wn": tuple(float(value) for value in wn),
            "zeta": tuple(float(value) for value in zeta),
        },
    )


def _sample_signal_spec(
    family: str,
    duration: float,
    rng: np.random.Generator,
    seed: int,
    channel_index: int,
) -> SignalSpec:
    amplitude = float(0.6 + 0.5 * rng.random() + 0.15 * channel_index)
    start_time = float(rng.uniform(0.05 * duration, 0.35 * duration))
    if family == "step":
        return SignalSpec(kind="step", amplitude=amplitude, start_time=start_time, seed=seed)
    if family == "multistep":
        levels = tuple(float(level) for level in rng.uniform(-amplitude, amplitude, size=4))
        return SignalSpec(kind="multistep", amplitude=amplitude, start_time=start_time, levels=levels, seed=seed)
    if family == "ramp":
        return SignalSpec(kind="ramp", amplitude=amplitude / max(duration, 1.0), start_time=start_time, seed=seed)
    if family == "chirp":
        return SignalSpec(
            kind="chirp",
            amplitude=amplitude,
            frequency=float(rng.uniform(0.1, 0.6)),
            start_time=start_time,
            seed=seed,
        )
    if family == "prbs":
        return SignalSpec(
            kind="prbs",
            amplitude=amplitude,
            start_time=start_time,
            duration=float(rng.uniform(0.15, 0.6)),
            seed=seed,
        )
    if family == "load_step":
        return SignalSpec(kind="load_step", amplitude=amplitude, start_time=start_time, seed=seed)
    if family == "load_pulse":
        return SignalSpec(
            kind="load_pulse",
            amplitude=amplitude,
            start_time=start_time,
            duration=float(rng.uniform(0.4, 0.9)),
            seed=seed,
        )
    if family == "filtered_noise":
        return SignalSpec(
            kind="filtered_noise",
            amplitude=amplitude * 0.4,
            start_time=start_time,
            bandwidth=float(rng.uniform(0.1, 0.8)),
            seed=seed,
        )
    if family == "lowpass_noise":
        return SignalSpec(
            kind="lowpass_noise",
            amplitude=amplitude * 0.05,
            bandwidth=float(rng.uniform(0.2, 1.0)),
            seed=seed,
        )
    return SignalSpec(kind="white_noise", amplitude=amplitude * 0.05, seed=seed)


def _choose_weighted(weights: dict[str, float], rng: np.random.Generator) -> str:
    keys = tuple(weights.keys())
    values = np.asarray(tuple(weights.values()), dtype=float)
    if np.any(values < 0):
        raise ValueError("Weights must be non-negative")
    total = np.sum(values)
    if total <= 0:
        raise ValueError("At least one weight must be positive")
    probs = values / total
    return str(rng.choice(keys, p=probs))


def _partition_total(total: int, parts: int, rng: np.random.Generator) -> list[int]:
    if parts <= 0:
        raise ValueError("parts must be positive")
    if total < parts:
        return [1] * total + [1] * max(parts - total, 0)
    cuts = sorted(rng.choice(np.arange(1, total), size=parts - 1, replace=False).tolist()) if parts > 1 else []
    values = []
    previous = 0
    for cut in (*cuts, total):
        values.append(cut - previous)
        previous = cut
    return [int(max(value, 1)) for value in values]


def _selection_matrix(total_dim: int, start: int, width: int) -> np.ndarray:
    matrix = np.zeros((width, total_dim), dtype=float)
    matrix[np.arange(width), np.arange(start, start + width)] = 1.0
    return matrix


def _placement_matrix(total_dim: int, start: int, width: int) -> np.ndarray:
    matrix = np.zeros((total_dim, width), dtype=float)
    matrix[np.arange(start, start + width), np.arange(width)] = 1.0
    return matrix


def _random_mix_matrix(
    output_dim: int,
    input_dim: int,
    coupling_family: str,
    rng: np.random.Generator,
) -> np.ndarray:
    density = {"weak": 0.35, "mixed": 0.60, "strong": 0.90}[coupling_family]
    scale = {"weak": 0.45, "mixed": 0.70, "strong": 0.95}[coupling_family]
    matrix = rng.normal(0.0, scale, size=(output_dim, input_dim))
    mask = rng.random((output_dim, input_dim)) < density
    matrix *= mask
    for row in range(output_dim):
        if not np.any(mask[row]):
            column = int(rng.integers(0, input_dim))
            matrix[row, column] = rng.normal(1.0, 0.1)
    return matrix


def _diag_dominant_matrix(
    output_dim: int,
    input_dim: int,
    gain_range: tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    matrix = rng.normal(0.0, 0.08, size=(output_dim, input_dim))
    for row in range(output_dim):
        matrix[row, row % input_dim] += _sample_range(gain_range, rng)
    return matrix


def _sample_vector(
    value_range: tuple[float, float],
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    return rng.uniform(value_range[0], value_range[1], size=count)


def _sample_range(value_range: tuple[float, float], rng: np.random.Generator) -> float:
    return float(rng.uniform(value_range[0], value_range[1]))


def _offsets(dimensions: list[int]) -> list[int]:
    offsets: list[int] = []
    cursor = 0
    for dimension in dimensions:
        offsets.append(cursor)
        cursor += dimension
    return offsets


def _find_port(ports: tuple[PortSpec, ...], port_name: str) -> PortSpec | None:
    for port in ports:
        if port.name == port_name:
            return port
    return None


def _node_layer(graph: ControlGraph, node_id: str) -> str:
    for node in graph.nodes:
        if node.node_id == node_id:
            return node.layer
    raise KeyError(node_id)


def _node_state_dimension(node: ControlNode | None) -> int:
    if isinstance(node, SSNode):
        return int(node.states or 0)
    return 0


def _sample_id(graph: ControlGraph, seed: int) -> str:
    raw = f"{seed}:{graph.topology_family}:{len(graph.nodes)}:{len(graph.edges)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _size_bucket(node_count: int) -> str:
    if node_count <= 12:
        return "small"
    if node_count <= 18:
        return "medium"
    return "large"


def _block_diag(blocks: list[np.ndarray]) -> np.ndarray:
    if not blocks:
        return np.zeros((0, 0), dtype=float)
    size = sum(block.shape[0] for block in blocks)
    result = np.zeros((size, size), dtype=float)
    cursor = 0
    for block in blocks:
        width = block.shape[0]
        result[cursor : cursor + width, cursor : cursor + width] = block
        cursor += width
    return result
