from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from controlgen.dsl import ast_to_dict, serialize_dsl
from controlgen.simulate import ScenarioSpec, SignalBundle
from controlgen.transfer_function import block_diagonal_systems, tf_from_node
from controlgen.types import ControlNode, ParameterizedGraph, ParameterizedNode


@dataclass(frozen=True)
class DatasetSample:
    sample_id: str
    graph_dsl: str
    graph_ast: dict[str, object]
    graph_nodes: list[dict[str, object]]
    graph_edges: list[dict[str, object]]
    io_ports: dict[str, list[dict[str, object]]]
    tap_specs: list[dict[str, object]]
    module_params: dict[str, dict[str, object]]
    scenario: dict[str, object]
    t: list[float]
    signals: dict[str, object]
    teacher_signals: dict[str, list[list[float]]] = field(default_factory=dict)
    system_view: dict[str, object] = field(default_factory=dict)
    metrics: dict[str, object] = field(default_factory=dict)
    tags: dict[str, object] = field(default_factory=dict)
    split_tags: dict[str, object] = field(default_factory=dict)
    seed: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "graph_dsl": self.graph_dsl,
            "graph_ast": self.graph_ast,
            "graph_nodes": self.graph_nodes,
            "graph_edges": self.graph_edges,
            "io_ports": self.io_ports,
            "tap_specs": self.tap_specs,
            "module_params": self.module_params,
            "scenario": self.scenario,
            "t": self.t,
            "signals": self.signals,
            "teacher_signals": self.teacher_signals,
            "system_view": self.system_view,
            "metrics": self.metrics,
            "tags": self.tags,
            "split_tags": self.split_tags,
            "seed": self.seed,
        }


def build_sample(
    parameterized_graph: ParameterizedGraph,
    signal_bundle: SignalBundle,
    scenario: ScenarioSpec,
    metadata: dict[str, object],
) -> DatasetSample:
    graph = parameterized_graph.graph
    graph_ast = ast_to_dict(graph) or {}
    graph_dsl = serialize_dsl(graph)
    tags = {
        "topology_family": graph.topology_family,
        "controller_family": graph.metadata.get("controller_family"),
        "actuator_family": graph.metadata.get("actuator_family"),
        "process_family": graph.metadata.get("process_family"),
        "sensor_family": graph.metadata.get("sensor_family"),
        "disturbance_family": graph.metadata.get("disturbance_family"),
        "noise_family": graph.metadata.get("noise_family"),
        "reference_family": graph.metadata.get("reference_family"),
        "coupling_family": graph.metadata.get("coupling_family"),
        "closed_loop": True,
        "io_shape": f"{parameterized_graph.graph_metrics['input_channels']}x{parameterized_graph.graph_metrics['output_channels']}",
    }
    split_tags = {
        "graph_size_bucket": parameterized_graph.graph_metrics["graph_size_bucket"],
        "topology_family": graph.topology_family,
        "coupling_family": graph.metadata.get("coupling_family"),
        "controller_family": graph.metadata.get("controller_family"),
        "process_family": graph.metadata.get("process_family"),
        "disturbance_family": graph.metadata.get("disturbance_family"),
        "noise_family": graph.metadata.get("noise_family"),
    }
    return DatasetSample(
        sample_id=str(metadata["sample_id"]),
        graph_dsl=graph_dsl,
        graph_ast=graph_ast,
        graph_nodes=graph_ast.get("nodes", []),  # type: ignore[assignment]
        graph_edges=graph_ast.get("edges", []),  # type: ignore[assignment]
        io_ports={
            "input_ports": graph_ast.get("input_ports", []),  # type: ignore[assignment]
            "output_ports": graph_ast.get("output_ports", []),  # type: ignore[assignment]
        },
        tap_specs=graph_ast.get("taps", []),  # type: ignore[assignment]
        module_params=parameterized_graph.module_params,
        scenario=scenario.as_serializable(),
        t=signal_bundle.t.tolist(),
        signals=_signals_to_serializable(signal_bundle),
        teacher_signals={name: values.tolist() for name, values in signal_bundle.teacher_signals.items()},
        system_view=_build_system_view(parameterized_graph, graph_ast),
        metrics=_compute_metrics(parameterized_graph, signal_bundle),
        tags=tags,
        split_tags=split_tags,
        seed=int(metadata["seed"]),
    )


def _signals_to_serializable(signal_bundle: SignalBundle) -> dict[str, object]:
    return {
        "external_inputs": {name: values.tolist() for name, values in signal_bundle.external_inputs.items()},
        "primary_outputs": {name: values.tolist() for name, values in signal_bundle.primary_outputs.items()},
        "tap_signals": {name: values.tolist() for name, values in signal_bundle.tap_signals.items()},
        "disturbance_signals": {name: values.tolist() for name, values in signal_bundle.disturbance_signals.items()},
        "noise_signals": {name: values.tolist() for name, values in signal_bundle.noise_signals.items()},
    }


def _build_system_view(
    parameterized_graph: ParameterizedGraph,
    graph_ast: dict[str, object],
) -> dict[str, object]:
    node_models = {}
    systems = []
    for node in parameterized_graph.nodes:
        if node.dynamics is None:
            node_models[node.spec.node_id] = {
                "kind": node.spec.kind,
                "family": node.spec.family,
                "dynamics": None,
                "nonlinearity": node.nonlinearity,
                "bias": node.bias,
            }
            continue
        state_space = _state_space_view(node.dynamics)
        node_models[node.spec.node_id] = {
            "kind": node.spec.kind,
            "family": node.spec.family,
            "dynamics": state_space,
            "nonlinearity": node.nonlinearity,
            "saturation_limit": node.saturation_limit,
            "rate_limit": node.rate_limit,
            "bias": node.bias,
        }
        systems.append(tf_from_node(node.dynamics))
    assembled = None
    if systems:
        block = block_diagonal_systems(systems)
        assembled = {
            "state_dimension": block.state_dimension,
            "a": block.a.tolist(),
            "b": block.b.tolist(),
            "c": block.c.tolist(),
            "d": block.d.tolist(),
        }
    return {
        "graph_ast": graph_ast,
        "node_models": node_models,
        "assembled_linear_core": assembled,
        "optional_hidden_states": {
            "enabled": True,
            "policy": "optional_teacher",
            "signals": [f"{node.spec.node_id}:state" for node in parameterized_graph.nodes if _node_state_dimension(node) > 0],
        },
    }


def _state_space_view(node: ControlNode) -> dict[str, object]:
    system = tf_from_node(node)
    return {
        "dsl": serialize_dsl(node),
        "a": system.a.tolist(),
        "b": system.b.tolist(),
        "c": system.c.tolist(),
        "d": system.d.tolist(),
        "input_channels": system.input_channels,
        "output_channels": system.output_channels,
        "state_dimension": system.state_dimension,
    }


def _compute_metrics(parameterized_graph: ParameterizedGraph, signal_bundle: SignalBundle) -> dict[str, object]:
    system_metrics = {
        "input_channels": parameterized_graph.graph_metrics["input_channels"],
        "output_channels": parameterized_graph.graph_metrics["output_channels"],
        "total_state_dimension": int(sum(_node_state_dimension(node) for node in parameterized_graph.nodes)),
        "tap_count": parameterized_graph.graph_metrics["tap_count"],
        "coupling_density": parameterized_graph.graph_metrics["coupling_density"],
    }
    primary = next(iter(signal_bundle.primary_outputs.values()))
    reference = next(iter(signal_bundle.external_inputs.values()))
    tracking = _tracking_metrics(reference, primary)
    channel_metrics = {
        "primary_outputs": {
            name: _channel_summary(values)
            for name, values in signal_bundle.primary_outputs.items()
        },
        "tap_signals": {
            name: _channel_summary(values)
            for name, values in signal_bundle.tap_signals.items()
        },
    }
    return {
        "graph_metrics": parameterized_graph.graph_metrics,
        "system_metrics": system_metrics,
        "tracking_metrics": tracking,
        "channel_metrics": channel_metrics,
    }


def _tracking_metrics(reference: np.ndarray, output: np.ndarray) -> dict[str, object]:
    channels = min(reference.shape[1], output.shape[1]) if reference.ndim == 2 and output.ndim == 2 else 0
    if channels == 0:
        return {"dimension": 0, "error_rms": [], "output_final": [], "reference_final": []}
    error = output[:, :channels] - reference[:, :channels]
    return {
        "dimension": channels,
        "error_rms": _vector_rms(error),
        "output_final": [float(value) for value in output[-1, :channels]],
        "reference_final": [float(value) for value in reference[-1, :channels]],
        "peak_error": [float(value) for value in np.max(np.abs(error), axis=0)],
    }


def _channel_summary(values: np.ndarray) -> dict[str, object]:
    if values.ndim == 1:
        values = values[:, None]
    return {
        "dimension": int(values.shape[1]),
        "rms": _vector_rms(values),
        "peak_abs": [float(value) for value in np.max(np.abs(values), axis=0)],
        "final": [float(value) for value in values[-1]],
    }


def _vector_rms(values: np.ndarray) -> list[float]:
    if values.ndim == 1:
        values = values[:, None]
    return [float(np.sqrt(np.mean(np.square(values[:, idx])))) for idx in range(values.shape[1])]


def _node_state_dimension(node: ParameterizedNode) -> int:
    if node.dynamics is None:
        return 0
    system = tf_from_node(node.dynamics)
    return int(system.state_dimension)
