from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from controlgen.dsl import ast_to_dict, serialize_dsl
from controlgen.simulate import ScenarioSpec, SignalBundle
from controlgen.transfer_function import tf_from_node
from controlgen.types import ControlNode, ParameterizedGraph


@dataclass(frozen=True)
class DatasetSample:
    sample_id: str
    graph_dsl: str
    graph_ast: dict[str, object]
    module_params: dict[str, dict[str, object]]
    scenario: dict[str, object]
    t: list[float]
    signals: dict[str, list[float]]
    system_view: dict[str, object]
    metrics: dict[str, object]
    tags: dict[str, object]
    split_tags: dict[str, object]
    seed: int

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "graph_dsl": self.graph_dsl,
            "graph_ast": self.graph_ast,
            "module_params": self.module_params,
            "scenario": self.scenario,
            "t": self.t,
            "signals": self.signals,
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
    graph_ast = ast_to_dict(parameterized_graph.graph)
    graph_dsl = serialize_dsl(parameterized_graph.graph)
    metrics = _compute_metrics(parameterized_graph, signal_bundle)
    tags = {
        "structure_family": parameterized_graph.structure_family,
        "controller_family": parameterized_graph.controller_family,
        "actuator_family": parameterized_graph.actuator_family,
        "plant_family": parameterized_graph.plant_family,
        "sensor_family": parameterized_graph.sensor_family,
        "disturbance_family": parameterized_graph.disturbance_family,
        "noise_family": parameterized_graph.noise_family,
        "reference_family": parameterized_graph.graph.reference.kind,
        "realism_level": "linear_with_disturbance_and_limits",
        "closed_loop": True,
    }
    split_tags = {
        "structure_family": parameterized_graph.structure_family,
        "controller_family": parameterized_graph.controller_family,
        "plant_family": parameterized_graph.plant_family,
        "disturbance_family": parameterized_graph.disturbance_family,
        "reference_family": parameterized_graph.graph.reference.kind,
    }
    return DatasetSample(
        sample_id=str(metadata["sample_id"]),
        graph_dsl=graph_dsl,
        graph_ast=graph_ast or {},
        module_params=parameterized_graph.module_params,
        scenario=scenario.as_serializable(),
        t=signal_bundle.t.tolist(),
        signals={name: values.tolist() for name, values in signal_bundle.signals.items()},
        system_view=_build_system_view(parameterized_graph, graph_ast or {}),
        metrics=metrics,
        tags=tags,
        split_tags=split_tags,
        seed=int(metadata["seed"]),
    )


def _build_system_view(
    parameterized_graph: ParameterizedGraph,
    graph_ast: dict[str, object],
) -> dict[str, object]:
    module_state_space = {}
    for name, node in (
        ("controller", parameterized_graph.controller_dynamics),
        ("actuator", parameterized_graph.actuator_dynamics),
        ("plant_control", parameterized_graph.plant_control_dynamics),
        ("plant_disturbance", parameterized_graph.plant_disturbance_dynamics),
        ("sensor", parameterized_graph.sensor_dynamics),
    ):
        if node is None:
            module_state_space[name] = None
            continue
        module_state_space[name] = _state_space_view(node)
    nonlinear_wrappers = {
        "actuator_saturation": {
            "enabled": parameterized_graph.actuator_saturation_limit is not None,
            "limit": parameterized_graph.actuator_saturation_limit,
        }
    }
    return {
        "graph_ast": graph_ast,
        "state_space_linear_core": module_state_space,
        "nonlinear_wrappers": nonlinear_wrappers,
    }


def _state_space_view(node: ControlNode) -> dict[str, object]:
    system = tf_from_node(node)
    return {
        "dsl": serialize_dsl(node),
        "a": system.a.tolist(),
        "b": system.b.tolist(),
        "c": system.c.tolist(),
        "d": system.d.tolist(),
    }


def _compute_metrics(parameterized_graph: ParameterizedGraph, signal_bundle: SignalBundle) -> dict[str, object]:
    signals = signal_bundle.signals
    t = signal_bundle.t
    r = np.asarray(signals["r"], dtype=float)
    d = np.asarray(signals["d"], dtype=float)
    e = np.asarray(signals["e"], dtype=float)
    u_cmd = np.asarray(signals["u_cmd"], dtype=float)
    u_act = np.asarray(signals["u_act"], dtype=float)
    y = np.asarray(signals["y"], dtype=float)
    y_m = np.asarray(signals["y_m"], dtype=float)
    n = np.asarray(signals["n"], dtype=float)

    tracking = _tracking_metrics(t, r, y, e)
    control_effort = {
        "u_cmd_rms": _rms(u_cmd),
        "u_act_rms": _rms(u_act),
        "u_cmd_peak": float(np.max(np.abs(u_cmd))) if len(u_cmd) else 0.0,
        "u_act_peak": float(np.max(np.abs(u_act))) if len(u_act) else 0.0,
        "saturation_ratio": float(np.mean(np.abs(u_act - u_cmd) > 1e-6)) if len(u_act) else 0.0,
    }
    disturbance_metrics = _disturbance_metrics(t, r, d, y)
    measurement_metrics = {
        "measurement_noise_rms": _rms(y_m - y),
        "sensor_bias": parameterized_graph.sensor_bias,
        "noise_signal_rms": _rms(n),
    }
    return {
        "tracking": tracking,
        "control_effort": control_effort,
        "disturbance_rejection": disturbance_metrics,
        "measurement": measurement_metrics,
    }


def _tracking_metrics(t: np.ndarray, r: np.ndarray, y: np.ndarray, e: np.ndarray) -> dict[str, float]:
    final_ref = float(r[-1]) if len(r) else 0.0
    final_output = float(y[-1]) if len(y) else 0.0
    steady_state_error = final_ref - final_output
    peak = float(np.max(y)) if len(y) else 0.0
    overshoot = 0.0
    if abs(final_ref) > 1e-8:
        overshoot = max(peak - final_ref, 0.0) / abs(final_ref)
    settling_time = float(t[-1]) if len(t) else 0.0
    if len(y):
        band = 0.02 * max(abs(final_ref), 1e-6)
        outside = np.where(np.abs(y - final_ref) > band)[0]
        if len(outside) == 0:
            settling_time = 0.0
        elif outside[-1] + 1 < len(t):
            settling_time = float(t[outside[-1] + 1])
    return {
        "steady_state_error": float(steady_state_error),
        "overshoot": float(overshoot),
        "settling_time": float(settling_time),
        "error_rms": _rms(e),
        "output_final": final_output,
    }


def _disturbance_metrics(t: np.ndarray, r: np.ndarray, d: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    active = np.where(np.abs(d) > 1e-9)[0]
    if len(active) == 0:
        return {
            "disturbance_present": False,
            "disturbance_onset": None,
            "max_deviation_after_onset": None,
            "recovery_time": None,
        }
    onset = int(active[0])
    baseline = float(y[onset - 1]) if onset > 0 else float(y[0])
    deviation = np.abs(y[onset:] - baseline)
    max_deviation = float(np.max(deviation)) if len(deviation) else 0.0
    recovery_time = float(t[-1]) if len(t) else 0.0
    reference_tail = r[onset:]
    band = 0.05 * max(abs(reference_tail[-1]) if len(reference_tail) else 0.0, 1e-6)
    recovered = np.where(np.abs(y[onset:] - reference_tail) <= band)[0]
    if len(recovered):
        recovery_time = float(t[onset + recovered[0]] - t[onset])
    return {
        "disturbance_present": True,
        "disturbance_onset": float(t[onset]),
        "max_deviation_after_onset": max_deviation,
        "recovery_time": recovery_time,
    }


def _rms(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(values))))
