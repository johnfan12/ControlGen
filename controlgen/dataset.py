from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from controlgen.dsl import ast_to_dict
from controlgen.simulate import Trajectory
from controlgen.transfer_function import ParameterizedSystem
from controlgen.types import FeedbackNode, MatrixGainNode, PIDNode, SSNode


@dataclass(frozen=True)
class DatasetSample:
    sample_id: str
    dsl_text: str
    ast: dict[str, object]
    system_type: str
    domain: str
    input_channels: int
    output_channels: int
    input_spec: dict[str, object]
    t: list[float]
    u: list[list[float]]
    y: list[list[float]]
    system_view: dict[str, object]
    metrics: dict[str, object]
    tags: dict[str, object]
    split_tags: dict[str, object]
    seed: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_sample(
    system: ParameterizedSystem,
    trajectory: Trajectory,
    metadata: dict[str, object],
) -> DatasetSample:
    ast_dict = ast_to_dict(system.ast)
    seed = int(metadata["seed"])
    system_type = str(metadata.get("system_type", "control_siso"))
    structure_family = str(metadata.get("structure_family", "unknown"))
    parameter_family = str(metadata.get("parameter_family", "unknown"))
    input_family = str(metadata.get("input_family", trajectory.input_spec.pattern_family))
    controller_family = str(metadata.get("controller_family", _controller_type(system.ast)))
    coupling_level = str(metadata.get("coupling_level", "none"))
    tags = {
        "closed_loop": isinstance(system.ast, FeedbackNode),
        "controller_type": _controller_type(system.ast),
        "controller_family": controller_family,
        "structure_family": structure_family,
        "parameter_family": parameter_family,
        "input_family": input_family,
        "io_shape": f"{system.input_channels}x{system.output_channels}",
        "stable": system.is_stable,
        "state_dimension": system.state_dimension,
        "coupling_level": coupling_level,
    }
    split_tags = {
        "structure_family": structure_family,
        "parameter_family": parameter_family,
        "input_family": input_family,
        "io_shape": f"{system.input_channels}x{system.output_channels}",
        "system_type": system_type,
    }
    return DatasetSample(
        sample_id=str(metadata["sample_id"]),
        dsl_text=system.dsl_text,
        ast=ast_dict,
        system_type=system_type,
        domain=system.domain,
        input_channels=system.input_channels,
        output_channels=system.output_channels,
        input_spec=asdict(trajectory.input_spec),
        t=trajectory.t.tolist(),
        u=trajectory.u.tolist(),
        y=trajectory.y.tolist(),
        system_view=_system_view(system, ast_dict),
        metrics=_compute_metrics(system, trajectory),
        tags=tags,
        split_tags=split_tags,
        seed=seed,
    )


def _system_view(system: ParameterizedSystem, ast_dict: dict[str, object]) -> dict[str, object]:
    transfer = None
    if system.transfer_function is not None:
        transfer = {
            "num": list(system.transfer_function.num),
            "den": list(system.transfer_function.den),
        }
    return {
        "dsl_text": system.dsl_text,
        "ast": ast_dict,
        "state_space": {
            "a": system.a.tolist(),
            "b": system.b.tolist(),
            "c": system.c.tolist(),
            "d": system.d.tolist(),
        },
        "transfer_function": transfer,
    }


def _controller_type(node: object) -> str:
    if isinstance(node, PIDNode):
        return "pid"
    if isinstance(node, MatrixGainNode):
        return "matrix_gain"
    if isinstance(node, SSNode) and "controller" in node.name:
        return "state_space_controller"
    if isinstance(node, FeedbackNode):
        return _controller_type(node.forward)
    if hasattr(node, "blocks"):
        for block in getattr(node, "blocks"):
            controller = _controller_type(block)
            if controller != "none":
                return controller
    return "none"


def _compute_metrics(system: ParameterizedSystem, trajectory: Trajectory) -> dict[str, object]:
    poles = [
        {"real": float(np.real(pole)), "imag": float(np.imag(pole))}
        for pole in system.poles
    ]
    zeros = [
        {"real": float(np.real(zero)), "imag": float(np.imag(zero))}
        for zero in system.zeros
    ]
    dc_gain = np.asarray(system.dc_gain, dtype=float)
    metrics: dict[str, object] = {
        "system_metrics": {
            "stable": system.is_stable,
            "poles": poles,
            "zeros": zeros,
            "spectral_radius": system.spectral_radius,
            "state_dimension": system.state_dimension,
            "input_channels": system.input_channels,
            "output_channels": system.output_channels,
            "dc_gain": dc_gain.tolist(),
        },
        "channel_metrics": [
            _channel_metrics(trajectory.t, trajectory.u, trajectory.y[:, idx], idx)
            for idx in range(trajectory.y.shape[1])
        ],
    }
    return metrics


def _channel_metrics(
    t: np.ndarray,
    u: np.ndarray,
    y_channel: np.ndarray,
    channel_index: int,
) -> dict[str, object]:
    target_source = u[:, min(channel_index, u.shape[1] - 1)] if u.shape[1] else np.zeros_like(y_channel)
    target = float(target_source[-1]) if len(target_source) else 0.0
    final_value = float(y_channel[-1]) if len(y_channel) else 0.0
    peak = float(np.max(y_channel)) if len(y_channel) else 0.0
    overshoot = 0.0
    if abs(target) > 1e-8:
        overshoot = max(peak - target, 0.0) / abs(target)
    settling_time = float(t[-1]) if len(t) else 0.0
    if len(y_channel) > 0:
        band = 0.02 * max(abs(final_value), 1e-6)
        outside = np.where(np.abs(y_channel - final_value) > band)[0]
        if len(outside) == 0:
            settling_time = 0.0
        elif outside[-1] + 1 < len(t):
            settling_time = float(t[outside[-1] + 1])
    return {
        "channel": channel_index,
        "final_value": final_value,
        "peak": peak,
        "overshoot": float(overshoot),
        "steady_state_error": float(target - final_value),
        "settling_time": settling_time,
        "rms": float(np.sqrt(np.mean(np.square(y_channel)))) if len(y_channel) else 0.0,
    }
