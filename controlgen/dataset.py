from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from controlgen.dsl import ast_to_dict
from controlgen.simulate import Trajectory
from controlgen.transfer_function import ParameterizedSystem
from controlgen.types import FeedbackNode, PIDNode


@dataclass(frozen=True)
class DatasetSample:
    sample_id: str
    dsl_text: str
    ast: dict[str, object]
    system_type: str
    domain: str
    input_spec: dict[str, object]
    t: list[float]
    u: list[float]
    y: list[float]
    metrics: dict[str, object]
    tags: dict[str, object]
    seed: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_sample(
    system: ParameterizedSystem,
    trajectory: Trajectory,
    metadata: dict[str, object],
) -> DatasetSample:
    seed = int(metadata["seed"])
    return DatasetSample(
        sample_id=str(metadata["sample_id"]),
        dsl_text=system.dsl_text,
        ast=ast_to_dict(system.ast),
        system_type=str(metadata.get("system_type", "control_siso")),
        domain=system.domain,
        input_spec=asdict(trajectory.input_spec),
        t=trajectory.t.tolist(),
        u=trajectory.u.tolist(),
        y=trajectory.y.tolist(),
        metrics=_compute_metrics(system, trajectory),
        tags=_derive_tags(system),
        seed=seed,
    )


def _derive_tags(system: ParameterizedSystem) -> dict[str, object]:
    ast = system.ast
    return {
        "closed_loop": isinstance(ast, FeedbackNode),
        "controller_type": _controller_type(ast),
        "order": max(len(system.transfer_function.den) - 1, 0),
        "stable": system.is_stable,
        "pole_count": len(system.poles),
        "zero_count": len(system.zeros),
    }


def _controller_type(node: object) -> str:
    if isinstance(node, PIDNode):
        return "pid"
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
    metrics: dict[str, object] = {
        "stable": system.is_stable,
        "poles": poles,
        "zeros": zeros,
        "final_value": float(trajectory.y[-1]),
    }
    if trajectory.input_spec.kind == "step":
        metrics.update(_step_metrics(trajectory.t, trajectory.y, trajectory.u))
    return metrics


def _step_metrics(t: np.ndarray, y: np.ndarray, u: np.ndarray) -> dict[str, object]:
    target = float(u[-1]) if len(u) else 0.0
    final_value = float(y[-1]) if len(y) else 0.0
    peak = float(np.max(y)) if len(y) else 0.0
    overshoot = 0.0
    if abs(target) > 1e-8:
        overshoot = max(peak - target, 0.0) / abs(target)
    error = float(target - final_value)
    settling_time = float(t[-1]) if len(t) else 0.0
    if len(y) > 0:
        band = 0.02 * max(abs(target), 1e-6)
        outside = np.where(np.abs(y - final_value) > band)[0]
        if len(outside) == 0:
            settling_time = 0.0
        elif outside[-1] + 1 < len(t):
            settling_time = float(t[outside[-1] + 1])
    return {
        "steady_state_error": error,
        "overshoot": float(overshoot),
        "peak": peak,
        "settling_time": settling_time,
    }
