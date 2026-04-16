from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union

Matrix = tuple[tuple[float, ...], ...]


def _default_name(prefix: str) -> str:
    return prefix


@dataclass(frozen=True)
class TFNode:
    num: tuple[float, ...] | None = None
    den: tuple[float, ...] | None = None
    name: str = field(default_factory=lambda: _default_name("tf"))


@dataclass(frozen=True)
class GainNode:
    k: float | None = None
    name: str = field(default_factory=lambda: _default_name("gain"))


@dataclass(frozen=True)
class MatrixGainNode:
    k: Matrix | None = None
    rows: int | None = None
    cols: int | None = None
    name: str = field(default_factory=lambda: _default_name("matgain"))


@dataclass(frozen=True)
class PIDNode:
    kp: float | None = None
    ki: float | None = None
    kd: float | None = None
    tau: float = 0.05
    name: str = field(default_factory=lambda: _default_name("pid"))


@dataclass(frozen=True)
class DelayNode:
    t: float | None = None
    order: int = 1
    name: str = field(default_factory=lambda: _default_name("delay"))


@dataclass(frozen=True)
class SSNode:
    a: Matrix | None = None
    b: Matrix | None = None
    c: Matrix | None = None
    d: Matrix | None = None
    states: int | None = None
    inputs: int | None = None
    outputs: int | None = None
    name: str = field(default_factory=lambda: _default_name("ss"))


@dataclass(frozen=True)
class SeriesNode:
    blocks: tuple["ControlNode", ...]


@dataclass(frozen=True)
class ParallelNode:
    blocks: tuple["ControlNode", ...]


@dataclass(frozen=True)
class FeedbackNode:
    forward: "ControlNode"
    feedback: "ControlNode"
    sign: int = -1


ControlNode = Union[
    TFNode,
    GainNode,
    MatrixGainNode,
    PIDNode,
    DelayNode,
    SSNode,
    SeriesNode,
    ParallelNode,
    FeedbackNode,
]


@dataclass(frozen=True)
class ReferenceNode:
    kind: str = "step"
    name: str = field(default_factory=lambda: _default_name("reference"))


@dataclass(frozen=True)
class SumNode:
    signs: tuple[int, ...] = (1, -1)
    name: str = field(default_factory=lambda: _default_name("sum"))


@dataclass(frozen=True)
class ControllerNode:
    kind: str = "pid"
    name: str = field(default_factory=lambda: _default_name("controller"))


@dataclass(frozen=True)
class ActuatorNode:
    kind: str = "ideal"
    name: str = field(default_factory=lambda: _default_name("actuator"))


@dataclass(frozen=True)
class PlantNode:
    kind: str = "first_order"
    name: str = field(default_factory=lambda: _default_name("plant"))


@dataclass(frozen=True)
class DisturbanceNode:
    kind: str = "load_step"
    injection: str = "output"
    name: str = field(default_factory=lambda: _default_name("disturbance"))


@dataclass(frozen=True)
class SensorNode:
    kind: str = "ideal"
    name: str = field(default_factory=lambda: _default_name("sensor"))


@dataclass(frozen=True)
class NoiseNode:
    kind: str = "none"
    name: str = field(default_factory=lambda: _default_name("noise"))


@dataclass(frozen=True)
class TapNode:
    signal: str
    name: str = field(default_factory=lambda: _default_name("tap"))


@dataclass(frozen=True)
class ControlGraph:
    reference: ReferenceNode
    sum_node: SumNode
    controller: ControllerNode
    actuator: ActuatorNode
    plant: PlantNode
    sensor: SensorNode
    disturbance: DisturbanceNode | None = None
    noise: NoiseNode | None = None
    taps: tuple[TapNode, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ParameterizedGraph:
    graph: ControlGraph
    structure_family: str
    controller_family: str
    actuator_family: str
    plant_family: str
    sensor_family: str
    disturbance_family: str
    noise_family: str
    controller_dynamics: ControlNode
    actuator_dynamics: ControlNode | None
    plant_control_dynamics: ControlNode
    plant_disturbance_dynamics: ControlNode | None
    sensor_dynamics: ControlNode | None
    actuator_saturation_limit: float | None = None
    sensor_bias: float = 0.0
    module_params: dict[str, dict[str, Any]] = field(default_factory=dict)


def is_parameterized(node: ControlNode) -> bool:
    if isinstance(node, TFNode):
        return node.num is not None and node.den is not None
    if isinstance(node, GainNode):
        return node.k is not None
    if isinstance(node, MatrixGainNode):
        return node.k is not None
    if isinstance(node, PIDNode):
        return None not in (node.kp, node.ki, node.kd)
    if isinstance(node, DelayNode):
        return node.t is not None
    if isinstance(node, SSNode):
        return None not in (node.a, node.b, node.c, node.d)
    if isinstance(node, SeriesNode | ParallelNode):
        return all(is_parameterized(block) for block in node.blocks)
    if isinstance(node, FeedbackNode):
        return is_parameterized(node.forward) and is_parameterized(node.feedback)
    raise TypeError(f"Unsupported node type: {type(node)!r}")
