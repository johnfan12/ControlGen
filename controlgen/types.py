from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

Matrix = tuple[tuple[float, ...], ...]


def _default_name(prefix: str) -> str:
    return prefix


@dataclass(frozen=True)
class TFNode:
    num: tuple[float, ...] | None = None
    den: tuple[float, ...] | None = None
    name: str = field(default_factory=lambda: _default_name("plant"))


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
