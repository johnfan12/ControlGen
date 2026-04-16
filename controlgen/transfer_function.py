from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from controlgen.dsl import serialize_dsl
from controlgen.types import (
    ControlNode,
    DelayNode,
    FeedbackNode,
    GainNode,
    PIDNode,
    ParallelNode,
    SeriesNode,
    TFNode,
    is_parameterized,
)


@dataclass(frozen=True)
class RationalTransferFunction:
    num: tuple[float, ...]
    den: tuple[float, ...]

    def normalized(self) -> "RationalTransferFunction":
        num = _trim(self.num)
        den = _trim(self.den)
        if den[0] == 0:
            raise ValueError("Denominator leading coefficient cannot be zero")
        scale = den[0]
        num = tuple(float(v / scale) for v in num)
        den = tuple(float(v / scale) for v in den)
        return RationalTransferFunction(num=num, den=den)

    @property
    def poles(self) -> np.ndarray:
        return np.roots(np.asarray(self.den, dtype=float))

    @property
    def zeros(self) -> np.ndarray:
        if len(self.num) <= 1:
            return np.array([], dtype=complex)
        return np.roots(np.asarray(self.num, dtype=float))


@dataclass(frozen=True)
class ParameterizedSystem:
    ast: ControlNode
    transfer_function: RationalTransferFunction
    dsl_text: str
    domain: str = "continuous"

    @property
    def poles(self) -> np.ndarray:
        return self.transfer_function.poles

    @property
    def zeros(self) -> np.ndarray:
        return self.transfer_function.zeros

    @property
    def is_stable(self) -> bool:
        if len(self.poles) == 0:
            return True
        return bool(np.all(np.real(self.poles) < -1e-8))


def tf_from_node(node: ControlNode) -> ParameterizedSystem:
    if not is_parameterized(node):
        raise ValueError("Control node must be fully parameterized before simulation")
    tf = _to_tf(node).normalized()
    return ParameterizedSystem(ast=node, transfer_function=tf, dsl_text=serialize_dsl(node))


def series_tf(*blocks: RationalTransferFunction) -> RationalTransferFunction:
    num = np.array([1.0])
    den = np.array([1.0])
    for block in blocks:
        num = np.polymul(num, block.num)
        den = np.polymul(den, block.den)
    return RationalTransferFunction(tuple(num), tuple(den)).normalized()


def parallel_tf(*blocks: RationalTransferFunction) -> RationalTransferFunction:
    if len(blocks) < 2:
        raise ValueError("parallel_tf requires at least two blocks")
    acc = blocks[0]
    for block in blocks[1:]:
        num = _poly_add(np.polymul(acc.num, block.den), np.polymul(block.num, acc.den))
        den = np.polymul(acc.den, block.den)
        acc = RationalTransferFunction(tuple(num), tuple(den)).normalized()
    return acc


def feedback_tf(
    forward: RationalTransferFunction,
    feedback: RationalTransferFunction,
    sign: int = -1,
) -> RationalTransferFunction:
    num = np.polymul(forward.num, feedback.den)
    den = _poly_add(
        np.polymul(forward.den, feedback.den),
        -sign * np.polymul(forward.num, feedback.num),
    )
    return RationalTransferFunction(tuple(num), tuple(den)).normalized()


def _to_tf(node: ControlNode) -> RationalTransferFunction:
    if isinstance(node, TFNode):
        return RationalTransferFunction(node.num or (1.0,), node.den or (1.0, 1.0))
    if isinstance(node, GainNode):
        return RationalTransferFunction((float(node.k),), (1.0,))
    if isinstance(node, PIDNode):
        return _pid_tf(node)
    if isinstance(node, DelayNode):
        return _delay_tf(node)
    if isinstance(node, SeriesNode):
        return series_tf(*(_to_tf(block) for block in node.blocks))
    if isinstance(node, ParallelNode):
        return parallel_tf(*(_to_tf(block) for block in node.blocks))
    if isinstance(node, FeedbackNode):
        return feedback_tf(_to_tf(node.forward), _to_tf(node.feedback), sign=node.sign)
    raise TypeError(f"Unsupported node type: {type(node)!r}")


def _pid_tf(node: PIDNode) -> RationalTransferFunction:
    s = np.array([1.0, 0.0])
    if node.tau > 0:
        tau_poly = np.array([node.tau, 1.0])
        kp_term = node.kp * np.polymul(s, tau_poly)
        ki_term = node.ki * tau_poly
        kd_term = node.kd * np.array([1.0, 0.0, 0.0])
        num = _poly_add(_poly_add(kp_term, ki_term), kd_term)
        den = np.polymul(s, tau_poly)
    else:
        num = _poly_add(np.array([node.kd, node.kp, node.ki]), np.array([0.0]))
        den = s
    return RationalTransferFunction(tuple(num), tuple(den)).normalized()


def _delay_tf(node: DelayNode) -> RationalTransferFunction:
    if node.t is None or node.t <= 0:
        return RationalTransferFunction((1.0,), (1.0,))
    if node.order == 1:
        return RationalTransferFunction((-node.t / 2.0, 1.0), (node.t / 2.0, 1.0))
    return RationalTransferFunction(
        (node.t**2 / 12.0, -node.t / 2.0, 1.0),
        (node.t**2 / 12.0, node.t / 2.0, 1.0),
    )


def _poly_add(a: np.ndarray | tuple[float, ...], b: np.ndarray | tuple[float, ...]) -> np.ndarray:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if len(a_arr) < len(b_arr):
        a_arr = np.pad(a_arr, (len(b_arr) - len(a_arr), 0))
    elif len(b_arr) < len(a_arr):
        b_arr = np.pad(b_arr, (len(a_arr) - len(b_arr), 0))
    return a_arr + b_arr


def _trim(values: tuple[float, ...]) -> tuple[float, ...]:
    arr = np.asarray(values, dtype=float)
    arr = np.trim_zeros(arr, trim="f")
    if arr.size == 0:
        arr = np.array([0.0])
    return tuple(float(v) for v in arr)
