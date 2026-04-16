from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg, signal

from controlgen.dsl import serialize_dsl
from controlgen.types import (
    ControlNode,
    DelayNode,
    FeedbackNode,
    GainNode,
    Matrix,
    MatrixGainNode,
    PIDNode,
    ParallelNode,
    SSNode,
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
        return RationalTransferFunction(
            num=tuple(float(v / scale) for v in num),
            den=tuple(float(v / scale) for v in den),
        )

    @property
    def poles(self) -> np.ndarray:
        return np.roots(np.asarray(self.den, dtype=float))

    @property
    def zeros(self) -> np.ndarray:
        if len(self.num) <= 1:
            return np.array([], dtype=complex)
        return np.roots(np.asarray(self.num, dtype=float))


@dataclass(frozen=True)
class StateSpaceModel:
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    d: np.ndarray
    ast: ControlNode
    dsl_text: str
    domain: str = "continuous"

    @property
    def input_channels(self) -> int:
        return int(self.b.shape[1]) if self.b.ndim == 2 else 0

    @property
    def output_channels(self) -> int:
        return int(self.c.shape[0]) if self.c.ndim == 2 else 0

    @property
    def state_dimension(self) -> int:
        return int(self.a.shape[0]) if self.a.ndim == 2 else 0

    @property
    def poles(self) -> np.ndarray:
        if self.state_dimension == 0:
            return np.array([], dtype=complex)
        return np.linalg.eigvals(self.a)

    @property
    def zeros(self) -> np.ndarray:
        tf = self.transfer_function
        return tf.zeros if tf is not None else np.array([], dtype=complex)

    @property
    def is_stable(self) -> bool:
        return bool(len(self.poles) == 0 or np.all(np.real(self.poles) < -1e-8))

    @property
    def spectral_radius(self) -> float:
        if len(self.poles) == 0:
            return 0.0
        return float(np.max(np.abs(self.poles)))

    @property
    def dc_gain(self) -> np.ndarray:
        if self.state_dimension == 0:
            return np.asarray(self.d, dtype=float)
        try:
            return self.d - self.c @ np.linalg.solve(self.a, self.b)
        except np.linalg.LinAlgError:
            return self.d - self.c @ np.linalg.pinv(self.a) @ self.b

    @property
    def transfer_function(self) -> RationalTransferFunction | None:
        if self.input_channels != 1 or self.output_channels != 1:
            return None
        if self.state_dimension == 0:
            return RationalTransferFunction((float(self.d[0, 0]),), (1.0,))
        num, den = signal.ss2tf(self.a, self.b, self.c, self.d)
        return RationalTransferFunction(tuple(num[0]), tuple(den)).normalized()


ParameterizedSystem = StateSpaceModel


def tf_from_node(node: ControlNode) -> ParameterizedSystem:
    if not is_parameterized(node):
        raise ValueError("Control node must be fully parameterized before simulation")
    a, b, c, d = _to_state_space(node)
    return StateSpaceModel(
        a=a,
        b=b,
        c=c,
        d=d,
        ast=node,
        dsl_text=serialize_dsl(node),
    )


def block_diagonal_systems(systems: list[StateSpaceModel]) -> StateSpaceModel:
    if not systems:
        raise ValueError("At least one subsystem is required")
    a = _block_diag(*(system.a for system in systems))
    input_channels = sum(system.input_channels for system in systems)
    output_channels = sum(system.output_channels for system in systems)
    b = np.zeros((a.shape[0], input_channels), dtype=float)
    c = np.zeros((output_channels, a.shape[0]), dtype=float)
    d = np.zeros((output_channels, input_channels), dtype=float)
    state_offset = 0
    input_offset = 0
    output_offset = 0
    for system in systems:
        states = system.state_dimension
        inputs = system.input_channels
        outputs = system.output_channels
        if states:
            b[state_offset : state_offset + states, input_offset : input_offset + inputs] = system.b
            c[output_offset : output_offset + outputs, state_offset : state_offset + states] = system.c
        d[output_offset : output_offset + outputs, input_offset : input_offset + inputs] = system.d
        state_offset += states
        input_offset += inputs
        output_offset += outputs
    return StateSpaceModel(a=a, b=b, c=c, d=d, ast=systems[0].ast, dsl_text=systems[0].dsl_text)


def _to_state_space(node: ControlNode) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(node, TFNode):
        return _tf_to_ss(node.num or (1.0,), node.den or (1.0, 1.0))
    if isinstance(node, GainNode):
        return _static_gain(np.array([[float(node.k)]], dtype=float))
    if isinstance(node, MatrixGainNode):
        return _static_gain(np.asarray(node.k, dtype=float))
    if isinstance(node, PIDNode):
        tf = _pid_tf(node)
        return _tf_to_ss(tf.num, tf.den)
    if isinstance(node, DelayNode):
        tf = _delay_tf(node)
        return _tf_to_ss(tf.num, tf.den)
    if isinstance(node, SSNode):
        return (
            np.asarray(node.a, dtype=float),
            np.asarray(node.b, dtype=float),
            np.asarray(node.c, dtype=float),
            np.asarray(node.d, dtype=float),
        )
    if isinstance(node, SeriesNode):
        systems = [_to_state_space(block) for block in node.blocks]
        current = systems[0]
        for nxt in systems[1:]:
            current = _series_state_space(current, nxt)
        return current
    if isinstance(node, ParallelNode):
        systems = [_to_state_space(block) for block in node.blocks]
        current = systems[0]
        for nxt in systems[1:]:
            current = _parallel_state_space(current, nxt)
        return current
    if isinstance(node, FeedbackNode):
        return _feedback_state_space(_to_state_space(node.forward), _to_state_space(node.feedback), sign=node.sign)
    raise TypeError(f"Unsupported node type: {type(node)!r}")


def _tf_to_ss(
    num: tuple[float, ...] | np.ndarray,
    den: tuple[float, ...] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_arr = np.trim_zeros(np.asarray(num, dtype=float), trim="f")
    den_arr = np.trim_zeros(np.asarray(den, dtype=float), trim="f")
    if den_arr.size == 0:
        raise ValueError("Denominator cannot be empty")
    if num_arr.size == 0:
        num_arr = np.array([0.0], dtype=float)
    if len(den_arr) == 1:
        return _static_gain(np.array([[float(num_arr[-1] / den_arr[-1])]], dtype=float))
    a, b, c, d = signal.tf2ss(num_arr, den_arr)
    return np.asarray(a, dtype=float), np.asarray(b, dtype=float), np.asarray(c, dtype=float), np.asarray(d, dtype=float)


def _static_gain(gain: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gain_2d = np.atleast_2d(np.asarray(gain, dtype=float))
    outputs, inputs = gain_2d.shape
    return (
        np.zeros((0, 0), dtype=float),
        np.zeros((0, inputs), dtype=float),
        np.zeros((outputs, 0), dtype=float),
        gain_2d,
    )


def _series_state_space(
    first: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a1, b1, c1, d1 = first
    a2, b2, c2, d2 = second
    if c1.shape[0] != b2.shape[1]:
        raise ValueError("Series connection shape mismatch")
    n1 = a1.shape[0]
    n2 = a2.shape[0]
    top = np.hstack([a1, np.zeros((n1, n2), dtype=float)])
    bottom = np.hstack([b2 @ c1, a2])
    a = np.vstack([top, bottom])
    b = np.vstack([b1, b2 @ d1])
    c = np.hstack([d2 @ c1, c2])
    d = d2 @ d1
    return a, b, c, d


def _parallel_state_space(
    first: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a1, b1, c1, d1 = first
    a2, b2, c2, d2 = second
    if b1.shape[1] != b2.shape[1] or c1.shape[0] != c2.shape[0]:
        raise ValueError("Parallel connection requires matching input/output dimensions")
    a = _block_diag(a1, a2)
    b = np.vstack([b1, b2])
    c = np.hstack([c1, c2])
    d = d1 + d2
    return a, b, c, d


def _feedback_state_space(
    forward: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    feedback: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    sign: int = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ag, bg, cg, dg = forward
    ah, bh, ch, dh = feedback
    ck = sign * ch
    dk = sign * dh
    identity = np.eye(dg.shape[0], dtype=float)
    matrix = identity - dg @ dk
    try:
        inv_matrix = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Feedback interconnection is singular") from exc

    a11 = ag + bg @ dk @ inv_matrix @ cg
    a12 = bg @ (ck + dk @ inv_matrix @ dg @ ck)
    a21 = bh @ inv_matrix @ cg
    a22 = ah + bh @ inv_matrix @ dg @ ck
    a = np.vstack([np.hstack([a11, a12]), np.hstack([a21, a22])])

    b1 = bg + bg @ dk @ inv_matrix @ dg
    b2 = bh @ inv_matrix @ dg
    b = np.vstack([b1, b2])

    c = np.hstack([inv_matrix @ cg, inv_matrix @ dg @ ck])
    d = inv_matrix @ dg
    return a, b, c, d


def _pid_tf(node: PIDNode) -> RationalTransferFunction:
    s = np.array([1.0, 0.0], dtype=float)
    tau_poly = np.array([node.tau, 1.0], dtype=float)
    kp_term = node.kp * np.polymul(s, tau_poly)
    ki_term = node.ki * tau_poly
    kd_term = node.kd * np.array([1.0, 0.0, 0.0], dtype=float)
    num = _poly_add(_poly_add(kp_term, ki_term), kd_term)
    den = np.polymul(s, tau_poly)
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


def _poly_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) < len(b):
        a = np.pad(a, (len(b) - len(a), 0))
    elif len(b) < len(a):
        b = np.pad(b, (len(a) - len(b), 0))
    return a + b


def _block_diag(*matrices: np.ndarray) -> np.ndarray:
    if not matrices:
        return np.zeros((0, 0), dtype=float)
    if all(matrix.size == 0 for matrix in matrices):
        return np.zeros((0, 0), dtype=float)
    return np.asarray(linalg.block_diag(*matrices), dtype=float)


def _trim(values: tuple[float, ...]) -> tuple[float, ...]:
    arr = np.asarray(values, dtype=float)
    arr = np.trim_zeros(arr, trim="f")
    if arr.size == 0:
        arr = np.array([0.0], dtype=float)
    return tuple(float(v) for v in arr)


def matrix_to_tuple(matrix: np.ndarray) -> Matrix:
    return tuple(tuple(float(value) for value in row) for row in np.asarray(matrix, dtype=float))
