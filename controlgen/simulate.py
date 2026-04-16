from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import signal

from controlgen.transfer_function import ParameterizedSystem


@dataclass(frozen=True)
class InputSpec:
    kind: str = "step"
    amplitude: float = 1.0
    frequency: float = 1.0
    phase: float = 0.0
    channel_mode: str = "all"
    active_channels: tuple[int, ...] = field(default_factory=tuple)
    pattern_family: str = "standard"


@dataclass(frozen=True)
class SimulationConfig:
    duration: float = 10.0
    dt: float = 0.01


@dataclass(frozen=True)
class Trajectory:
    t: np.ndarray
    u: np.ndarray
    y: np.ndarray
    input_spec: InputSpec

    def as_serializable(self) -> dict[str, object]:
        return {
            "t": self.t.tolist(),
            "u": self.u.tolist(),
            "y": self.y.tolist(),
            "input_spec": asdict(self.input_spec),
        }


def simulate(
    system: ParameterizedSystem,
    input_spec: InputSpec,
    sim_config: SimulationConfig,
) -> Trajectory:
    t = np.arange(0.0, sim_config.duration + sim_config.dt, sim_config.dt, dtype=float)
    u = build_input_signal(t, input_spec, system.input_channels, sim_config.dt)
    lti = signal.StateSpace(system.a, system.b, system.c, system.d)
    u_arg = u[:, 0] if system.input_channels == 1 else u
    _, y, _ = signal.lsim(lti, U=u_arg, T=t)
    y_array = np.asarray(y, dtype=float)
    if y_array.ndim == 1:
        y_array = y_array.reshape(-1, 1)
    return Trajectory(t=t, u=u, y=y_array, input_spec=input_spec)


def build_input_signal(
    t: np.ndarray,
    input_spec: InputSpec,
    input_channels: int,
    dt: float,
) -> np.ndarray:
    if input_channels <= 0:
        raise ValueError("Input channels must be positive")
    active_channels = tuple(input_spec.active_channels) or _default_channels(
        input_spec.channel_mode,
        input_channels,
    )
    base = _build_base_signal(t, input_spec, dt)
    u = np.zeros((len(t), input_channels), dtype=float)
    for channel in active_channels:
        if 0 <= channel < input_channels:
            u[:, channel] = base
    return u


def _build_base_signal(t: np.ndarray, input_spec: InputSpec, dt: float) -> np.ndarray:
    kind = input_spec.kind.lower()
    if kind == "step":
        return np.full(len(t), input_spec.amplitude, dtype=float)
    if kind == "impulse":
        u = np.zeros(len(t), dtype=float)
        if len(u) > 0:
            u[0] = input_spec.amplitude / max(dt, 1e-12)
        return u
    if kind == "ramp":
        return input_spec.amplitude * t
    if kind == "sine":
        return input_spec.amplitude * np.sin(2.0 * np.pi * input_spec.frequency * t + input_spec.phase)
    if kind == "chirp":
        f1 = max(input_spec.frequency * 4.0, input_spec.frequency + 0.1)
        return input_spec.amplitude * signal.chirp(t, f0=input_spec.frequency, t1=max(t[-1], dt), f1=f1)
    if kind == "multistep":
        levels = np.array([0.4, 1.0, -0.6, 0.8], dtype=float) * input_spec.amplitude
        segments = np.array_split(np.arange(len(t)), len(levels))
        u = np.zeros(len(t), dtype=float)
        for level, indices in zip(levels, segments, strict=False):
            u[indices] = level
        return u
    if kind == "prbs":
        u = np.zeros(len(t), dtype=float)
        hold = max(int(0.25 / max(dt, 1e-6)), 1)
        for start in range(0, len(t), hold):
            bit = 1.0 if ((start // hold) % 2 == 0) else -1.0
            u[start : start + hold] = bit * input_spec.amplitude
        return u
    raise ValueError(f"Unsupported input kind: {input_spec.kind}")


def _default_channels(mode: str, input_channels: int) -> tuple[int, ...]:
    if mode == "single":
        return (0,)
    if mode == "all":
        return tuple(range(input_channels))
    raise ValueError(f"Unsupported channel mode: {mode}")
