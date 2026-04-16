from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import signal

from controlgen.transfer_function import ParameterizedSystem


@dataclass(frozen=True)
class InputSpec:
    kind: str = "step"
    amplitude: float = 1.0
    frequency: float = 1.0
    phase: float = 0.0


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
    u = build_input_signal(t, input_spec, sim_config.dt)
    lti = signal.TransferFunction(system.transfer_function.num, system.transfer_function.den)
    _, y, _ = signal.lsim(lti, U=u, T=t)
    return Trajectory(t=t, u=np.asarray(u), y=np.asarray(y), input_spec=input_spec)


def build_input_signal(t: np.ndarray, input_spec: InputSpec, dt: float) -> np.ndarray:
    kind = input_spec.kind.lower()
    if kind == "step":
        return np.full_like(t, input_spec.amplitude, dtype=float)
    if kind == "impulse":
        u = np.zeros_like(t, dtype=float)
        if len(u) > 0:
            u[0] = input_spec.amplitude / max(dt, 1e-12)
        return u
    if kind == "ramp":
        return input_spec.amplitude * t
    if kind == "sine":
        return input_spec.amplitude * np.sin(2.0 * np.pi * input_spec.frequency * t + input_spec.phase)
    raise ValueError(f"Unsupported input kind: {input_spec.kind}")
