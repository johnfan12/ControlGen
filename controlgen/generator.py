from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from controlgen.dataset import DatasetSample, build_sample
from controlgen.simulate import InputSpec, SimulationConfig, simulate
from controlgen.transfer_function import ParameterizedSystem, tf_from_node
from controlgen.types import (
    ControlNode,
    DelayNode,
    FeedbackNode,
    GainNode,
    PIDNode,
    ParallelNode,
    SeriesNode,
    TFNode,
)


@dataclass(frozen=True)
class GrammarConfig:
    allow_delay: bool = False
    closed_loop_probability: float = 0.8
    controller_probability: float = 0.8
    max_plant_order: int = 3
    difficulty_levels: tuple[str, ...] = ("level1", "level2", "level3")


@dataclass(frozen=True)
class SamplingConfig:
    gain_range: tuple[float, float] = (0.3, 3.0)
    pole_range: tuple[float, float] = (0.2, 2.5)
    zero_range: tuple[float, float] = (0.1, 1.5)
    pid_kp_range: tuple[float, float] = (0.2, 4.0)
    pid_ki_range: tuple[float, float] = (0.0, 2.0)
    pid_kd_range: tuple[float, float] = (0.0, 0.6)
    delay_range: tuple[float, float] = (0.05, 0.4)
    max_sampling_attempts: int = 20


@dataclass(frozen=True)
class DatasetConfig:
    count: int = 10
    seed: int = 0
    input_kinds: tuple[str, ...] = ("step", "impulse", "ramp", "sine")
    difficulty_weights: dict[str, float] = field(
        default_factory=lambda: {"level1": 0.45, "level2": 0.35, "level3": 0.20}
    )


def generate_structure(
    config: GrammarConfig,
    rng: np.random.Generator,
    difficulty: str | None = None,
) -> ControlNode:
    level = difficulty or _sample_difficulty(config.difficulty_levels, rng)
    plant = TFNode(name="plant")

    if level == "level1":
        if rng.random() > config.closed_loop_probability:
            return plant
        controller = PIDNode(name="controller") if rng.random() < config.controller_probability else GainNode(name="controller")
        return FeedbackNode(forward=SeriesNode((controller, plant)), feedback=GainNode(k=1.0, name="sensor"))

    if level == "level2":
        controller = PIDNode(name="controller")
        forward_blocks: list[ControlNode] = [controller, plant]
        if rng.random() < 0.5:
            forward_blocks.insert(1, TFNode(name="compensator"))
        else:
            forward_blocks[0] = ParallelNode((controller, GainNode(name="feedforward")))
        return FeedbackNode(
            forward=SeriesNode(tuple(forward_blocks)),
            feedback=GainNode(name="sensor"),
        )

    controller = PIDNode(name="controller")
    shaped_plant: ControlNode = SeriesNode((TFNode(name="prefilter"), plant))
    if config.allow_delay and rng.random() < 0.5:
        shaped_plant = SeriesNode((shaped_plant, DelayNode(name="transport_delay", order=1)))
    return FeedbackNode(
        forward=SeriesNode((ParallelNode((controller, GainNode(name="bypass"))), shaped_plant)),
        feedback=SeriesNode((GainNode(name="sensor"), TFNode(name="sensor_filter"))),
    )


def sample_parameters(
    ast: ControlNode,
    config: SamplingConfig,
    rng: np.random.Generator,
) -> ParameterizedSystem:
    for _ in range(config.max_sampling_attempts):
        parameterized = _parameterize_node(ast, config, rng)
        system = tf_from_node(parameterized)
        if system.is_stable:
            return system
    raise RuntimeError("Failed to sample a stable control system")


def generate_sample(
    dataset_config: DatasetConfig,
    grammar_config: GrammarConfig | None = None,
    sampling_config: SamplingConfig | None = None,
    sample_index: int = 0,
) -> DatasetSample:
    grammar = grammar_config or GrammarConfig()
    sampling = sampling_config or SamplingConfig()
    sample_seed = int(dataset_config.seed + sample_index)
    rng = np.random.default_rng(sample_seed)
    difficulty = _choose_weighted(dataset_config.difficulty_weights, rng)
    structure = generate_structure(grammar, rng, difficulty=difficulty)
    system = sample_parameters(structure, sampling, rng)
    input_spec = _sample_input_spec(dataset_config.input_kinds, rng)
    trajectory = simulate(system, input_spec, SimulationConfig())
    metadata = {
        "sample_id": _sample_id(sample_seed, sample_index),
        "seed": sample_seed,
        "difficulty": difficulty,
        "system_type": "control_siso",
    }
    sample = build_sample(system, trajectory, metadata)
    sample.tags["difficulty"] = difficulty
    return sample


def generate_dataset(
    dataset_config: DatasetConfig,
    grammar_config: GrammarConfig | None = None,
    sampling_config: SamplingConfig | None = None,
) -> list[DatasetSample]:
    return [
        generate_sample(dataset_config, grammar_config, sampling_config, sample_index=index)
        for index in range(dataset_config.count)
    ]


def iter_dataset(
    dataset_config: DatasetConfig,
    grammar_config: GrammarConfig | None = None,
    sampling_config: SamplingConfig | None = None,
) -> Iterator[DatasetSample]:
    for index in range(dataset_config.count):
        yield generate_sample(dataset_config, grammar_config, sampling_config, sample_index=index)


def _parameterize_node(
    node: ControlNode,
    config: SamplingConfig,
    rng: np.random.Generator,
) -> ControlNode:
    if isinstance(node, TFNode):
        if node.num is not None and node.den is not None:
            return node
        if node.name in {"plant", "prefilter", "sensor_filter"}:
            order = int(rng.integers(1, 4))
            num, den = _sample_stable_tf(order, config, rng)
            return TFNode(num=num, den=den, name=node.name)
        num, den = _sample_compensator_tf(config, rng)
        return TFNode(num=num, den=den, name=node.name)
    if isinstance(node, GainNode):
        if node.k is not None:
            return node
        low, high = config.gain_range
        return GainNode(k=float(rng.uniform(low, high)), name=node.name)
    if isinstance(node, PIDNode):
        return PIDNode(
            kp=_sample_range(config.pid_kp_range, rng),
            ki=_sample_range(config.pid_ki_range, rng),
            kd=_sample_range(config.pid_kd_range, rng),
            tau=0.05,
            name=node.name,
        )
    if isinstance(node, DelayNode):
        if node.t is not None:
            return node
        return DelayNode(t=_sample_range(config.delay_range, rng), order=node.order, name=node.name)
    if isinstance(node, SeriesNode):
        return SeriesNode(tuple(_parameterize_node(block, config, rng) for block in node.blocks))
    if isinstance(node, ParallelNode):
        return ParallelNode(tuple(_parameterize_node(block, config, rng) for block in node.blocks))
    if isinstance(node, FeedbackNode):
        return FeedbackNode(
            forward=_parameterize_node(node.forward, config, rng),
            feedback=_parameterize_node(node.feedback, config, rng),
            sign=node.sign,
        )
    raise TypeError(f"Unsupported node type: {type(node)!r}")


def _sample_stable_tf(
    order: int,
    config: SamplingConfig,
    rng: np.random.Generator,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    poles = -rng.uniform(config.pole_range[0], config.pole_range[1], size=order)
    den = tuple(float(v) for v in np.poly(poles))
    gain = _sample_range(config.gain_range, rng)
    num = (gain,)
    return num, den


def _sample_compensator_tf(
    config: SamplingConfig,
    rng: np.random.Generator,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    zero = -_sample_range(config.zero_range, rng)
    pole = -_sample_range(config.pole_range, rng)
    gain = _sample_range(config.gain_range, rng)
    num = tuple(float(v * gain) for v in np.poly([zero]))
    den = tuple(float(v) for v in np.poly([pole]))
    return num, den


def _sample_input_spec(kinds: tuple[str, ...], rng: np.random.Generator) -> InputSpec:
    kind = kinds[int(rng.integers(0, len(kinds)))]
    amplitude = float(rng.uniform(0.5, 2.0))
    frequency = float(rng.uniform(0.1, 1.0))
    return InputSpec(kind=kind, amplitude=amplitude, frequency=frequency)


def _choose_weighted(weights: dict[str, float], rng: np.random.Generator) -> str:
    keys = list(weights)
    probs = np.asarray([weights[key] for key in keys], dtype=float)
    probs = probs / probs.sum()
    index = int(rng.choice(len(keys), p=probs))
    return keys[index]


def _sample_difficulty(levels: tuple[str, ...], rng: np.random.Generator) -> str:
    return levels[int(rng.integers(0, len(levels)))]


def _sample_range(bounds: tuple[float, float], rng: np.random.Generator) -> float:
    return float(rng.uniform(bounds[0], bounds[1]))


def _sample_id(seed: int, sample_index: int) -> str:
    value = f"{seed}:{sample_index}".encode("utf-8")
    return hashlib.sha1(value).hexdigest()[:12]
