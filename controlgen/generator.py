from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from controlgen.dataset import DatasetSample, build_sample
from controlgen.simulate import InputSpec, SimulationConfig, simulate
from controlgen.transfer_function import (
    ParameterizedSystem,
    block_diagonal_systems,
    matrix_to_tuple,
    tf_from_node,
)
from controlgen.types import (
    ControlNode,
    DelayNode,
    FeedbackNode,
    GainNode,
    MatrixGainNode,
    PIDNode,
    ParallelNode,
    SSNode,
    SeriesNode,
    TFNode,
)


@dataclass(frozen=True)
class GrammarConfig:
    system_type_weights: dict[str, float] = field(
        default_factory=lambda: {"siso": 0.8, "mimo2x2": 0.2}
    )
    siso_structure_weights: dict[str, float] = field(
        default_factory=lambda: {
            "open_loop_plant": 0.10,
            "pid_feedback": 0.22,
            "lead_lag_feedback": 0.16,
            "cascade_compensator": 0.16,
            "feedforward_pid": 0.18,
            "two_dof_pid": 0.10,
            "sensor_filter_loop": 0.08,
        }
    )
    mimo_structure_weights: dict[str, float] = field(
        default_factory=lambda: {
            "near_diagonal": 0.35,
            "weak_coupled": 0.35,
            "strong_coupled": 0.30,
        }
    )
    mimo_controller_weights: dict[str, float] = field(
        default_factory=lambda: {
            "diag_pid": 0.7,
            "static_decoupler_pid": 0.3,
        }
    )
    allow_delay: bool = True
    max_plant_order: int = 4


@dataclass(frozen=True)
class SamplingConfig:
    gain_range: tuple[float, float] = (0.3, 3.0)
    pole_range: tuple[float, float] = (0.2, 2.5)
    zero_range: tuple[float, float] = (0.1, 1.8)
    pid_kp_range: tuple[float, float] = (0.2, 4.5)
    pid_ki_range: tuple[float, float] = (0.0, 2.5)
    pid_kd_range: tuple[float, float] = (0.0, 0.8)
    delay_range: tuple[float, float] = (0.05, 0.4)
    parameter_family_weights: dict[str, float] = field(
        default_factory=lambda: {
            "balanced": 0.35,
            "fast": 0.20,
            "oscillatory": 0.25,
            "stiff": 0.20,
        }
    )
    max_sampling_attempts: int = 30


@dataclass(frozen=True)
class DatasetConfig:
    count: int = 10
    seed: int = 0
    system_mode: str = "mixed"
    input_family_weights: dict[str, float] = field(
        default_factory=lambda: {"standard": 0.7, "benchmark": 0.3}
    )
    sim_config: SimulationConfig = field(default_factory=SimulationConfig)


def generate_structure(
    config: GrammarConfig,
    rng: np.random.Generator,
    difficulty: str | None = None,
    system_type: str | None = None,
    structure_family: str | None = None,
    controller_family: str | None = None,
) -> ControlNode:
    del difficulty
    ast, _ = _generate_structure_bundle(
        config=config,
        rng=rng,
        system_type=system_type,
        structure_family=structure_family,
        controller_family=controller_family,
    )
    return ast


def sample_parameters(
    ast: ControlNode,
    config: SamplingConfig,
    rng: np.random.Generator,
    parameter_family: str | None = None,
) -> ParameterizedSystem:
    family = parameter_family or _choose_weighted(config.parameter_family_weights, rng)
    for _ in range(config.max_sampling_attempts):
        if _contains_marker(ast, "plant_2x2"):
            parameterized = _parameterize_mimo_ast(ast, config, rng, family)
        else:
            parameterized = _parameterize_siso_node(ast, config, rng, family)
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

    system_type = _choose_system_type(dataset_config, grammar, rng)
    ast, structure_meta = _generate_structure_bundle(grammar, rng, system_type=system_type)
    parameter_family = _choose_weighted(sampling.parameter_family_weights, rng)
    system = sample_parameters(ast, sampling, rng, parameter_family=parameter_family)
    input_family = _choose_weighted(dataset_config.input_family_weights, rng)
    input_spec = _sample_input_spec(system.input_channels, input_family, rng)
    trajectory = simulate(system, input_spec, dataset_config.sim_config)
    metadata = {
        "sample_id": _sample_id(sample_seed, sample_index),
        "seed": sample_seed,
        "system_type": "control_mimo_2x2" if system_type == "mimo2x2" else "control_siso",
        "structure_family": structure_meta["structure_family"],
        "parameter_family": parameter_family,
        "input_family": input_family,
        "controller_family": structure_meta["controller_family"],
        "coupling_level": structure_meta["coupling_level"],
    }
    return build_sample(system, trajectory, metadata)


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


def _generate_structure_bundle(
    config: GrammarConfig,
    rng: np.random.Generator,
    system_type: str | None = None,
    structure_family: str | None = None,
    controller_family: str | None = None,
) -> tuple[ControlNode, dict[str, str]]:
    chosen_system_type = system_type or _choose_weighted(config.system_type_weights, rng)
    if chosen_system_type == "mimo2x2":
        coupling_family = structure_family or _choose_weighted(config.mimo_structure_weights, rng)
        chosen_controller = controller_family or _choose_weighted(config.mimo_controller_weights, rng)
        ast = _make_mimo_structure(coupling_family, chosen_controller)
        return ast, {
            "structure_family": coupling_family,
            "controller_family": chosen_controller,
            "coupling_level": coupling_family,
        }
    family = structure_family or _choose_weighted(config.siso_structure_weights, rng)
    ast = _make_siso_structure(family, config.allow_delay)
    return ast, {
        "structure_family": family,
        "controller_family": _controller_family_for_structure(family),
        "coupling_level": "none",
    }


def _make_siso_structure(structure_family: str, allow_delay: bool) -> ControlNode:
    plant = TFNode(name="plant")
    sensor = GainNode(k=1.0, name="sensor")
    if structure_family == "open_loop_plant":
        return plant
    if structure_family == "pid_feedback":
        return FeedbackNode(
            forward=SeriesNode((PIDNode(name="controller"), plant)),
            feedback=sensor,
        )
    if structure_family == "lead_lag_feedback":
        return FeedbackNode(
            forward=SeriesNode((TFNode(name="lead_lag"), PIDNode(name="controller"), plant)),
            feedback=sensor,
        )
    if structure_family == "cascade_compensator":
        return FeedbackNode(
            forward=SeriesNode((PIDNode(name="outer_controller"), TFNode(name="inner_compensator"), plant)),
            feedback=SeriesNode((sensor, TFNode(name="sensor_filter"))),
        )
    if structure_family == "feedforward_pid":
        return FeedbackNode(
            forward=SeriesNode(
                (
                    ParallelNode((PIDNode(name="controller"), TFNode(name="feedforward_path"))),
                    plant,
                )
            ),
            feedback=sensor,
        )
    if structure_family == "two_dof_pid":
        return FeedbackNode(
            forward=SeriesNode(
                (
                    ParallelNode((PIDNode(name="feedback_controller"), TFNode(name="prefilter"))),
                    plant,
                )
            ),
            feedback=sensor,
        )
    if structure_family == "sensor_filter_loop":
        blocks: tuple[ControlNode, ...] = (
            PIDNode(name="controller"),
            plant,
            DelayNode(name="transport_delay") if allow_delay else GainNode(k=1.0, name="unity"),
        )
        return FeedbackNode(
            forward=SeriesNode(blocks),
            feedback=SeriesNode((sensor, TFNode(name="sensor_filter"))),
        )
    raise ValueError(f"Unknown SISO structure family: {structure_family}")


def _make_mimo_structure(coupling_family: str, controller_family: str) -> ControlNode:
    plant = SSNode(inputs=2, outputs=2, name=f"plant_2x2__{coupling_family}")
    controller = SSNode(inputs=2, outputs=2, name="mimo_controller__diag_pid")
    sensor = MatrixGainNode(
        k=((1.0, 0.0), (0.0, 1.0)),
        rows=2,
        cols=2,
        name="sensor_2x2",
    )
    if controller_family == "static_decoupler_pid":
        forward = SeriesNode(
            (
                MatrixGainNode(rows=2, cols=2, name="decoupler_2x2"),
                controller,
                plant,
            )
        )
    else:
        forward = SeriesNode((controller, plant))
    return FeedbackNode(forward=forward, feedback=sensor)


def _parameterize_siso_node(
    node: ControlNode,
    config: SamplingConfig,
    rng: np.random.Generator,
    parameter_family: str,
) -> ControlNode:
    if isinstance(node, TFNode):
        if node.num is not None and node.den is not None:
            return node
        if node.name == "plant":
            order = _sample_plant_order(parameter_family, rng)
            num, den = _sample_plant_tf(order, config, rng, parameter_family)
            return TFNode(num=num, den=den, name=node.name)
        num, den = _sample_compensator_tf(node.name, config, rng, parameter_family)
        return TFNode(num=num, den=den, name=node.name)
    if isinstance(node, GainNode):
        if node.k is not None:
            return node
        return GainNode(k=_sample_gain(config, rng, parameter_family), name=node.name)
    if isinstance(node, PIDNode):
        return PIDNode(
            kp=_sample_pid_gain(config.pid_kp_range, rng, parameter_family, gain_type="kp"),
            ki=_sample_pid_gain(config.pid_ki_range, rng, parameter_family, gain_type="ki"),
            kd=_sample_pid_gain(config.pid_kd_range, rng, parameter_family, gain_type="kd"),
            tau=0.03 if parameter_family == "fast" else 0.05,
            name=node.name,
        )
    if isinstance(node, DelayNode):
        if node.t is not None:
            return node
        return DelayNode(
            t=_sample_range(config.delay_range, rng, scale=1.2 if parameter_family == "stiff" else 1.0),
            order=node.order,
            name=node.name,
        )
    if isinstance(node, SeriesNode):
        return SeriesNode(
            tuple(_parameterize_siso_node(block, config, rng, parameter_family) for block in node.blocks)
        )
    if isinstance(node, ParallelNode):
        return ParallelNode(
            tuple(_parameterize_siso_node(block, config, rng, parameter_family) for block in node.blocks)
        )
    if isinstance(node, FeedbackNode):
        return FeedbackNode(
            forward=_parameterize_siso_node(node.forward, config, rng, parameter_family),
            feedback=_parameterize_siso_node(node.feedback, config, rng, parameter_family),
            sign=node.sign,
        )
    if isinstance(node, MatrixGainNode | SSNode):
        return node
    raise TypeError(f"Unsupported node type: {type(node)!r}")


def _parameterize_mimo_ast(
    ast: ControlNode,
    config: SamplingConfig,
    rng: np.random.Generator,
    parameter_family: str,
) -> ControlNode:
    coupling_family = _extract_marker(ast, "plant_2x2__") or "weak_coupled"
    plant = _sample_mimo_plant(coupling_family, parameter_family, rng)
    controller = _sample_mimo_controller(config, rng, parameter_family)
    decoupler = None
    if _contains_marker(ast, "decoupler_2x2"):
        dc_gain = plant.dc_gain + 0.15 * np.eye(2, dtype=float)
        decoupler = np.linalg.pinv(dc_gain)

    def replace(node: ControlNode) -> ControlNode:
        if isinstance(node, SSNode) and node.name.startswith("plant_2x2__"):
            return SSNode(
                a=matrix_to_tuple(plant.a),
                b=matrix_to_tuple(plant.b),
                c=matrix_to_tuple(plant.c),
                d=matrix_to_tuple(plant.d),
                states=plant.state_dimension,
                inputs=plant.input_channels,
                outputs=plant.output_channels,
                name=node.name,
            )
        if isinstance(node, SSNode) and node.name.startswith("mimo_controller__"):
            return SSNode(
                a=matrix_to_tuple(controller.a),
                b=matrix_to_tuple(controller.b),
                c=matrix_to_tuple(controller.c),
                d=matrix_to_tuple(controller.d),
                states=controller.state_dimension,
                inputs=controller.input_channels,
                outputs=controller.output_channels,
                name=node.name,
            )
        if isinstance(node, MatrixGainNode) and node.name == "decoupler_2x2":
            return MatrixGainNode(k=matrix_to_tuple(decoupler), rows=2, cols=2, name=node.name)
        if isinstance(node, SeriesNode):
            return SeriesNode(tuple(replace(block) for block in node.blocks))
        if isinstance(node, ParallelNode):
            return ParallelNode(tuple(replace(block) for block in node.blocks))
        if isinstance(node, FeedbackNode):
            return FeedbackNode(forward=replace(node.forward), feedback=replace(node.feedback), sign=node.sign)
        return node

    return replace(ast)


def _sample_plant_order(parameter_family: str, rng: np.random.Generator) -> int:
    if parameter_family == "fast":
        return int(rng.integers(1, 4))
    if parameter_family == "oscillatory":
        return 2 if rng.random() < 0.8 else 4
    if parameter_family == "stiff":
        return 3 if rng.random() < 0.7 else 4
    return int(rng.integers(1, 5))


def _sample_plant_tf(
    order: int,
    config: SamplingConfig,
    rng: np.random.Generator,
    parameter_family: str,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    gain = _sample_gain(config, rng, parameter_family)
    if parameter_family == "oscillatory" and order >= 2:
        wn = rng.uniform(0.8, 2.2)
        zeta = rng.uniform(0.15, 0.35)
        second_order = np.array([1.0, 2.0 * zeta * wn, wn**2], dtype=float)
        remaining = order - 2
        den = second_order
        if remaining > 0:
            poles = -rng.uniform(config.pole_range[0], config.pole_range[1], size=remaining)
            den = np.polymul(den, np.poly(poles))
        return (gain,), tuple(float(v) for v in den)
    if parameter_family == "stiff":
        poles = [-rng.uniform(0.15, 0.35)]
        poles.extend(-rng.uniform(2.5, 5.0, size=max(order - 1, 0)))
        den = np.poly(poles)
        return (gain,), tuple(float(v) for v in den)
    pole_scale = 1.8 if parameter_family == "fast" else 1.0
    poles = -rng.uniform(
        config.pole_range[0] * pole_scale,
        config.pole_range[1] * pole_scale,
        size=order,
    )
    den = np.poly(poles)
    return (gain,), tuple(float(v) for v in den)


def _sample_compensator_tf(
    name: str,
    config: SamplingConfig,
    rng: np.random.Generator,
    parameter_family: str,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    gain = _sample_gain(config, rng, parameter_family)
    zero = _sample_range(config.zero_range, rng, scale=1.3 if parameter_family == "fast" else 1.0)
    pole = _sample_range(config.pole_range, rng, scale=0.8 if parameter_family == "oscillatory" else 1.0)
    if name in {"lead_lag", "prefilter"}:
        zero *= 1.4
    if name in {"sensor_filter", "feedforward_path"}:
        pole *= 1.8
    num = tuple(float(v * gain) for v in np.poly([-zero]))
    den = tuple(float(v) for v in np.poly([-pole]))
    return num, den


def _sample_gain(
    config: SamplingConfig,
    rng: np.random.Generator,
    parameter_family: str,
) -> float:
    scale = {"balanced": 1.0, "fast": 1.4, "oscillatory": 0.9, "stiff": 1.2}[parameter_family]
    return _sample_range(config.gain_range, rng, scale=scale)


def _sample_pid_gain(
    bounds: tuple[float, float],
    rng: np.random.Generator,
    parameter_family: str,
    gain_type: str,
) -> float:
    scale = 1.0
    if parameter_family == "fast":
        scale = 1.4 if gain_type in {"kp", "kd"} else 1.2
    elif parameter_family == "oscillatory":
        scale = 0.8 if gain_type == "ki" else 1.1
    elif parameter_family == "stiff":
        scale = 1.6 if gain_type == "ki" else 1.2
    return _sample_range(bounds, rng, scale=scale)


def _sample_mimo_plant(
    coupling_family: str,
    parameter_family: str,
    rng: np.random.Generator,
) -> ParameterizedSystem:
    wn1 = rng.uniform(0.8, 1.8)
    wn2 = rng.uniform(0.6, 1.6)
    if parameter_family == "fast":
        wn1 *= 1.8
        wn2 *= 1.8
    elif parameter_family == "stiff":
        wn2 *= 2.8
    zeta = {"balanced": 0.6, "fast": 0.7, "oscillatory": 0.22, "stiff": 0.5}[parameter_family]
    block1 = np.array([[0.0, 1.0], [-wn1**2, -2.0 * zeta * wn1]], dtype=float)
    block2 = np.array([[0.0, 1.0], [-wn2**2, -2.0 * zeta * wn2]], dtype=float)
    a = np.block(
        [
            [block1, np.zeros((2, 2), dtype=float)],
            [np.zeros((2, 2), dtype=float), block2],
        ]
    )
    coupling_strength = {
        "near_diagonal": 0.05,
        "weak_coupled": 0.18,
        "strong_coupled": 0.45,
    }[coupling_family]
    coupling = coupling_strength * rng.normal(size=(4, 4))
    np.fill_diagonal(coupling, 0.0)
    a = a + coupling
    max_real = np.max(np.real(np.linalg.eigvals(a)))
    if max_real > -0.25:
        a -= np.eye(4, dtype=float) * (max_real + 0.5)

    b = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    b += coupling_strength * 0.35 * rng.normal(size=b.shape)
    c = np.array(
        [
            [1.0, 0.0, coupling_strength, 0.0],
            [0.0, coupling_strength, 1.0, 0.0],
        ],
        dtype=float,
    )
    d = coupling_strength * 0.05 * rng.normal(size=(2, 2))
    ast = SSNode(
        a=matrix_to_tuple(a),
        b=matrix_to_tuple(b),
        c=matrix_to_tuple(c),
        d=matrix_to_tuple(d),
        states=4,
        inputs=2,
        outputs=2,
        name=f"plant_2x2__{coupling_family}",
    )
    return tf_from_node(ast)


def _sample_mimo_controller(
    config: SamplingConfig,
    rng: np.random.Generator,
    parameter_family: str,
) -> ParameterizedSystem:
    systems = []
    for idx in range(2):
        pid = PIDNode(
            kp=_sample_pid_gain(config.pid_kp_range, rng, parameter_family, gain_type="kp"),
            ki=_sample_pid_gain(config.pid_ki_range, rng, parameter_family, gain_type="ki"),
            kd=_sample_pid_gain(config.pid_kd_range, rng, parameter_family, gain_type="kd"),
            tau=0.04,
            name=f"controller_{idx}",
        )
        systems.append(tf_from_node(pid))
    combined = block_diagonal_systems(systems)
    return ParameterizedSystem(
        a=combined.a,
        b=combined.b,
        c=combined.c,
        d=combined.d,
        ast=SSNode(
            a=matrix_to_tuple(combined.a),
            b=matrix_to_tuple(combined.b),
            c=matrix_to_tuple(combined.c),
            d=matrix_to_tuple(combined.d),
            states=combined.state_dimension,
            inputs=combined.input_channels,
            outputs=combined.output_channels,
            name="mimo_controller__diag_pid",
        ),
        dsl_text=combined.dsl_text,
    )


def _sample_input_spec(
    input_channels: int,
    input_family: str,
    rng: np.random.Generator,
) -> InputSpec:
    if input_family == "standard":
        kind = str(rng.choice(["step", "impulse", "ramp", "sine"]))
    else:
        kind = str(rng.choice(["multistep", "chirp", "prbs"]))
    channel_mode = "all" if input_channels == 1 or rng.random() < 0.45 else "single"
    active_channels: tuple[int, ...] = tuple()
    if channel_mode == "single" and input_channels > 1:
        active_channels = (int(rng.integers(0, input_channels)),)
    return InputSpec(
        kind=kind,
        amplitude=float(rng.uniform(0.5, 2.0)),
        frequency=float(rng.uniform(0.1, 1.0)),
        channel_mode=channel_mode,
        active_channels=active_channels,
        pattern_family=input_family,
    )


def _choose_system_type(
    dataset_config: DatasetConfig,
    grammar_config: GrammarConfig,
    rng: np.random.Generator,
) -> str:
    if dataset_config.system_mode == "siso":
        return "siso"
    if dataset_config.system_mode == "mimo":
        return "mimo2x2"
    return _choose_weighted(grammar_config.system_type_weights, rng)


def _controller_family_for_structure(structure_family: str) -> str:
    if structure_family == "open_loop_plant":
        return "none"
    if structure_family in {"lead_lag_feedback", "cascade_compensator"}:
        return "classical_compensator"
    return "pid_family"


def _contains_marker(node: ControlNode, marker: str) -> bool:
    return _extract_marker(node, marker) is not None


def _extract_marker(node: ControlNode, marker: str) -> str | None:
    name = getattr(node, "name", "")
    if isinstance(name, str) and marker in name:
        suffix = name.split(marker, maxsplit=1)[1]
        return suffix if suffix else marker
    if isinstance(node, SeriesNode | ParallelNode):
        for block in node.blocks:
            value = _extract_marker(block, marker)
            if value is not None:
                return value
    if isinstance(node, FeedbackNode):
        forward_value = _extract_marker(node.forward, marker)
        if forward_value is not None:
            return forward_value
        return _extract_marker(node.feedback, marker)
    return None


def _choose_weighted(weights: dict[str, float], rng: np.random.Generator) -> str:
    keys = list(weights)
    probs = np.asarray([weights[key] for key in keys], dtype=float)
    probs = probs / probs.sum()
    return keys[int(rng.choice(len(keys), p=probs))]


def _sample_range(
    bounds: tuple[float, float],
    rng: np.random.Generator,
    scale: float = 1.0,
) -> float:
    low = bounds[0] * scale
    high = bounds[1] * scale
    return float(rng.uniform(low, high))


def _sample_id(seed: int, sample_index: int) -> str:
    value = f"{seed}:{sample_index}".encode("utf-8")
    return hashlib.sha1(value).hexdigest()[:12]
