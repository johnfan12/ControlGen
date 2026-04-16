from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from controlgen.dataset import DatasetSample, build_sample
from controlgen.simulate import ScenarioSpec, SignalSpec, SimulationConfig, simulate
from controlgen.transfer_function import matrix_to_tuple
from controlgen.types import (
    ActuatorNode,
    ControlGraph,
    ControlNode,
    ControllerNode,
    DelayNode,
    DisturbanceNode,
    GainNode,
    NoiseNode,
    PIDNode,
    ParameterizedGraph,
    PlantNode,
    ReferenceNode,
    SensorNode,
    SeriesNode,
    SumNode,
    TFNode,
    TapNode,
)


@dataclass(frozen=True)
class GrammarConfig:
    structure_family_weights: dict[str, float] = field(
        default_factory=lambda: {
            "basic_pid_loop": 0.18,
            "actuator_lag_loop": 0.20,
            "sensor_filter_loop": 0.20,
            "disturbance_rejection_loop": 0.22,
            "saturated_actuator_loop": 0.20,
        }
    )
    controller_family_weights: dict[str, float] = field(
        default_factory=lambda: {"p": 0.10, "pi": 0.35, "pid": 0.40, "lead_lag": 0.15}
    )
    plant_family_weights: dict[str, float] = field(
        default_factory=lambda: {
            "first_order": 0.25,
            "second_order": 0.25,
            "integrator_plus_lag": 0.15,
            "delay_plus_lag": 0.20,
            "oscillatory_process": 0.15,
        }
    )
    reference_family_weights: dict[str, float] = field(
        default_factory=lambda: {"step": 0.40, "multistep": 0.25, "ramp": 0.15, "chirp": 0.20}
    )
    disturbance_family_weights: dict[str, float] = field(
        default_factory=lambda: {
            "none": 0.20,
            "load_step": 0.35,
            "load_pulse": 0.25,
            "filtered_noise": 0.20,
        }
    )
    noise_family_weights: dict[str, float] = field(
        default_factory=lambda: {"none": 0.30, "white_noise": 0.50, "lowpass_noise": 0.20}
    )


@dataclass(frozen=True)
class SamplingConfig:
    controller_gain_range: tuple[float, float] = (0.2, 4.0)
    controller_ki_range: tuple[float, float] = (0.0, 2.5)
    controller_kd_range: tuple[float, float] = (0.0, 0.6)
    lead_zero_range: tuple[float, float] = (0.2, 1.2)
    lead_pole_range: tuple[float, float] = (1.0, 4.0)
    actuator_gain_range: tuple[float, float] = (0.8, 1.4)
    actuator_tau_range: tuple[float, float] = (0.05, 0.4)
    actuator_saturation_range: tuple[float, float] = (0.8, 2.0)
    plant_gain_range: tuple[float, float] = (0.8, 2.5)
    plant_tau_range: tuple[float, float] = (0.5, 3.0)
    plant_delay_range: tuple[float, float] = (0.1, 0.8)
    plant_zeta_range: tuple[float, float] = (0.15, 0.8)
    plant_wn_range: tuple[float, float] = (0.4, 2.5)
    disturbance_gain_range: tuple[float, float] = (0.3, 1.5)
    sensor_gain_range: tuple[float, float] = (0.95, 1.05)
    sensor_tau_range: tuple[float, float] = (0.03, 0.4)
    sensor_bias_range: tuple[float, float] = (-0.05, 0.05)
    noise_std_range: tuple[float, float] = (0.01, 0.08)


@dataclass(frozen=True)
class DatasetConfig:
    count: int = 10
    seed: int = 0
    sim_config: SimulationConfig = field(default_factory=SimulationConfig)


def generate_graph(
    config: GrammarConfig,
    rng: np.random.Generator,
    structure_family: str | None = None,
) -> tuple[ControlGraph, dict[str, str]]:
    family = structure_family or _choose_weighted(config.structure_family_weights, rng)
    controller_family = _choose_weighted(config.controller_family_weights, rng)
    plant_family = _choose_weighted(config.plant_family_weights, rng)
    reference_family = _choose_weighted(config.reference_family_weights, rng)

    actuator_family = "ideal"
    sensor_family = "ideal"
    disturbance_family = "none"
    noise_family = "none"

    if family == "actuator_lag_loop":
        actuator_family = "lag"
    elif family == "sensor_filter_loop":
        sensor_family = "lag"
        noise_family = _choose_weighted({"white_noise": 0.65, "lowpass_noise": 0.35}, rng)
    elif family == "disturbance_rejection_loop":
        disturbance_family = _choose_weighted(
            {"load_step": 0.5, "load_pulse": 0.25, "filtered_noise": 0.25},
            rng,
        )
        sensor_family = "lag"
    elif family == "saturated_actuator_loop":
        actuator_family = "lag_saturation"
        disturbance_family = _choose_weighted({"none": 0.5, "load_step": 0.5}, rng)

    graph = ControlGraph(
        reference=ReferenceNode(kind=reference_family, name="reference"),
        sum_node=SumNode(signs=(1, -1), name="error_sum"),
        controller=ControllerNode(kind=controller_family, name="controller"),
        actuator=ActuatorNode(kind=actuator_family, name="actuator"),
        plant=PlantNode(kind=plant_family, name="plant"),
        disturbance=(
            DisturbanceNode(kind=disturbance_family, injection="output", name="disturbance")
            if disturbance_family != "none"
            else None
        ),
        sensor=SensorNode(kind=sensor_family, name="sensor"),
        noise=NoiseNode(kind=noise_family, name="measurement_noise") if noise_family != "none" else None,
        taps=(
            TapNode(signal="r", name="tap_r"),
            TapNode(signal="d", name="tap_d"),
            TapNode(signal="e", name="tap_e"),
            TapNode(signal="u_cmd", name="tap_u_cmd"),
            TapNode(signal="u_act", name="tap_u_act"),
            TapNode(signal="y", name="tap_y"),
            TapNode(signal="y_m", name="tap_y_m"),
        ),
    )
    metadata = {
        "structure_family": family,
        "controller_family": controller_family,
        "actuator_family": actuator_family,
        "plant_family": plant_family,
        "sensor_family": sensor_family,
        "disturbance_family": disturbance_family,
        "noise_family": noise_family,
    }
    return graph, metadata


def generate_structure(
    config: GrammarConfig,
    rng: np.random.Generator,
    difficulty: str | None = None,
    system_type: str | None = None,
    structure_family: str | None = None,
    controller_family: str | None = None,
) -> ControlGraph:
    del difficulty, system_type, controller_family
    graph, _ = generate_graph(config, rng, structure_family=structure_family)
    return graph


def sample_parameters(
    graph: ControlGraph,
    config: SamplingConfig,
    rng: np.random.Generator,
    parameter_family: str | None = None,
) -> ParameterizedGraph:
    del parameter_family
    controller_node, controller_params = _sample_controller(graph.controller.kind, config, rng)
    actuator_node, saturation_limit, actuator_params = _sample_actuator(graph.actuator.kind, config, rng)
    plant_control_node, plant_dist_node, plant_params = _sample_plant(graph.plant.kind, graph.disturbance, config, rng)
    sensor_node, sensor_bias, sensor_params = _sample_sensor(graph.sensor.kind, config, rng)
    noise_family = graph.noise.kind if graph.noise else "none"
    disturbance_family = graph.disturbance.kind if graph.disturbance else "none"
    return ParameterizedGraph(
        graph=graph,
        structure_family=_structure_family_from_graph(graph),
        controller_family=graph.controller.kind,
        actuator_family=graph.actuator.kind,
        plant_family=graph.plant.kind,
        sensor_family=graph.sensor.kind,
        disturbance_family=disturbance_family,
        noise_family=noise_family,
        controller_dynamics=controller_node,
        actuator_dynamics=actuator_node,
        plant_control_dynamics=plant_control_node,
        plant_disturbance_dynamics=plant_dist_node,
        sensor_dynamics=sensor_node,
        actuator_saturation_limit=saturation_limit,
        sensor_bias=sensor_bias,
        module_params={
            "controller": controller_params,
            "actuator": actuator_params,
            "plant": plant_params,
            "sensor": sensor_params,
            "disturbance": {
                "family": disturbance_family,
                "injection": graph.disturbance.injection if graph.disturbance else "none",
            },
            "noise": {"family": noise_family},
        },
    )


def sample_scenario(
    graph: ControlGraph,
    dataset_config: DatasetConfig,
    rng: np.random.Generator,
    sample_seed: int,
) -> ScenarioSpec:
    duration = dataset_config.sim_config.duration
    reference = _sample_reference_spec(graph.reference.kind, duration, rng, sample_seed + 11)
    disturbance = None
    if graph.disturbance is not None:
        disturbance = _sample_disturbance_spec(graph.disturbance.kind, duration, rng, sample_seed + 29)
    noise = None
    if graph.noise is not None:
        noise = _sample_noise_spec(graph.noise.kind, duration, rng, sample_seed + 47)
    return ScenarioSpec(
        reference=reference,
        disturbance=disturbance,
        noise=noise,
        sim_config=dataset_config.sim_config,
    )


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
    graph, metadata = generate_graph(grammar, rng)
    parameterized_graph = sample_parameters(graph, sampling, rng)
    scenario = sample_scenario(graph, dataset_config, rng, sample_seed)
    signal_bundle = simulate(parameterized_graph, scenario)
    return build_sample(
        parameterized_graph,
        signal_bundle,
        scenario,
        {"sample_id": _sample_id(sample_seed, sample_index), "seed": sample_seed, **metadata},
    )


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


def _sample_controller(
    family: str,
    config: SamplingConfig,
    rng: np.random.Generator,
) -> tuple[ControlNode, dict[str, object]]:
    if family == "p":
        gain = _sample_range(config.controller_gain_range, rng)
        return GainNode(k=gain, name="controller_p"), {"family": family, "gain": gain}
    kp = _sample_range(config.controller_gain_range, rng)
    ki = _sample_range(config.controller_ki_range, rng) if family in {"pi", "pid"} else 0.0
    kd = _sample_range(config.controller_kd_range, rng) if family == "pid" else 0.0
    if family == "lead_lag":
        zero = _sample_range(config.lead_zero_range, rng)
        pole = _sample_range(config.lead_pole_range, rng)
        gain = _sample_range(config.controller_gain_range, rng)
        lead = TFNode(num=(gain, gain * zero), den=(1.0, pole), name="lead_lag")
        return lead, {"family": family, "gain": gain, "zero": zero, "pole": pole}
    node = PIDNode(kp=kp, ki=ki, kd=kd, tau=0.05, name="controller_pid")
    return node, {"family": family, "kp": kp, "ki": ki, "kd": kd, "tau": 0.05}


def _sample_actuator(
    family: str,
    config: SamplingConfig,
    rng: np.random.Generator,
) -> tuple[ControlNode | None, float | None, dict[str, object]]:
    gain = _sample_range(config.actuator_gain_range, rng)
    if family == "ideal":
        return GainNode(k=gain, name="actuator_gain"), None, {"family": family, "gain": gain}
    tau = _sample_range(config.actuator_tau_range, rng)
    node = TFNode(num=(gain,), den=(tau, 1.0), name="actuator_lag")
    saturation_limit = None
    params = {"family": family, "gain": gain, "tau": tau}
    if family == "lag_saturation":
        saturation_limit = _sample_range(config.actuator_saturation_range, rng)
        params["saturation_limit"] = saturation_limit
    return node, saturation_limit, params


def _sample_plant(
    family: str,
    disturbance: DisturbanceNode | None,
    config: SamplingConfig,
    rng: np.random.Generator,
) -> tuple[ControlNode, ControlNode | None, dict[str, object]]:
    gain = _sample_range(config.plant_gain_range, rng)
    tau = _sample_range(config.plant_tau_range, rng)
    if family == "first_order":
        control_path: ControlNode = TFNode(num=(gain,), den=(tau, 1.0), name="plant_first_order")
        params = {"family": family, "gain": gain, "tau": tau}
    elif family == "second_order":
        wn = _sample_range(config.plant_wn_range, rng)
        zeta = _sample_range(config.plant_zeta_range, rng)
        control_path = TFNode(num=(gain * wn**2,), den=(1.0, 2.0 * zeta * wn, wn**2), name="plant_second_order")
        params = {"family": family, "gain": gain, "wn": wn, "zeta": zeta}
    elif family == "integrator_plus_lag":
        control_path = TFNode(num=(gain,), den=(tau, 1.0, 0.0), name="plant_integrator_lag")
        params = {"family": family, "gain": gain, "tau": tau}
    elif family == "delay_plus_lag":
        delay = _sample_range(config.plant_delay_range, rng)
        control_path = SeriesNode(
            (
                DelayNode(t=delay, order=1, name="plant_delay"),
                TFNode(num=(gain,), den=(tau, 1.0), name="plant_lag"),
            )
        )
        params = {"family": family, "gain": gain, "tau": tau, "delay": delay}
    else:
        wn = _sample_range(config.plant_wn_range, rng)
        zeta = _sample_range((0.15, 0.35), rng)
        control_path = TFNode(
            num=(gain * wn**2,),
            den=(1.0, 2.0 * zeta * wn, wn**2),
            name="plant_oscillatory",
        )
        params = {"family": "oscillatory_process", "gain": gain, "wn": wn, "zeta": zeta}

    disturbance_path = None
    if disturbance is not None:
        disturbance_gain = _sample_range(config.disturbance_gain_range, rng)
        disturbance_tau = _sample_range((0.2, 1.4), rng)
        disturbance_path = TFNode(
            num=(disturbance_gain,),
            den=(disturbance_tau, 1.0),
            name="disturbance_path",
        )
        params["disturbance_gain"] = disturbance_gain
        params["disturbance_tau"] = disturbance_tau
    return control_path, disturbance_path, params


def _sample_sensor(
    family: str,
    config: SamplingConfig,
    rng: np.random.Generator,
) -> tuple[ControlNode | None, float, dict[str, object]]:
    gain = _sample_range(config.sensor_gain_range, rng)
    bias = _sample_range(config.sensor_bias_range, rng)
    if family == "ideal":
        return GainNode(k=gain, name="sensor_gain"), bias, {"family": family, "gain": gain, "bias": bias}
    tau = _sample_range(config.sensor_tau_range, rng)
    node = TFNode(num=(gain,), den=(tau, 1.0), name="sensor_lag")
    return node, bias, {"family": family, "gain": gain, "tau": tau, "bias": bias}


def _sample_reference_spec(
    family: str,
    duration: float,
    rng: np.random.Generator,
    seed: int,
) -> SignalSpec:
    if family == "multistep":
        levels = tuple(float(value) for value in rng.uniform(-1.2, 1.8, size=4))
        return SignalSpec(kind=family, amplitude=1.0, start_time=0.2 * duration, levels=levels, seed=seed)
    if family == "ramp":
        return SignalSpec(
            kind=family,
            amplitude=float(rng.uniform(0.1, 0.35)),
            start_time=0.1 * duration,
            seed=seed,
        )
    if family == "chirp":
        return SignalSpec(
            kind=family,
            amplitude=float(rng.uniform(0.6, 1.4)),
            frequency=float(rng.uniform(0.1, 0.4)),
            start_time=0.1 * duration,
            seed=seed,
        )
    return SignalSpec(
        kind="step",
        amplitude=float(rng.uniform(0.8, 1.8)),
        start_time=0.08 * duration,
        seed=seed,
    )


def _sample_disturbance_spec(
    family: str,
    duration: float,
    rng: np.random.Generator,
    seed: int,
) -> SignalSpec:
    start_time = float(rng.uniform(0.25 * duration, 0.6 * duration))
    if family == "load_pulse":
        return SignalSpec(
            kind=family,
            amplitude=float(rng.uniform(0.2, 1.0)),
            start_time=start_time,
            duration=float(rng.uniform(0.1 * duration, 0.25 * duration)),
            seed=seed,
        )
    if family == "filtered_noise":
        return SignalSpec(
            kind=family,
            amplitude=float(rng.uniform(0.08, 0.25)),
            start_time=start_time,
            bandwidth=float(rng.uniform(0.3, 1.2)),
            seed=seed,
        )
    return SignalSpec(
        kind="load_step",
        amplitude=float(rng.uniform(0.2, 1.0)),
        start_time=start_time,
        seed=seed,
    )


def _sample_noise_spec(
    family: str,
    duration: float,
    rng: np.random.Generator,
    seed: int,
) -> SignalSpec:
    amplitude = float(rng.uniform(0.01, 0.06))
    if family == "lowpass_noise":
        return SignalSpec(
            kind=family,
            amplitude=amplitude,
            bandwidth=float(rng.uniform(0.8, 2.0)),
            seed=seed,
        )
    return SignalSpec(kind="white_noise", amplitude=amplitude, seed=seed)


def _choose_weighted(weights: dict[str, float], rng: np.random.Generator) -> str:
    keys = list(weights)
    probs = np.asarray([weights[key] for key in keys], dtype=float)
    probs = probs / probs.sum()
    return keys[int(rng.choice(len(keys), p=probs))]


def _sample_range(bounds: tuple[float, float], rng: np.random.Generator) -> float:
    return float(rng.uniform(bounds[0], bounds[1]))


def _structure_family_from_graph(graph: ControlGraph) -> str:
    if graph.actuator.kind == "lag_saturation":
        return "saturated_actuator_loop"
    if graph.disturbance is not None:
        return "disturbance_rejection_loop"
    if graph.sensor.kind == "lag":
        return "sensor_filter_loop"
    if graph.actuator.kind == "lag":
        return "actuator_lag_loop"
    return "basic_pid_loop"


def _sample_id(seed: int, sample_index: int) -> str:
    value = f"{seed}:{sample_index}".encode("utf-8")
    return hashlib.sha1(value).hexdigest()[:12]
