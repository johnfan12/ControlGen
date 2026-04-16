import pytest

from controlgen.generator import DatasetConfig, GrammarConfig, SamplingConfig, generate_graph, sample_parameters, sample_scenario
from controlgen.simulate import SimulationConfig, simulate


def test_saturated_actuator_clips_control_signal() -> None:
    graph, _ = generate_graph(
        GrammarConfig(structure_family_weights={"saturated_actuator_loop": 1.0}),
        rng=__import__("numpy").random.default_rng(5),
        structure_family="saturated_actuator_loop",
    )
    parameterized = sample_parameters(graph, SamplingConfig(), __import__("numpy").random.default_rng(5))
    scenario = sample_scenario(
        graph,
        DatasetConfig(seed=5, sim_config=SimulationConfig(duration=6.0, dt=0.01)),
        __import__("numpy").random.default_rng(5),
        sample_seed=5,
    )
    bundle = simulate(parameterized, scenario)

    assert max(abs(value) for value in bundle.signals["u_act"]) <= parameterized.actuator_saturation_limit + 1e-6
    assert any(abs(a - b) > 1e-5 for a, b in zip(bundle.signals["u_act"], bundle.signals["u_cmd"], strict=False))


def test_disturbance_loop_produces_nonzero_disturbance_signal() -> None:
    graph, _ = generate_graph(
        GrammarConfig(structure_family_weights={"disturbance_rejection_loop": 1.0}),
        rng=__import__("numpy").random.default_rng(11),
        structure_family="disturbance_rejection_loop",
    )
    parameterized = sample_parameters(graph, SamplingConfig(), __import__("numpy").random.default_rng(11))
    scenario = sample_scenario(
        graph,
        DatasetConfig(seed=11, sim_config=SimulationConfig(duration=8.0, dt=0.01)),
        __import__("numpy").random.default_rng(11),
        sample_seed=11,
    )
    bundle = simulate(parameterized, scenario)

    assert any(abs(value) > 1e-9 for value in bundle.signals["d"])
    assert max(abs(a - b) for a, b in zip(bundle.signals["y_m"], bundle.signals["y"], strict=False)) > 1e-4
    assert bundle.signals["d"][-1] != 0.0
