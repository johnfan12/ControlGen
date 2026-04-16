import numpy as np

from controlgen.generator import DatasetConfig, GrammarConfig, SamplingConfig, generate_graph, sample_parameters, sample_scenario
from controlgen.simulate import SimulationConfig, simulate_graph


def test_graph_simulation_shapes_are_consistent() -> None:
    graph, _ = generate_graph(
        GrammarConfig(graph_size_range=(16, 16), io_channels_range=(4, 4)),
        rng=np.random.default_rng(5),
    )
    parameterized = sample_parameters(graph, SamplingConfig(), np.random.default_rng(5))
    scenario = sample_scenario(
        graph,
        DatasetConfig(seed=5, sim_config=SimulationConfig(duration=4.0, dt=0.02)),
        np.random.default_rng(5),
        sample_seed=5,
    )
    bundle = simulate_graph(parameterized, scenario)

    assert bundle.primary_outputs["system_output"].shape[0] == len(bundle.t)
    assert bundle.primary_outputs["system_output"].shape[1] == parameterized.graph_metrics["output_channels"]
    assert len(bundle.tap_signals) == len(graph.taps)


def test_saturated_actuator_graph_produces_clipped_taps() -> None:
    grammar = GrammarConfig(
        actuator_family_weights={"lag_saturation": 1.0},
        io_channels_range=(4, 4),
        graph_size_range=(14, 14),
    )
    graph, _ = generate_graph(grammar, rng=np.random.default_rng(11))
    parameterized = sample_parameters(graph, SamplingConfig(), np.random.default_rng(11))
    scenario = sample_scenario(
        graph,
        DatasetConfig(seed=11, sim_config=SimulationConfig(duration=5.0, dt=0.01)),
        np.random.default_rng(11),
        sample_seed=11,
    )
    bundle = simulate_graph(parameterized, scenario)

    actuator_limits = {
        node.spec.node_id: node.saturation_limit
        for node in parameterized.nodes
        if node.spec.kind == "actuator_bank" and node.saturation_limit is not None
    }
    assert actuator_limits
    for tap_id, values in bundle.tap_signals.items():
        if tap_id.startswith("tap_act_"):
            node_id = tap_id.removeprefix("tap_")
            limit = actuator_limits[node_id]
            assert float(np.max(np.abs(values))) <= limit + 1e-6


def test_disturbance_and_noise_sources_are_nonzero() -> None:
    graph, _ = generate_graph(
        GrammarConfig(
            disturbance_family_weights={"filtered_noise": 1.0},
            noise_family_weights={"white_noise": 1.0},
            graph_size_range=(15, 15),
        ),
        rng=np.random.default_rng(17),
    )
    parameterized = sample_parameters(graph, SamplingConfig(), np.random.default_rng(17))
    scenario = sample_scenario(
        graph,
        DatasetConfig(seed=17, sim_config=SimulationConfig(duration=6.0, dt=0.01)),
        np.random.default_rng(17),
        sample_seed=17,
    )
    bundle = simulate_graph(parameterized, scenario)

    assert any(np.any(np.abs(values) > 1e-9) for values in bundle.disturbance_signals.values())
    assert any(np.any(np.abs(values) > 1e-9) for values in bundle.noise_signals.values())
