from controlgen.generator import DatasetConfig, generate_sample


def test_sample_generation_is_reproducible() -> None:
    config = DatasetConfig(seed=123, count=1)
    sample_a = generate_sample(config, sample_index=0)
    sample_b = generate_sample(config, sample_index=0)

    assert sample_a.graph_dsl == sample_b.graph_dsl
    assert sample_a.module_params == sample_b.module_params
    assert sample_a.scenario == sample_b.scenario
    assert sample_a.signals == sample_b.signals
    assert sample_a.teacher_signals == sample_b.teacher_signals


def test_generated_sample_contains_grouped_multi_signal_outputs() -> None:
    sample = generate_sample(DatasetConfig(seed=7, count=1), sample_index=0)

    assert "external_inputs" in sample.signals
    assert "primary_outputs" in sample.signals
    assert "tap_signals" in sample.signals
    assert sample.tap_specs
    assert len(sample.graph_nodes) >= 8
    assert len(sample.t) == len(next(iter(sample.signals["primary_outputs"].values())))
