from controlgen.generator import DatasetConfig, GrammarConfig, generate_sample


def test_sample_generation_is_reproducible() -> None:
    config = DatasetConfig(seed=123, count=1, system_mode="siso")
    sample_a = generate_sample(config, sample_index=0)
    sample_b = generate_sample(config, sample_index=0)

    assert sample_a.dsl_text == sample_b.dsl_text
    assert sample_a.input_spec == sample_b.input_spec
    assert sample_a.metrics == sample_b.metrics
    assert sample_a.y == sample_b.y


def test_mimo_sample_generation_produces_two_channel_trajectories() -> None:
    config = DatasetConfig(seed=7, count=1, system_mode="mimo")
    grammar = GrammarConfig(system_type_weights={"mimo2x2": 1.0})
    sample = generate_sample(config, grammar_config=grammar, sample_index=0)

    assert sample.input_channels == 2
    assert sample.output_channels == 2
    assert len(sample.u[0]) == 2
    assert len(sample.y[0]) == 2
    assert sample.tags["io_shape"] == "2x2"
