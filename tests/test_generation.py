from controlgen.generator import DatasetConfig, generate_sample


def test_sample_generation_is_reproducible() -> None:
    config = DatasetConfig(seed=123, count=1)
    sample_a = generate_sample(config, sample_index=0)
    sample_b = generate_sample(config, sample_index=0)

    assert sample_a.dsl_text == sample_b.dsl_text
    assert sample_a.input_spec == sample_b.input_spec
    assert sample_a.metrics == sample_b.metrics
    assert sample_a.y == sample_b.y
