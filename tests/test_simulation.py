import pytest

from controlgen.dataset import build_sample
from controlgen.dsl import parse_dsl
from controlgen.simulate import InputSpec, SimulationConfig, simulate
from controlgen.transfer_function import tf_from_node


def test_first_order_step_response_tracks_expected_final_value() -> None:
    system = tf_from_node(parse_dsl("tf(num=[1.0], den=[1.0, 1.0], name='plant')"))
    trajectory = simulate(
        system,
        InputSpec(kind="step", amplitude=1.0),
        SimulationConfig(duration=6.0, dt=0.01),
    )

    assert trajectory.y[-1, 0] == pytest.approx(1.0, abs=0.02)
    assert trajectory.y[100, 0] < trajectory.y[-1, 0]


def test_build_sample_records_nested_metrics() -> None:
    system = tf_from_node(parse_dsl("tf(num=[2.0], den=[1.0, 2.0], name='plant')"))
    trajectory = simulate(
        system,
        InputSpec(kind="step", amplitude=1.0),
        SimulationConfig(duration=5.0, dt=0.01),
    )
    sample = build_sample(
        system,
        trajectory,
        {
            "sample_id": "demo",
            "seed": 7,
            "structure_family": "open_loop_plant",
            "parameter_family": "balanced",
            "input_family": "standard",
            "controller_family": "none",
            "coupling_level": "none",
        },
    )

    assert sample.metrics["system_metrics"]["stable"] is True
    assert sample.metrics["channel_metrics"][0]["steady_state_error"] == pytest.approx(0.0, abs=0.05)
    assert sample.tags["state_dimension"] == 1
