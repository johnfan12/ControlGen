import pytest

from controlgen.dataset import build_sample
from controlgen.dsl import parse_dsl
from controlgen.simulate import InputSpec, SimulationConfig, simulate
from controlgen.transfer_function import tf_from_node


def test_first_order_step_response_tracks_expected_final_value() -> None:
    system = tf_from_node(parse_dsl("tf(num=[1.0], den=[1.0, 1.0], name='plant')"))
    trajectory = simulate(system, InputSpec(kind="step", amplitude=1.0), SimulationConfig(duration=6.0, dt=0.01))

    assert trajectory.y[-1] == pytest.approx(1.0, abs=0.02)
    assert trajectory.y[100] < trajectory.y[-1]


def test_build_sample_records_step_metrics() -> None:
    system = tf_from_node(parse_dsl("tf(num=[2.0], den=[1.0, 2.0], name='plant')"))
    trajectory = simulate(system, InputSpec(kind="step", amplitude=1.0), SimulationConfig(duration=5.0, dt=0.01))
    sample = build_sample(system, trajectory, {"sample_id": "demo", "seed": 7})

    assert sample.metrics["stable"] is True
    assert sample.metrics["steady_state_error"] == pytest.approx(0.0, abs=0.05)
    assert sample.tags["order"] == 1
