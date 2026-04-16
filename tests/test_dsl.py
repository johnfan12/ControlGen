from controlgen.dsl import ast_to_dict, parse_dsl, serialize_dsl
from controlgen.types import ControlGraph, TFNode


def test_parse_and_serialize_graph_roundtrip() -> None:
    dsl = (
        "loop("
        "reference=reference(kind='step', name='reference'), "
        "sum=sum(signs=[1, -1], name='error_sum'), "
        "controller=controller(kind='pid', name='controller'), "
        "actuator=actuator(kind='lag_saturation', name='actuator'), "
        "plant=plant(kind='delay_plus_lag', name='plant'), "
        "disturbance=disturbance(kind='load_step', injection='output', name='disturbance'), "
        "sensor=sensor(kind='lag', name='sensor'), "
        "noise=noise(kind='white_noise', name='measurement_noise'), "
        "taps=[tap(signal='r', name='tap_r'), tap(signal='y', name='tap_y')])"
    )
    graph = parse_dsl(dsl)
    rendered = serialize_dsl(graph)
    reparsed = parse_dsl(rendered)

    assert isinstance(graph, ControlGraph)
    assert ast_to_dict(graph) == ast_to_dict(reparsed)


def test_low_level_tf_parse_still_works() -> None:
    node = parse_dsl("tf(num=[1.0], den=[1.0, 1.0], name='plant')")
    assert isinstance(node, TFNode)
