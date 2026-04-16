from controlgen.dsl import ast_to_dict, parse_dsl, serialize_dsl
from controlgen.types import FeedbackNode, MatrixGainNode, SeriesNode, SSNode


def test_parse_and_serialize_roundtrip_for_siso() -> None:
    dsl = (
        "feedback("
        "forward=series(pid(kp=1.2, ki=0.4, kd=0.1, tau=0.05, name='controller'), "
        "tf(num=[1.0], den=[1.0, 2.0, 1.0], name='plant')), "
        "feedback=gain(k=1.0, name='sensor'), sign=-1)"
    )
    node = parse_dsl(dsl)
    rendered = serialize_dsl(node)
    reparsed = parse_dsl(rendered)

    assert isinstance(node, FeedbackNode)
    assert isinstance(node.forward, SeriesNode)
    assert ast_to_dict(node) == ast_to_dict(reparsed)


def test_parse_state_space_and_matrix_gain_nodes() -> None:
    dsl = (
        "feedback("
        "forward=series("
        "matgain(k=[[1.0, 0.0], [0.0, 1.0]], rows=2, cols=2, name='decoupler_2x2'), "
        "ss(a=[[0.0, 1.0], [-1.0, -0.5]], b=[[0.0], [1.0]], c=[[1.0, 0.0]], d=[[0.0]], "
        "states=2, inputs=1, outputs=1, name='plant_ss')), "
        "feedback=gain(k=1.0, name='sensor'))"
    )
    node = parse_dsl(dsl)

    assert isinstance(node, FeedbackNode)
    assert isinstance(node.forward, SeriesNode)
    assert isinstance(node.forward.blocks[0], MatrixGainNode)
    assert isinstance(node.forward.blocks[1], SSNode)
