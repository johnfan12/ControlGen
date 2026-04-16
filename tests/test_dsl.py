from controlgen.dsl import ast_to_dict, parse_dsl, serialize_dsl
from controlgen.types import FeedbackNode, SeriesNode


def test_parse_and_serialize_roundtrip() -> None:
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
