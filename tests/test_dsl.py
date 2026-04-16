from controlgen.dsl import ast_to_dict, parse_dsl, serialize_dsl
from controlgen.types import ControlGraph, TFNode


def test_parse_and_serialize_graph_roundtrip() -> None:
    dsl = (
        "graph("
        "topology_family='layered_mimo_graph', "
        "nodes=["
        "node(id='ref_0', kind='reference_source', family='step', layer='reference', "
        "inputs=[], outputs=[port(name='y', direction='output', dimension=2, role='reference')], parameters={}, metadata={}), "
        "node(id='ctrl_0', kind='controller_bank', family='p', layer='controller', "
        "inputs=[port(name='u', direction='input', dimension=2, role='controller_input')], "
        "outputs=[port(name='y', direction='output', dimension=2, role='controller_output')], parameters={}, metadata={})"
        "], "
        "edges=[edge(id='edge_0', source='ref_0', source_port='y', target='ctrl_0', target_port='u', matrix=[[1.0, 0.0], [0.0, 1.0]], role='reference')], "
        "input_ports=[port(name='reference', direction='input', dimension=2, role='external_input')], "
        "output_ports=[port(name='system_output', direction='output', dimension=2, role='primary_output')], "
        "taps=[tap(id='tap_ref', source='ref_0', source_port='y', role='reference', visibility='primary', dimension=2)], "
        "metadata={'primary_output_node': 'ctrl_0', 'primary_output_port': 'y'})"
    )
    graph = parse_dsl(dsl)
    rendered = serialize_dsl(graph)
    reparsed = parse_dsl(rendered)

    assert isinstance(graph, ControlGraph)
    assert ast_to_dict(graph) == ast_to_dict(reparsed)


def test_low_level_tf_parse_still_works() -> None:
    node = parse_dsl("tf(num=[1.0], den=[1.0, 1.0], name='plant')")
    assert isinstance(node, TFNode)
