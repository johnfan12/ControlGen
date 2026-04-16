from __future__ import annotations

import ast
from typing import Any

from controlgen.types import (
    ControlGraph,
    ControlNode,
    DelayNode,
    FeedbackNode,
    GainNode,
    GraphEdgeSpec,
    GraphNodeSpec,
    Matrix,
    MatrixGainNode,
    PIDNode,
    ParallelNode,
    PortSpec,
    SSNode,
    SeriesNode,
    TFNode,
    TapSpec,
)


class DSLParseError(ValueError):
    """Raised when a DSL string cannot be parsed."""


def parse_dsl(text: str) -> ControlNode | ControlGraph:
    try:
        expr = ast.parse(text, mode="eval").body
    except SyntaxError as exc:
        raise DSLParseError(str(exc)) from exc
    return _parse_expr(expr)


def serialize_dsl(node: ControlNode | ControlGraph) -> str:
    if isinstance(node, ControlGraph):
        nodes = ", ".join(serialize_dsl(item) for item in node.nodes)
        edges = ", ".join(serialize_dsl(item) for item in node.edges)
        input_ports = ", ".join(serialize_dsl(item) for item in node.input_ports)
        output_ports = ", ".join(serialize_dsl(item) for item in node.output_ports)
        taps = ", ".join(serialize_dsl(item) for item in node.taps)
        return (
            "graph("
            f"topology_family={node.topology_family!r}, "
            f"nodes=[{nodes}], "
            f"edges=[{edges}], "
            f"input_ports=[{input_ports}], "
            f"output_ports=[{output_ports}], "
            f"taps=[{taps}], "
            f"metadata={_format_value(node.metadata)})"
        )
    if isinstance(node, GraphNodeSpec):
        input_ports = ", ".join(serialize_dsl(item) for item in node.input_ports)
        output_ports = ", ".join(serialize_dsl(item) for item in node.output_ports)
        return (
            "node("
            f"id={node.node_id!r}, "
            f"kind={node.kind!r}, "
            f"family={node.family!r}, "
            f"layer={node.layer!r}, "
            f"inputs=[{input_ports}], "
            f"outputs=[{output_ports}], "
            f"parameters={_format_value(node.parameters)}, "
            f"metadata={_format_value(node.metadata)})"
        )
    if isinstance(node, GraphEdgeSpec):
        return (
            "edge("
            f"id={node.edge_id!r}, "
            f"source={node.source_node!r}, "
            f"source_port={node.source_port!r}, "
            f"target={node.target_node!r}, "
            f"target_port={node.target_port!r}, "
            f"matrix={_format_value(node.matrix)}, "
            f"role={node.signal_role!r})"
        )
    if isinstance(node, PortSpec):
        return (
            "port("
            f"name={node.name!r}, "
            f"direction={node.direction!r}, "
            f"dimension={node.dimension!r}, "
            f"role={node.role!r})"
        )
    if isinstance(node, TapSpec):
        return (
            "tap("
            f"id={node.tap_id!r}, "
            f"source={node.source_node!r}, "
            f"source_port={node.source_port!r}, "
            f"role={node.signal_role!r}, "
            f"visibility={node.visibility!r}, "
            f"dimension={node.dimension!r})"
        )
    if isinstance(node, TFNode):
        return (
            f"tf(num={_format_value(node.num)}, den={_format_value(node.den)}, "
            f"name={node.name!r})"
        )
    if isinstance(node, GainNode):
        return f"gain(k={_format_value(node.k)}, name={node.name!r})"
    if isinstance(node, MatrixGainNode):
        return (
            "matgain("
            f"k={_format_value(node.k)}, "
            f"rows={node.rows!r}, "
            f"cols={node.cols!r}, "
            f"name={node.name!r})"
        )
    if isinstance(node, PIDNode):
        return (
            "pid("
            f"kp={_format_value(node.kp)}, "
            f"ki={_format_value(node.ki)}, "
            f"kd={_format_value(node.kd)}, "
            f"tau={_format_value(node.tau)}, "
            f"name={node.name!r})"
        )
    if isinstance(node, DelayNode):
        return f"delay(t={_format_value(node.t)}, order={node.order}, name={node.name!r})"
    if isinstance(node, SSNode):
        return (
            "ss("
            f"a={_format_value(node.a)}, "
            f"b={_format_value(node.b)}, "
            f"c={_format_value(node.c)}, "
            f"d={_format_value(node.d)}, "
            f"states={node.states!r}, "
            f"inputs={node.inputs!r}, "
            f"outputs={node.outputs!r}, "
            f"name={node.name!r})"
        )
    if isinstance(node, SeriesNode):
        return f"series({', '.join(serialize_dsl(block) for block in node.blocks)})"
    if isinstance(node, ParallelNode):
        return f"parallel({', '.join(serialize_dsl(block) for block in node.blocks)})"
    if isinstance(node, FeedbackNode):
        return (
            "feedback("
            f"forward={serialize_dsl(node.forward)}, "
            f"feedback={serialize_dsl(node.feedback)}, "
            f"sign={node.sign})"
        )
    raise TypeError(f"Unsupported DSL object: {type(node)!r}")


def ast_to_dict(node: ControlNode | ControlGraph | GraphNodeSpec | GraphEdgeSpec | PortSpec | TapSpec | None) -> dict[str, Any] | None:
    if node is None:
        return None
    if isinstance(node, ControlGraph):
        return {
            "type": "graph",
            "topology_family": node.topology_family,
            "nodes": [ast_to_dict(item) for item in node.nodes],
            "edges": [ast_to_dict(item) for item in node.edges],
            "input_ports": [ast_to_dict(item) for item in node.input_ports],
            "output_ports": [ast_to_dict(item) for item in node.output_ports],
            "taps": [ast_to_dict(item) for item in node.taps],
            "metadata": node.metadata,
        }
    if isinstance(node, GraphNodeSpec):
        return {
            "type": "node",
            "id": node.node_id,
            "kind": node.kind,
            "family": node.family,
            "layer": node.layer,
            "inputs": [ast_to_dict(item) for item in node.input_ports],
            "outputs": [ast_to_dict(item) for item in node.output_ports],
            "parameters": node.parameters,
            "metadata": node.metadata,
        }
    if isinstance(node, GraphEdgeSpec):
        return {
            "type": "edge",
            "id": node.edge_id,
            "source": node.source_node,
            "source_port": node.source_port,
            "target": node.target_node,
            "target_port": node.target_port,
            "matrix": _to_json_value(node.matrix),
            "role": node.signal_role,
        }
    if isinstance(node, PortSpec):
        return {
            "type": "port",
            "name": node.name,
            "direction": node.direction,
            "dimension": node.dimension,
            "role": node.role,
        }
    if isinstance(node, TapSpec):
        return {
            "type": "tap",
            "id": node.tap_id,
            "source": node.source_node,
            "source_port": node.source_port,
            "role": node.signal_role,
            "visibility": node.visibility,
            "dimension": node.dimension,
        }
    if isinstance(node, TFNode):
        return {"type": "tf", "num": _to_json_value(node.num), "den": _to_json_value(node.den), "name": node.name}
    if isinstance(node, GainNode):
        return {"type": "gain", "k": node.k, "name": node.name}
    if isinstance(node, MatrixGainNode):
        return {
            "type": "matgain",
            "k": _to_json_value(node.k),
            "rows": node.rows,
            "cols": node.cols,
            "name": node.name,
        }
    if isinstance(node, PIDNode):
        return {
            "type": "pid",
            "kp": node.kp,
            "ki": node.ki,
            "kd": node.kd,
            "tau": node.tau,
            "name": node.name,
        }
    if isinstance(node, DelayNode):
        return {"type": "delay", "t": node.t, "order": node.order, "name": node.name}
    if isinstance(node, SSNode):
        return {
            "type": "ss",
            "a": _to_json_value(node.a),
            "b": _to_json_value(node.b),
            "c": _to_json_value(node.c),
            "d": _to_json_value(node.d),
            "states": node.states,
            "inputs": node.inputs,
            "outputs": node.outputs,
            "name": node.name,
        }
    if isinstance(node, SeriesNode):
        return {"type": "series", "blocks": [ast_to_dict(block) for block in node.blocks]}
    if isinstance(node, ParallelNode):
        return {"type": "parallel", "blocks": [ast_to_dict(block) for block in node.blocks]}
    if isinstance(node, FeedbackNode):
        return {
            "type": "feedback",
            "forward": ast_to_dict(node.forward),
            "feedback": ast_to_dict(node.feedback),
            "sign": node.sign,
        }
    raise TypeError(f"Unsupported DSL object: {type(node)!r}")


def _parse_expr(expr: ast.AST) -> ControlNode | ControlGraph | GraphNodeSpec | GraphEdgeSpec | PortSpec | TapSpec:
    if not isinstance(expr, ast.Call):
        raise DSLParseError("Top-level expression must be a function call")
    if not isinstance(expr.func, ast.Name):
        raise DSLParseError("Only named DSL functions are allowed")

    name = expr.func.id
    kwargs = {kw.arg: _literal(kw.value) for kw in expr.keywords if kw.arg is not None}

    if name == "graph":
        nodes_literal = kwargs.get("nodes", [])
        edges_literal = kwargs.get("edges", [])
        input_ports_literal = kwargs.get("input_ports", [])
        output_ports_literal = kwargs.get("output_ports", [])
        taps_literal = kwargs.get("taps", [])
        if not all(isinstance(item, list) for item in (nodes_literal, edges_literal, input_ports_literal, output_ports_literal, taps_literal)):
            raise DSLParseError("graph(...) nodes/edges/ports/taps must use list syntax")
        return ControlGraph(
            topology_family=str(kwargs.get("topology_family", "layered_mimo_graph")),
            nodes=tuple(_parse_graph_node(item) for item in nodes_literal),
            edges=tuple(_parse_graph_edge(item) for item in edges_literal),
            input_ports=tuple(_parse_port(item) for item in input_ports_literal),
            output_ports=tuple(_parse_port(item) for item in output_ports_literal),
            taps=tuple(_parse_tap(item) for item in taps_literal),
            metadata=_as_dict(kwargs.get("metadata")),
        )
    if name == "node":
        input_ports_literal = kwargs.get("inputs", [])
        output_ports_literal = kwargs.get("outputs", [])
        if not isinstance(input_ports_literal, list) or not isinstance(output_ports_literal, list):
            raise DSLParseError("node(..., inputs=[...], outputs=[...]) requires lists")
        return GraphNodeSpec(
            node_id=str(kwargs.get("id", "node")),
            kind=str(kwargs.get("kind", "process_unit")),
            family=str(kwargs.get("family", "generic")),
            layer=str(kwargs.get("layer", "process")),
            input_ports=tuple(_parse_port(item) for item in input_ports_literal),
            output_ports=tuple(_parse_port(item) for item in output_ports_literal),
            parameters=_as_dict(kwargs.get("parameters")),
            metadata=_as_dict(kwargs.get("metadata")),
        )
    if name == "edge":
        return GraphEdgeSpec(
            edge_id=str(kwargs.get("id", "edge")),
            source_node=str(kwargs.get("source", "")),
            source_port=str(kwargs.get("source_port", "y")),
            target_node=str(kwargs.get("target", "")),
            target_port=str(kwargs.get("target_port", "u")),
            matrix=_as_matrix(kwargs.get("matrix")),
            signal_role=str(kwargs.get("role", "internal")),
        )
    if name == "port":
        return PortSpec(
            name=str(kwargs.get("name", "port")),
            direction=str(kwargs.get("direction", "input")),
            dimension=int(kwargs.get("dimension", 1)),
            role=str(kwargs.get("role", "internal")),
        )
    if name == "tap":
        return TapSpec(
            tap_id=str(kwargs.get("id", "tap")),
            source_node=str(kwargs.get("source", "")),
            source_port=str(kwargs.get("source_port", "y")),
            signal_role=str(kwargs.get("role", "internal")),
            visibility=str(kwargs.get("visibility", "primary")),
            dimension=int(kwargs.get("dimension", 1)),
        )
    if name == "tf":
        return TFNode(
            num=_as_tuple(kwargs.get("num")),
            den=_as_tuple(kwargs.get("den")),
            name=str(kwargs.get("name", "tf")),
        )
    if name == "gain":
        return GainNode(k=_as_number(kwargs.get("k")), name=str(kwargs.get("name", "gain")))
    if name == "matgain":
        return MatrixGainNode(
            k=_as_matrix(kwargs.get("k")),
            rows=int(kwargs["rows"]) if kwargs.get("rows") is not None else None,
            cols=int(kwargs["cols"]) if kwargs.get("cols") is not None else None,
            name=str(kwargs.get("name", "matgain")),
        )
    if name == "pid":
        return PIDNode(
            kp=_as_number(kwargs.get("kp")),
            ki=_as_number(kwargs.get("ki")),
            kd=_as_number(kwargs.get("kd")),
            tau=float(kwargs.get("tau", 0.05)),
            name=str(kwargs.get("name", "pid")),
        )
    if name == "delay":
        return DelayNode(
            t=_as_number(kwargs.get("t")),
            order=int(kwargs.get("order", 1)),
            name=str(kwargs.get("name", "delay")),
        )
    if name == "ss":
        return SSNode(
            a=_as_matrix(kwargs.get("a")),
            b=_as_matrix(kwargs.get("b")),
            c=_as_matrix(kwargs.get("c")),
            d=_as_matrix(kwargs.get("d")),
            states=int(kwargs["states"]) if kwargs.get("states") is not None else None,
            inputs=int(kwargs["inputs"]) if kwargs.get("inputs") is not None else None,
            outputs=int(kwargs["outputs"]) if kwargs.get("outputs") is not None else None,
            name=str(kwargs.get("name", "ss")),
        )
    if name == "series":
        blocks = tuple(_parse_control_expr(arg) for arg in expr.args)
        if len(blocks) < 2:
            raise DSLParseError("series(...) requires at least two blocks")
        return SeriesNode(blocks=blocks)
    if name == "parallel":
        blocks = tuple(_parse_control_expr(arg) for arg in expr.args)
        if len(blocks) < 2:
            raise DSLParseError("parallel(...) requires at least two blocks")
        return ParallelNode(blocks=blocks)
    if name == "feedback":
        if "forward" not in kwargs or "feedback" not in kwargs:
            raise DSLParseError("feedback(...) requires forward= and feedback=")
        return FeedbackNode(
            forward=_parse_control_expr(kwargs["forward"]),
            feedback=_parse_control_expr(kwargs["feedback"]),
            sign=int(kwargs.get("sign", -1)),
        )
    raise DSLParseError(f"Unknown DSL function: {name}")


def _parse_control_expr(value: Any) -> ControlNode:
    node = _parse_module_expr(value, "control")
    if isinstance(node, (ControlGraph, GraphNodeSpec, GraphEdgeSpec, PortSpec, TapSpec)):
        raise DSLParseError("Graph DSL cannot be nested inside low-level control nodes")
    return node


def _parse_module_expr(value: Any, label: str) -> Any:
    if not isinstance(value, ast.AST):
        raise DSLParseError(f"{label} must be a nested DSL expression")
    return _parse_expr(value)


def _parse_graph_node(value: Any) -> GraphNodeSpec:
    node = _parse_module_expr(value, "node")
    if not isinstance(node, GraphNodeSpec):
        raise DSLParseError("Expected node(...) in graph nodes list")
    return node


def _parse_graph_edge(value: Any) -> GraphEdgeSpec:
    edge = _parse_module_expr(value, "edge")
    if not isinstance(edge, GraphEdgeSpec):
        raise DSLParseError("Expected edge(...) in graph edges list")
    return edge


def _parse_port(value: Any) -> PortSpec:
    port = _parse_module_expr(value, "port")
    if not isinstance(port, PortSpec):
        raise DSLParseError("Expected port(...)")
    return port


def _parse_tap(value: Any) -> TapSpec:
    tap = _parse_module_expr(value, "tap")
    if not isinstance(tap, TapSpec):
        raise DSLParseError("Expected tap(...)")
    return tap


def _literal(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_literal(element) for element in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_literal(element) for element in node.elts)
    if isinstance(node, ast.Dict):
        return {_literal(key): _literal(value) for key, value in zip(node.keys, node.values, strict=False)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_literal(node.operand)
    if isinstance(node, ast.Call):
        return node
    raise DSLParseError(f"Unsupported literal in DSL: {ast.dump(node)}")


def _as_tuple(value: Any) -> tuple[float, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise DSLParseError("Expected tuple/list of numbers")
    return tuple(float(v) for v in value)


def _as_number(value: Any) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise DSLParseError("Expected numeric value")
    return float(value)


def _as_matrix(value: Any) -> Matrix | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise DSLParseError("Expected matrix as list/tuple of rows")
    rows = []
    for row in value:
        if not isinstance(row, (list, tuple)):
            raise DSLParseError("Expected matrix row as list/tuple")
        rows.append(tuple(float(item) for item in row))
    return tuple(rows)


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise DSLParseError("Expected dictionary literal")
    return {str(key): item for key, item in value.items()}


def _format_value(value: Any) -> str:
    if isinstance(value, tuple):
        return repr(list(value))
    return repr(value)


def _to_json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_to_json_value(item) for item in value]
    if isinstance(value, list):
        return [_to_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_json_value(item) for key, item in value.items()}
    return value
