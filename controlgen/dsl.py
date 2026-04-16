from __future__ import annotations

import ast
from typing import Any

from controlgen.types import (
    ActuatorNode,
    ControlGraph,
    ControlNode,
    ControllerNode,
    DelayNode,
    DisturbanceNode,
    FeedbackNode,
    GainNode,
    Matrix,
    MatrixGainNode,
    NoiseNode,
    PIDNode,
    ParallelNode,
    PlantNode,
    ReferenceNode,
    SSNode,
    SensorNode,
    SeriesNode,
    SumNode,
    TFNode,
    TapNode,
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
        taps = ", ".join(serialize_dsl(tap) for tap in node.taps)
        return (
            "loop("
            f"reference={serialize_dsl(node.reference)}, "
            f"sum={serialize_dsl(node.sum_node)}, "
            f"controller={serialize_dsl(node.controller)}, "
            f"actuator={serialize_dsl(node.actuator)}, "
            f"plant={serialize_dsl(node.plant)}, "
            f"disturbance={serialize_dsl(node.disturbance) if node.disturbance else 'None'}, "
            f"sensor={serialize_dsl(node.sensor)}, "
            f"noise={serialize_dsl(node.noise) if node.noise else 'None'}, "
            f"taps=[{taps}])"
        )
    if isinstance(node, ReferenceNode):
        return f"reference(kind={node.kind!r}, name={node.name!r})"
    if isinstance(node, SumNode):
        return f"sum(signs={list(node.signs)!r}, name={node.name!r})"
    if isinstance(node, ControllerNode):
        return f"controller(kind={node.kind!r}, name={node.name!r})"
    if isinstance(node, ActuatorNode):
        return f"actuator(kind={node.kind!r}, name={node.name!r})"
    if isinstance(node, PlantNode):
        return f"plant(kind={node.kind!r}, name={node.name!r})"
    if isinstance(node, DisturbanceNode):
        return (
            f"disturbance(kind={node.kind!r}, injection={node.injection!r}, name={node.name!r})"
        )
    if isinstance(node, SensorNode):
        return f"sensor(kind={node.kind!r}, name={node.name!r})"
    if isinstance(node, NoiseNode):
        return f"noise(kind={node.kind!r}, name={node.name!r})"
    if isinstance(node, TapNode):
        return f"tap(signal={node.signal!r}, name={node.name!r})"
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


def ast_to_dict(node: ControlNode | ControlGraph | None) -> dict[str, Any] | None:
    if node is None:
        return None
    if isinstance(node, ControlGraph):
        return {
            "type": "loop",
            "reference": ast_to_dict(node.reference),
            "sum": ast_to_dict(node.sum_node),
            "controller": ast_to_dict(node.controller),
            "actuator": ast_to_dict(node.actuator),
            "plant": ast_to_dict(node.plant),
            "disturbance": ast_to_dict(node.disturbance),
            "sensor": ast_to_dict(node.sensor),
            "noise": ast_to_dict(node.noise),
            "taps": [ast_to_dict(tap) for tap in node.taps],
        }
    if isinstance(node, ReferenceNode):
        return {"type": "reference", "kind": node.kind, "name": node.name}
    if isinstance(node, SumNode):
        return {"type": "sum", "signs": list(node.signs), "name": node.name}
    if isinstance(node, ControllerNode):
        return {"type": "controller", "kind": node.kind, "name": node.name}
    if isinstance(node, ActuatorNode):
        return {"type": "actuator", "kind": node.kind, "name": node.name}
    if isinstance(node, PlantNode):
        return {"type": "plant", "kind": node.kind, "name": node.name}
    if isinstance(node, DisturbanceNode):
        return {
            "type": "disturbance",
            "kind": node.kind,
            "injection": node.injection,
            "name": node.name,
        }
    if isinstance(node, SensorNode):
        return {"type": "sensor", "kind": node.kind, "name": node.name}
    if isinstance(node, NoiseNode):
        return {"type": "noise", "kind": node.kind, "name": node.name}
    if isinstance(node, TapNode):
        return {"type": "tap", "signal": node.signal, "name": node.name}
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


def _parse_expr(expr: ast.AST) -> ControlNode | ControlGraph:
    if not isinstance(expr, ast.Call):
        raise DSLParseError("Top-level expression must be a function call")
    if not isinstance(expr.func, ast.Name):
        raise DSLParseError("Only named DSL functions are allowed")

    name = expr.func.id
    kwargs = {kw.arg: _literal(kw.value) for kw in expr.keywords if kw.arg is not None}

    if name == "loop":
        taps_literal = kwargs.get("taps", [])
        if taps_literal is None:
            taps_literal = []
        if not isinstance(taps_literal, list):
            raise DSLParseError("loop(..., taps=[...]) requires a list")
        return ControlGraph(
            reference=_parse_module_expr(kwargs.get("reference"), "reference"),
            sum_node=_parse_module_expr(kwargs.get("sum"), "sum"),
            controller=_parse_module_expr(kwargs.get("controller"), "controller"),
            actuator=_parse_module_expr(kwargs.get("actuator"), "actuator"),
            plant=_parse_module_expr(kwargs.get("plant"), "plant"),
            disturbance=_parse_optional_module_expr(kwargs.get("disturbance"), "disturbance"),
            sensor=_parse_module_expr(kwargs.get("sensor"), "sensor"),
            noise=_parse_optional_module_expr(kwargs.get("noise"), "noise"),
            taps=tuple(_parse_module_expr(item, "tap") for item in taps_literal),
        )
    if name == "reference":
        return ReferenceNode(kind=str(kwargs.get("kind", "step")), name=str(kwargs.get("name", "reference")))
    if name == "sum":
        signs = kwargs.get("signs", [1, -1])
        if not isinstance(signs, list):
            raise DSLParseError("sum(signs=[...]) requires a list of signs")
        return SumNode(signs=tuple(int(sign) for sign in signs), name=str(kwargs.get("name", "sum")))
    if name == "controller":
        return ControllerNode(kind=str(kwargs.get("kind", "pid")), name=str(kwargs.get("name", "controller")))
    if name == "actuator":
        return ActuatorNode(kind=str(kwargs.get("kind", "ideal")), name=str(kwargs.get("name", "actuator")))
    if name == "plant":
        return PlantNode(kind=str(kwargs.get("kind", "first_order")), name=str(kwargs.get("name", "plant")))
    if name == "disturbance":
        return DisturbanceNode(
            kind=str(kwargs.get("kind", "load_step")),
            injection=str(kwargs.get("injection", "output")),
            name=str(kwargs.get("name", "disturbance")),
        )
    if name == "sensor":
        return SensorNode(kind=str(kwargs.get("kind", "ideal")), name=str(kwargs.get("name", "sensor")))
    if name == "noise":
        return NoiseNode(kind=str(kwargs.get("kind", "none")), name=str(kwargs.get("name", "noise")))
    if name == "tap":
        return TapNode(signal=str(kwargs.get("signal", "y")), name=str(kwargs.get("name", "tap")))
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
        blocks = tuple(_parse_expr(arg) for arg in expr.args)
        if len(blocks) < 2:
            raise DSLParseError("series(...) requires at least two blocks")
        return SeriesNode(blocks=blocks)  # type: ignore[arg-type]
    if name == "parallel":
        blocks = tuple(_parse_expr(arg) for arg in expr.args)
        if len(blocks) < 2:
            raise DSLParseError("parallel(...) requires at least two blocks")
        return ParallelNode(blocks=blocks)  # type: ignore[arg-type]
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
    if isinstance(node, ControlGraph):
        raise DSLParseError("Graph DSL cannot be nested inside low-level control nodes")
    return node


def _parse_module_expr(value: Any, label: str) -> Any:
    if not isinstance(value, ast.AST):
        raise DSLParseError(f"{label} must be a nested DSL expression")
    return _parse_expr(value)


def _parse_optional_module_expr(value: Any, label: str) -> Any:
    if value is None:
        return None
    return _parse_module_expr(value, label)


def _literal(expr: ast.AST) -> Any:
    if isinstance(expr, ast.Call):
        return expr
    if isinstance(expr, ast.List | ast.Tuple):
        return [_literal(item) for item in expr.elts]
    if isinstance(expr, ast.Constant):
        return expr.value
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.USub):
        value = _literal(expr.operand)
        if isinstance(value, (int, float)):
            return -value
    raise DSLParseError(f"Unsupported DSL literal: {ast.dump(expr, include_attributes=False)}")


def _as_tuple(value: Any) -> tuple[float, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise DSLParseError("Expected coefficient list")
    return tuple(float(item) for item in value)


def _as_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    raise DSLParseError("Expected numeric parameter")


def _as_matrix(value: Any) -> Matrix | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise DSLParseError("Expected matrix literal")
    if not value:
        return tuple()
    if not all(isinstance(row, list) for row in value):
        raise DSLParseError("Expected a nested list for matrix literal")
    widths = {len(row) for row in value}
    if len(widths) > 1:
        raise DSLParseError("Matrix rows must have equal width")
    return tuple(tuple(float(item) for item in row) for row in value)


def _to_json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, tuple):
        return [_to_json_value(item) for item in value]
    return value


def _format_value(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, tuple):
        return "[" + ", ".join(_format_value(item) for item in value) + "]"
    if isinstance(value, list):
        return "[" + ", ".join(_format_value(item) for item in value) + "]"
    if isinstance(value, float):
        formatted = f"{value:.8f}".rstrip("0").rstrip(".")
        return formatted if formatted else "0"
    return repr(value) if isinstance(value, str) else str(value)
