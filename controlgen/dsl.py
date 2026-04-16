from __future__ import annotations

import ast
from typing import Any

from controlgen.types import (
    ControlNode,
    DelayNode,
    FeedbackNode,
    GainNode,
    PIDNode,
    ParallelNode,
    SeriesNode,
    TFNode,
)


class DSLParseError(ValueError):
    """Raised when a DSL string cannot be parsed."""


def parse_dsl(text: str) -> ControlNode:
    """Parse function-style ControlDSL text into a control AST."""

    try:
        expr = ast.parse(text, mode="eval").body
    except SyntaxError as exc:
        raise DSLParseError(str(exc)) from exc
    return _parse_expr(expr)


def serialize_dsl(node: ControlNode) -> str:
    """Serialize a control AST into a stable DSL string."""

    if isinstance(node, TFNode):
        return (
            f"tf(num={_format_value(node.num)}, den={_format_value(node.den)}, "
            f"name={node.name!r})"
        )
    if isinstance(node, GainNode):
        return f"gain(k={_format_value(node.k)}, name={node.name!r})"
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
        return (
            f"delay(t={_format_value(node.t)}, order={node.order}, name={node.name!r})"
        )
    if isinstance(node, SeriesNode):
        items = ", ".join(serialize_dsl(block) for block in node.blocks)
        return f"series({items})"
    if isinstance(node, ParallelNode):
        items = ", ".join(serialize_dsl(block) for block in node.blocks)
        return f"parallel({items})"
    if isinstance(node, FeedbackNode):
        return (
            "feedback("
            f"forward={serialize_dsl(node.forward)}, "
            f"feedback={serialize_dsl(node.feedback)}, "
            f"sign={node.sign})"
        )
    raise TypeError(f"Unsupported node type: {type(node)!r}")


def ast_to_dict(node: ControlNode) -> dict[str, Any]:
    if isinstance(node, TFNode):
        return {"type": "tf", "num": _format_container(node.num), "den": _format_container(node.den), "name": node.name}
    if isinstance(node, GainNode):
        return {"type": "gain", "k": node.k, "name": node.name}
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
    raise TypeError(f"Unsupported node type: {type(node)!r}")


def _parse_expr(expr: ast.AST) -> ControlNode:
    if not isinstance(expr, ast.Call):
        raise DSLParseError("Top-level expression must be a function call")
    if not isinstance(expr.func, ast.Name):
        raise DSLParseError("Only named DSL functions are allowed")

    name = expr.func.id
    kwargs = {kw.arg: _literal(kw.value) for kw in expr.keywords if kw.arg is not None}

    if name == "tf":
        return TFNode(
            num=_as_tuple(kwargs.get("num")),
            den=_as_tuple(kwargs.get("den")),
            name=str(kwargs.get("name", "plant")),
        )
    if name == "gain":
        return GainNode(k=_as_number(kwargs.get("k")), name=str(kwargs.get("name", "gain")))
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
    if name == "series":
        blocks = tuple(_parse_expr(arg) for arg in expr.args)
        if len(blocks) < 2:
            raise DSLParseError("series(...) requires at least two blocks")
        return SeriesNode(blocks=blocks)
    if name == "parallel":
        blocks = tuple(_parse_expr(arg) for arg in expr.args)
        if len(blocks) < 2:
            raise DSLParseError("parallel(...) requires at least two blocks")
        return ParallelNode(blocks=blocks)
    if name == "feedback":
        if "forward" not in kwargs or "feedback" not in kwargs:
            raise DSLParseError("feedback(...) requires forward= and feedback=")
        return FeedbackNode(
            forward=_parse_expr(_ensure_expr(kwargs["forward"])),
            feedback=_parse_expr(_ensure_expr(kwargs["feedback"])),
            sign=int(kwargs.get("sign", -1)),
        )
    raise DSLParseError(f"Unknown DSL function: {name}")


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


def _ensure_expr(value: Any) -> ast.AST:
    if not isinstance(value, ast.AST):
        raise DSLParseError("Expected nested DSL expression")
    return value


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


def _format_container(values: tuple[float, ...] | None) -> list[float] | None:
    if values is None:
        return None
    return [float(v) for v in values]


def _format_value(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, tuple):
        return "[" + ", ".join(_format_value(v) for v in value) + "]"
    if isinstance(value, float):
        formatted = f"{value:.8f}".rstrip("0").rstrip(".")
        return formatted if formatted else "0"
    return repr(value) if isinstance(value, str) else str(value)
