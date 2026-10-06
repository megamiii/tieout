"""Safe arithmetic for the agent: LLMs make arithmetic mistakes, so the agent
must compute every derived number with this tool.

The expression is parsed with Python's `ast` and only a whitelist of nodes is
evaluated (numbers, + - * / // % **, unary minus, parentheses, and abs/round/
min/max). No names, attributes, subscripts or calls to anything else, so
`__import__('os')` or `(1).__class__` cannot run. Arithmetic uses Decimal,
so 0.1 + 0.2 == 0.3 and large financial figures stay exact.
"""

import ast
import operator
from collections.abc import Callable
from decimal import Decimal, DivisionByZero, InvalidOperation, getcontext

MAX_EXPRESSION_CHARS = 500
MAX_EXPONENT = 100
MAX_ABS_RESULT = Decimal(10) ** 30

getcontext().prec = 28


class CalcError(ValueError):
    pass


_BINARY: dict[type[ast.operator], Callable[[Decimal, Decimal], Decimal]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}


def _round(x: Decimal, ndigits: Decimal = Decimal(0)) -> Decimal:
    return round(x, int(ndigits))


_FUNCTIONS: dict[str, Callable[..., Decimal]] = {
    "abs": lambda x: abs(x),
    "round": _round,
    "min": lambda *xs: min(xs),
    "max": lambda *xs: max(xs),
}


def _eval(node: ast.AST) -> Decimal:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        # str() first: Decimal(0.1) would carry binary float error.
        return Decimal(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
        value = _eval(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp):
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(right) > MAX_EXPONENT:
                raise CalcError(f"exponent {right} too large (max {MAX_EXPONENT})")
            return left**right
        op = _BINARY.get(type(node.op))
        if op is None:
            raise CalcError(f"operator {type(node.op).__name__} is not allowed")
        return op(left, right)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        fn = _FUNCTIONS.get(node.func.id)
        if fn is None:
            raise CalcError(f"function {node.func.id!r} is not allowed; use {sorted(_FUNCTIONS)}")
        return fn(*(_eval(a) for a in node.args))
    raise CalcError(f"{type(node).__name__} is not allowed: only numbers and arithmetic")


def evaluate(expression: str) -> Decimal:
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise CalcError(f"expression longer than {MAX_EXPRESSION_CHARS} characters")
    # Accept thousands separators the model copies from filings: 1,577 -> 1577.
    cleaned = _strip_thousands_commas(expression)
    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError as err:
        raise CalcError(f"not a valid arithmetic expression: {err.msg}") from err
    try:
        result = _eval(tree)
    except (DivisionByZero, ZeroDivisionError) as err:
        raise CalcError("division by zero") from err
    except (InvalidOperation, TypeError) as err:
        raise CalcError(f"invalid operation: {err}") from err
    if abs(result) > MAX_ABS_RESULT:
        raise CalcError("result is too large")
    return result


def _strip_thousands_commas(expression: str) -> str:
    """'1,577 - 1,373' -> '1577 - 1373', but keep commas between function args."""
    out = []
    for i, ch in enumerate(expression):
        if (
            ch == ","
            and 0 < i < len(expression) - 3
            and expression[i - 1].isdigit()
            and expression[i + 1 : i + 4].isdigit()
            and (i + 4 == len(expression) or not expression[i + 4].isdigit())
        ):
            continue
        out.append(ch)
    return "".join(out)


def format_decimal(value: Decimal) -> str:
    """Plain notation, trailing zeros removed: Decimal('14.8900') -> '14.89'."""
    text = f"{value.normalize():f}"
    return text if text != "-0" else "0"
