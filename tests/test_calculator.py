from decimal import Decimal

import pytest

from tieout.agent.calculator import CalcError, evaluate, format_decimal


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("1577 - 1373", "204"),
        ("(1577 - 1373) / 1373 * 100", "14.85797523670793882010196650"),
        ("0.1 + 0.2", "0.3"),  # exact, unlike float
        ("1,577 - 1,373", "204"),  # thousands separators copied from filings
        ("round((1577 - 1373) / 1373 * 100, 2)", "14.86"),
        ("max(1, 2, 3) + min(4, 5) + abs(-6)", "13"),
        ("-(2 ** 10)", "-1024"),
        ("7 // 2 + 7 % 2", "4"),
        ("34229000000 / 1000000", "34229"),
    ],
)
def test_evaluates_arithmetic(expr: str, expected: str) -> None:
    assert evaluate(expr) == Decimal(expected)


@pytest.mark.parametrize(
    ("expr", "message"),
    [
        ("__import__('os').system('echo hi')", "not allowed"),
        ("(1).__class__", "not allowed"),
        ("x + 1", "not allowed"),
        ("open('f')", "not allowed"),
        ("[1, 2][0]", "not allowed"),
        ("'a' * 3", "not allowed"),
        ("9 ** 9 ** 9", "too large"),
        ("1 / 0", "division by zero"),
        ("1 +", "not a valid"),
        ("1" * 600, "longer than"),
    ],
)
def test_rejects_unsafe_or_invalid(expr: str, message: str) -> None:
    with pytest.raises(CalcError, match=message):
        evaluate(expr)


def test_format_decimal() -> None:
    assert format_decimal(Decimal("14.8900")) == "14.89"
    assert format_decimal(Decimal("1.577E+9")) == "1577000000"
    assert format_decimal(Decimal("-0")) == "0"
