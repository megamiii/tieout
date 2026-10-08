"""Find a citation's quote inside its source passage, for highlighting in the UI.

PDF text breaks lines inside table rows and sentences, and models quote numbers
with or without thousands separators, so the match is flexible: whitespace in
the quote matches any whitespace, and digits may have commas between them
("1577" matches "1,577"). If the quote as a whole isn't found, the figures it
contains are highlighted instead.
"""

import re
from typing import Literal, TypedDict

from tieout.agent.calculator import format_decimal
from tieout.agent.validation import numbers_in

MatchKind = Literal["quote", "figures", "none"]


class Segment(TypedDict):
    text: str
    mark: bool


def _token_pattern(token: str) -> str:
    """Regex for one quote token: commas optional between digits, and a number
    never matches inside a longer number ("1577" must not hit "15,770")."""
    token = re.sub(r"(?<=\d),(?=\d)", "", token)
    parts = []
    for i, ch in enumerate(token):
        parts.append(re.escape(ch))
        if ch.isdigit() and i + 1 < len(token) and token[i + 1].isdigit():
            parts.append(",?")
    pattern = "".join(parts)
    if token[:1].isdigit():
        pattern = r"(?<![\d,.])" + pattern
    if token[-1:].isdigit():
        pattern += r"(?![\d]|,\d)"
    return pattern


def quote_pattern(quote: str) -> re.Pattern[str] | None:
    tokens = quote.split()
    if not tokens:
        return None
    return re.compile(r"\s+".join(_token_pattern(t) for t in tokens), re.IGNORECASE)


def _segments(text: str, spans: list[tuple[int, int]]) -> list[Segment]:
    out: list[Segment] = []
    pos = 0
    for start, end in sorted(spans):
        if start < pos:  # overlapping span: already covered
            continue
        if start > pos:
            out.append({"text": text[pos:start], "mark": False})
        out.append({"text": text[start:end], "mark": True})
        pos = end
    if pos < len(text):
        out.append({"text": text[pos:], "mark": False})
    return out


def highlight(text: str, quote: str) -> tuple[list[Segment], MatchKind]:
    pattern = quote_pattern(quote)
    if pattern is not None:
        m = pattern.search(text)
        if m:
            return _segments(text, [m.span()]), "quote"
    spans = []
    for value in numbers_in(quote):
        digits = format_decimal(abs(value))
        if len(digits.replace(".", "")) < 2:
            continue  # single digits would light up the whole passage
        number = re.compile(_token_pattern(digits))
        spans += [m.span() for m in number.finditer(text)]
    if spans:
        return _segments(text, spans), "figures"
    return [{"text": text, "mark": False}], "none"
