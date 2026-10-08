from tieout.highlight import highlight, highlight_all

TABLE = (
    "Cash Flows from Investing Activities\n"
    "Purchases of property, plant and equipment (PP&E)\n"
    "(1,577)\n"
    "(1,373)\n"
    "(1,420)"
)


def marked(text: str, quote: str) -> tuple[list[str], str]:
    segments, kind = highlight(text, quote)
    assert "".join(s["text"] for s in segments) == text  # nothing lost or added
    return [s["text"] for s in segments if s["mark"]], kind


def test_quote_across_line_breaks_and_without_commas() -> None:
    spans, kind = marked(TABLE, "Purchases of property, plant and equipment (PP&E) (1577)")
    assert kind == "quote"
    assert spans == ["Purchases of property, plant and equipment (PP&E)\n(1,577)"]


def test_case_insensitive() -> None:
    assert marked(TABLE, "cash flows from INVESTING activities")[1] == "quote"


def test_falls_back_to_figures() -> None:
    spans, kind = marked(TABLE, "capital spending of 1,577 and 1,420")
    assert kind == "figures" and spans == ["1,577", "1,420"]


def test_figures_do_not_match_inside_longer_numbers() -> None:
    spans, _ = marked("Revenue 15,770 and 1,577", "1577")
    assert spans == ["1,577"]


def test_no_match() -> None:
    spans, kind = marked(TABLE, "goodwill impairment")
    assert spans == [] and kind == "none"


def test_several_quotes_of_one_source_are_merged() -> None:
    segments, kinds = highlight_all(TABLE, ["Cash Flows from Investing", "(1,373)", "goodwill"])
    assert [s["text"] for s in segments if s["mark"]] == ["Cash Flows from Investing", "(1,373)"]
    assert kinds == ["quote", "quote", "none"]
