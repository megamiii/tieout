from datetime import date

from tieout.parsing.pdf import cover_fiscal_year_end
from tieout.parsing.sections import FRONT, clean_page, find_headings, split_sections

TOC = "\n".join(
    [
        "TABLE OF CONTENTS",
        "ITEM 1",
        "Business",
        "ITEM 1A",
        "Risk Factors",
        "ITEM 7",
        "Management's Discussion and Analysis",
        "ITEM 7A",
        "Quantitative and Qualitative Disclosures",
        "ITEM 8",
        "Financial Statements",
    ]
)


def make_doc() -> list[str]:
    body = [
        "Item 1. Busines s.",  # split word from PDF extraction
        "We make things. See Item 7 for results.",
        "Item 7, Management's Discussion and Analysis, and the Notes are included.",
    ]
    return [
        "FORM 10-K\nFor the fiscal year ended December 31, 2022\nAcme Corp",
        TOC,
        "\n".join(body),
        "Item 1A. Risk Factors\nThings can go wrong.",
        "More risks.\nItem 7. Management's Discussion and Analysis of Financial Condition\n"
        "Revenue grew 10%.",
        "Margins improved.",
        "ITEM 8\nFINANCIAL STATEMENTS AND SUPPLEMENTARY DATA\nConsolidated balance sheet",
        *["Notes."] * 13,  # pad so the TOC sits in the first 15% of pages
    ]


def test_toc_page_is_skipped_and_cross_references_ignored() -> None:
    headings = find_headings([clean_page(p) for p in make_doc()])
    assert [(h.item, h.page) for h in headings] == [("1", 2), ("1A", 3), ("7", 4), ("8", 6)]


def test_sections_have_page_ranges_and_split_mid_page() -> None:
    sections = {s.item: s for s in split_sections(make_doc())}
    assert sections[FRONT].page_start == 1
    assert sections["1"].page_start == 3
    # Item 1A ends on page 5 where Item 7 starts mid-page.
    assert (sections["1A"].page_start, sections["1A"].page_end) == (4, 5)
    assert sections["1A"].pages[-1] == (5, "More risks.")
    assert sections["7"].page_start == 5 and sections["7"].page_end == 6
    assert "Revenue grew 10%." in sections["7"].text
    assert sections["8"].title == "Financial Statements and Supplementary Data"


def test_every_line_belongs_to_one_section() -> None:
    doc = make_doc()
    sections = split_sections(doc)
    total = sum(len(t.splitlines()) for s in sections for _, t in s.pages)
    assert total == sum(len([ln for ln in p.splitlines()]) for p in doc)


def test_out_of_order_heading_is_ignored() -> None:
    pages = ["x"] * 20
    pages[5] = "Item 7. Management's Discussion and Analysis"
    pages[6] = "Item 1A. Risk Factors"  # after Item 7: a stray match, not a heading
    assert [h.item for h in find_headings(pages)] == ["7"]


def test_clean_page_removes_nul_and_normalizes() -> None:
    assert clean_page("a\x00b\ufffds\u00a0\tc") == "a b's c"


def test_cover_fiscal_year_end() -> None:
    assert cover_fiscal_year_end(make_doc()) == date(2022, 12, 31)
    assert cover_fiscal_year_end(["fiscal year ended: May 31 , 2023"]) == date(2023, 5, 31)
    assert cover_fiscal_year_end(["no date here"]) is None
