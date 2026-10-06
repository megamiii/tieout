from itertools import pairwise

from tieout.parsing.sections import Section
from tieout.retrieval.chunking import DocMeta, chunk_sections, make_chunk_id, split_lines
from tieout.retrieval.search import bm25_idf, rrf, select_terms

META = DocMeta("ACME_2022_10K", "Acme", "ACME", 2022)


def words(n: int, tag: str) -> str:
    return " ".join(f"{tag}{i}" for i in range(n))


def test_split_lines_respects_target_and_overlap() -> None:
    lines = [words(10, f"l{i}_") for i in range(100)]  # 1000 words, 10 per line
    windows = split_lines(lines, max_tokens=300, overlap=50)
    assert all(sum(len(x.split()) for x in w) <= 300 for w in windows)
    # consecutive windows share ~50 words (5 lines)
    for a, b in pairwise(windows):
        assert a[-5:] == b[:5]
    # every line is covered
    assert {ln for w in windows for ln in w} == set(lines)


def test_split_lines_long_single_line_still_advances() -> None:
    windows = split_lines([words(1000, "x"), "tail line here"], max_tokens=300, overlap=50)
    assert windows[-1][-1] == "tail line here"


def test_chunks_stay_within_page_and_carry_metadata() -> None:
    section = Section(
        item="7",
        title="Management's Discussion and Analysis",
        page_start=10,
        page_end=11,
        pages=((10, "\n".join(words(10, "a") for _ in range(40))), (11, words(20, "b"))),
    )
    chunks = chunk_sections(META, [section])
    assert {c.page for c in chunks} == {10, 11}
    assert all(c.section == "7" for c in chunks)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert chunks[0].header.startswith("Acme (ACME) 10-K FY2022 | Item 7")
    assert chunks[0].indexed_text.startswith(chunks[0].header + "\n")


def test_tiny_fragments_are_dropped() -> None:
    section = Section("7", "MD&A", 3, 3, ((3, "42"),))
    assert chunk_sections(META, [section]) == []


def test_chunk_id_stable_and_content_sensitive() -> None:
    assert make_chunk_id("D", 1, "text") == make_chunk_id("D", 1, "text")
    assert make_chunk_id("D", 1, "text") != make_chunk_id("D", 1, "text!")
    assert make_chunk_id("D", 1, "text").startswith("ch_")


def test_rrf_rewards_agreement() -> None:
    vector = ["a", "b", "c"]
    keyword = ["c", "d", "a"]
    fused = [doc for doc, _ in rrf([vector, keyword])]
    # a: 1/61 + 1/63, c: 1/63 + 1/61 -> tie broken by id; both beat single-list docs
    assert fused[:2] == ["a", "c"]
    assert set(fused[2:]) == {"b", "d"}


def test_rrf_scores() -> None:
    scores = dict(rrf([["x", "y"]], k=60))
    assert scores["x"] == 1 / 61 and scores["y"] == 1 / 62
    assert rrf([]) == []


def test_long_line_is_split_to_fit() -> None:
    windows = split_lines([words(1000, "x")], max_tokens=300, overlap=50)
    assert all(sum(len(x.split()) for x in w) <= 300 for w in windows)
    assert " ".join(" ".join(w) for w in windows).split()[-1] == "x999"


def test_select_terms_drops_common_and_unknown_terms() -> None:
    ndoc = {"statement": 5741, "million": 3440, "expenditur": 383, "capit": 1506, "usd": 20}
    terms = select_terms(
        ["statement", "million", "expenditur", "capit", "usd", "zebra"], ndoc, total=13003
    )
    words = [w for w, _ in terms]
    assert "statement" not in words and "million" not in words  # > 20% of chunks
    assert "zebra" not in words  # not in the corpus
    assert words == ["usd", "expenditur", "capit"]  # rarest first


def test_bm25_idf_decreases_with_frequency() -> None:
    assert bm25_idf(10, 1000) > bm25_idf(100, 1000) > bm25_idf(500, 1000) > 0
