"""Sections -> retrieval chunks.

- A chunk never crosses a page or section boundary, so every citation is an
  exact (section, page).
- Chunks are built from whole lines (tables are extracted one cell per line,
  so splitting on lines keeps numbers intact) up to MAX_TOKENS tokens, with
  ~OVERLAP_TOKENS tokens of overlap between consecutive chunks on a page.
- Sizes are counted with the embedding model's own tokenizer. bge-small
  silently truncates input past 512 tokens, and financial tables tokenize
  densely ("1,577" is 3 tokens), so counting words let 4.7% of chunks
  overflow. MAX_TOKENS leaves room for the context header.
"""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

from tieout.parsing.sections import Section

MAX_TOKENS = 400
OVERLAP_TOKENS = 50
MIN_WORDS = 8  # drop near-empty fragments (page numbers, lone headers)

TokenCounter = Callable[[str], int]


def count_words(text: str) -> int:
    """Fallback counter (tests); indexing uses the embedding tokenizer."""
    return len(text.split())


@dataclass(frozen=True)
class DocMeta:
    doc_name: str
    company: str
    ticker: str
    fiscal_year: int


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_name: str
    section: str  # item code, e.g. "7"
    section_title: str
    page: int  # 1-based PDF page
    ordinal: int  # position within the filing
    text: str
    header: str  # context header, prepended for embedding and keyword search

    @property
    def indexed_text(self) -> str:
        return f"{self.header}\n{self.text}"


def make_chunk_id(doc_name: str, page: int, text: str) -> str:
    """Stable across rebuilds; changes if the chunk's text changes."""
    digest = hashlib.sha1(f"{doc_name}|{page}|{text}".encode()).hexdigest()[:16]
    return f"ch_{digest}"


def make_header(meta: DocMeta, section: Section, page: int) -> str:
    item = f"Item {section.item}" if section.item[0].isdigit() else "Front matter"
    return (
        f"{meta.company} ({meta.ticker}) 10-K FY{meta.fiscal_year} | "
        f"{item}: {section.title} | page {page}"
    )


def _explode(line: str, limit: int, count: TokenCounter) -> list[str]:
    """Split a line longer than `limit` tokens into word runs that fit."""
    if count(line) <= limit:
        return [line]
    pieces: list[str] = []
    current: list[str] = []
    used = 0
    for word in line.split():
        n = count(word)  # per-word counts sum to (at least) the joined count
        if current and used + n > limit:
            pieces.append(" ".join(current))
            current, used = [], 0
        current.append(word)
        used += n
    if current:
        pieces.append(" ".join(current))
    return pieces


def split_lines(
    lines: list[str],
    max_tokens: int = MAX_TOKENS,
    overlap: int = OVERLAP_TOKENS,
    count: TokenCounter = count_words,
) -> list[list[str]]:
    """Group lines into windows of at most max_tokens, overlapping by ~overlap.

    A window's token count is the sum of its lines' counts (a close upper
    bound, since tokenizers don't merge across the newline).
    """
    lines = [piece for line in lines for piece in _explode(line, max_tokens, count)]
    counts = [count(line) for line in lines]
    windows: list[list[str]] = []
    start = 0
    while start < len(lines):
        end, used = start, 0
        while end < len(lines) and (end == start or used + counts[end] <= max_tokens):
            used += counts[end]
            end += 1
        windows.append(lines[start:end])
        if end >= len(lines):
            break
        # Step back from `end` until ~overlap tokens are repeated, but always advance.
        back, kept = end, 0
        while back > start + 1 and kept < overlap:
            back -= 1
            kept += counts[back]
        start = max(back, start + 1)
    return windows


def chunk_sections(
    meta: DocMeta, sections: list[Section], count: TokenCounter = count_words
) -> list[Chunk]:
    chunks: list[Chunk] = []
    seen: set[str] = set()
    for section in sections:
        for page, page_text in section.pages:
            lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
            for window in split_lines(lines, count=count):
                text = "\n".join(window)
                if len(text.split()) < MIN_WORDS:
                    continue
                chunk_id = make_chunk_id(meta.doc_name, page, text)
                if chunk_id in seen:  # identical text on the same page
                    continue
                seen.add(chunk_id)
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_name=meta.doc_name,
                        section=section.item,
                        section_title=section.title,
                        page=page,
                        ordinal=len(chunks),
                        text=text,
                        header=make_header(meta, section, page),
                    )
                )
    return chunks
