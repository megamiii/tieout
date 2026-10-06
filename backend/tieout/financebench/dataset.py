"""FinanceBench open-source sample (Patronus AI, CC BY-NC 4.0).

https://github.com/patronus-ai/financebench
150 questions over public-company filings, each with a gold answer and the
evidence passages (with PDF page numbers) that support it.
"""

import json
from typing import Any

from pydantic import BaseModel, Field

from tieout.net.cached_client import CachedClient

REPO_RAW = "https://raw.githubusercontent.com/patronus-ai/financebench/main"
QUESTIONS_URL = f"{REPO_RAW}/data/financebench_open_source.jsonl"
DOCUMENTS_URL = f"{REPO_RAW}/data/financebench_document_information.jsonl"
PDF_URL = REPO_RAW + "/pdfs/{doc_name}.pdf"


class Evidence(BaseModel):
    evidence_text: str
    doc_name: str
    evidence_page_num: int  # 0-based page index in the FinanceBench PDF


class Document(BaseModel):
    doc_name: str  # e.g. "3M_2018_10K"
    company: str
    gics_sector: str
    doc_type: str  # "10k", "10q", "8k", "Earnings"
    doc_period: int
    doc_link: str

    @property
    def pdf_url(self) -> str:
        return PDF_URL.format(doc_name=self.doc_name)


class Question(BaseModel):
    financebench_id: str
    company: str
    doc_name: str
    question_type: str
    question_reasoning: str | None = None
    question: str
    answer: str
    justification: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)


def _jsonl(raw: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]


def load_documents(http: CachedClient) -> dict[str, Document]:
    rows = _jsonl(http.get_bytes(DOCUMENTS_URL))
    return {row["doc_name"]: Document.model_validate(row) for row in rows}


def load_questions(http: CachedClient) -> list[Question]:
    return [Question.model_validate(row) for row in _jsonl(http.get_bytes(QUESTIONS_URL))]
