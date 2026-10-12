"""Mutation test for the tie-out: plant known errors and check they are caught.

Audited 10-Ks rarely contain real MD&A/XBRL discrepancies, so "0 mismatches
found" says nothing about whether the detector works. Instead: take claims
that tie out (matched), change the amount in the sentence by 5-10% (beyond
its rounding, the size of a stale or mistyped figure), and run the tie-out on
the altered claims. A planted error is detected if it comes back mismatched
against the same XBRL fact it matched before.
"""

import random
from dataclasses import replace
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from sqlalchemy.orm import Session

from tieout.evals.grading import ChatModel
from tieout.retrieval.embeddings import Embedder
from tieout.workflows.claims import SCALES, Claim
from tieout.workflows.service import (
    concept_vectors,
    filing_facts,
    load_filing,
    mdna_claims,
    run_tieout,
)
from tieout.workflows.tieout import classify, concept_ranking

FACTORS = (Decimal("0.90"), Decimal("1.10"), Decimal("0.95"), Decimal("1.05"))


def mutate(claim: Claim, factor: Decimal) -> Claim:
    """The same claim with its amount scaled by `factor`, written with the same
    precision and style ("$1.347 billion" -> "$1.482 billion")."""
    scale_word = claim.amount_text.split()[-1].lower()
    mult = SCALES[scale_word]
    number = claim.amount_text.split()[0].lstrip("$")
    decimals = len(number.split(".")[1]) if "." in number else 0
    step = Decimal(1).scaleb(-decimals)
    new_number = (claim.value * factor / mult).quantize(step, rounding=ROUND_HALF_UP)
    text = f"{new_number:,.{decimals}f}" if "," in number else f"{new_number:.{decimals}f}"
    new_amount = f"${text} {claim.amount_text.split()[-1]}"
    sentence = (
        claim.sentence[: claim.start]
        + new_amount
        + claim.sentence[claim.start + len(claim.amount_text) :]
    )
    return replace(claim, amount_text=new_amount, value=new_number * mult, sentence=sentence)


def run_mutation_eval(
    session: Session,
    embedder: Embedder,
    llm: ChatModel,
    filings: list[tuple[str, int]],
    per_filing: int = 8,
    seed: int = 11,
) -> dict[str, Any]:
    rng = random.Random(seed)
    cases: list[dict[str, Any]] = []
    for ticker, year in filings:
        _, filing = load_filing(session, ticker, year)
        claims = mdna_claims(session, filing)
        facts = filing_facts(session, filing.accession)
        concepts = sorted({f.concept for f in facts})
        cvecs = concept_vectors(session, set(concepts))
        vectors = embedder.embed_documents([c.context for c in claims])
        matched = []
        for c, v in zip(claims, vectors, strict=True):
            r = classify(c, facts, concept_ranking(v, concepts, cvecs))
            if r.status == "matched" and r.fact is not None:
                matched.append((c, r.fact.fact_id))
        rng.shuffle(matched)
        chosen: list[tuple[Claim, str]] = []
        seen_facts: set[str] = set()
        for c, fact_id in matched:  # one planted error per fact, for variety
            if fact_id not in seen_facts and len(chosen) < per_filing:
                chosen.append((c, fact_id))
                seen_facts.add(fact_id)
        planted = [mutate(c, FACTORS[i % len(FACTORS)]) for i, (c, _) in enumerate(chosen)]
        report = run_tieout(session, embedder, llm, ticker, year, claims=planted)
        for (orig, fact_id), item in zip(chosen, report["items"], strict=True):
            fact = item.get("fact") or {}
            cases.append(
                {
                    "ticker": ticker,
                    "fiscal_year": year,
                    "original": orig.amount_text,
                    "planted": item["amount"],
                    "sentence": item["sentence"],
                    "expected_fact": fact_id,
                    "status": item["status"],
                    "flagged_fact": fact.get("fact_id"),
                    "detected": item["status"] == "mismatched" and fact.get("fact_id") == fact_id,
                    "reason": item["adjudication"] or item["note"],
                }
            )
    n = len(cases)
    detected = sum(c["detected"] for c in cases)
    statuses: dict[str, int] = {}
    for case in cases:
        statuses[case["status"]] = statuses.get(case["status"], 0) + 1
    return {
        "planted": n,
        "detected": detected,
        "detection_rate": detected / n if n else None,
        "flagged_wrong_fact": sum(c["status"] == "mismatched" and not c["detected"] for c in cases),
        "missed": n - detected,
        "outcomes": statuses,
        "cases": cases,
    }
