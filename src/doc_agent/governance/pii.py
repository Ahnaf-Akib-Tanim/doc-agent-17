"""Governance — PII detection + redaction (mandatory)"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from ..contracts import *  # noqa
from ..logging_conf import get_logger

log = get_logger(__name__)

REVIEW_QUEUE = Path("data/interim/pii_review.jsonl")

# Direct identifiers. These are redacted automatically because nothing in a 1853-1945
# natural-history monograph legitimately needs them to answer a question.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b")),
    ("phone", re.compile(r"\b(?:\+\d{1,3}[ -])?\(?\d{3}\)?[ -]\d{3}[ -]\d{4}\b")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    # written as four fixed groups rather than {13,16} repetitions of an optional-separator digit:
    # the repetition form backtracks catastrophically on a page of measurement figures, which this
    # corpus has by the hundred.
    ("card", re.compile(r"\b\d{4}[ -]\d{4}[ -]\d{4}[ -]\d{4}\b")),
    ("postal_addr", re.compile(r"\b\d{1,5}\s+[A-Z][a-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd)\b")),
    ("url_user", re.compile(r"https?://[^\s]*@[^\s]+")),
]

# Names get FLAGGED, never auto-redacted: "Leidy", "Culbertson", "Gilmore" are authors,
# collectors and type-specimen donors, and they are frequently the answer to a taxonomic
# priority question. Auto-redaction here would delete the evidence, not protect anyone.
_NAME_HINT = re.compile(
    r"\b(?:Mr|Mrs|Miss|Dr|Prof)\.?\s+[A-Z][a-z]+"
    r"|\bpresented by\s+[A-Z][a-z]+"
    r"|\bex libris\b.{0,40}"
    r"|\bfrom the library of\b.{0,40}",
    re.IGNORECASE,
)


def detect(text: str) -> list[tuple[int, int, str]]:
    """Return (start, end, type) PII spans.

    Two tiers. Direct identifiers (email, phone, national id, card, street address) are
    returned as `redact`-able spans. Personal-name signals — donor inscriptions, bookplates,
    "presented by" — are returned tagged `person_review` and are NOT redacted by the pipeline;
    they are queued for a human, because in this corpus a person's name is usually a citation.
    """
    spans: list[tuple[int, int, str]] = []
    for kind, pat in _PATTERNS:
        for m in pat.finditer(text):
            spans.append((m.start(), m.end(), kind))
    for m in _NAME_HINT.finditer(text):
        spans.append((m.start(), m.end(), "person_review"))
    return sorted(spans)


def redact(text: str) -> str:
    """Replace direct identifiers with a typed placeholder; leave review-tier spans intact."""
    spans = [s for s in detect(text) if s[2] != "person_review"]
    if not spans:
        return text
    out = text
    for start, end, kind in sorted(spans, reverse=True):
        out = f"{out[:start]}[REDACTED:{kind}]{out[end:]}"
    return out


def _queue(items: list[dict[str, Any]]) -> None:
    if not items:
        return
    REVIEW_QUEUE.parent.mkdir(parents=True, exist_ok=True)
    with open(REVIEW_QUEUE, "a", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


class _Seams(Protocol):
    """The slice of the `hooks` module this file actually calls.

    `wiring.py` passes the `doc_agent.hooks` module itself (not an instance), so this is a
    structural Protocol rather than an import of the concrete module — it documents the seam
    names and the register() signature without adding a dependency edge back into hooks.py.
    """

    AFTER_OCR: str
    BEFORE_ANSWER: str
    ON_LOG: str

    def register(self, seam: str, handler: Callable[[dict], dict]) -> None: ...


def register(hooks: _Seams) -> None:
    """Wire PII redaction into the pipeline (AFTER_OCR, BEFORE_ANSWER, ON_LOG)."""

    def _scrub(ctx: dict) -> dict:
        pending: list[dict[str, Any]] = []
        n_redacted = 0

        chunks = ctx.get("chunks")
        if chunks:
            for c in chunks:
                spans = detect(c.text)
                flagged = [s for s in spans if s[2] == "person_review"]
                if flagged:
                    pending.append(
                        {
                            "chunk_id": getattr(c, "id", "?"),
                            "page_ids": list(getattr(c, "page_ids", [])),
                            "spans": [[s[0], s[1], s[2]] for s in flagged],
                            "excerpt": c.text[:200],
                        }
                    )
                if any(s[2] != "person_review" for s in spans):
                    c.text = redact(c.text)
                    n_redacted += 1

        answer = ctx.get("answer")
        if answer is not None and getattr(answer, "text", None):
            answer.text = redact(answer.text)

        if isinstance(ctx.get("msg"), str):
            ctx["msg"] = redact(ctx["msg"])

        _queue(pending)
        if n_redacted or pending:
            log.info(f"pii: redacted {n_redacted} chunks, queued {len(pending)} for human review")
        return ctx

    hooks.register(hooks.AFTER_OCR, _scrub)  # scrub extracted text before indexing
    hooks.register(hooks.BEFORE_ANSWER, _scrub)  # scrub the outgoing answer
    hooks.register(hooks.ON_LOG, _scrub)  # scrub logs
