"""Stage 4 — chunk text"""

from __future__ import annotations

import re
from typing import Any

from ..contracts import *  # noqa
from ..logging_conf import get_logger

log = get_logger(__name__)

_TAG = re.compile(r"^<(text|table|figure|heading)>\n?", re.IGNORECASE)
_HTML_ROW = re.compile(r"<tr\b.*?</tr>", re.IGNORECASE | re.DOTALL)
_HTML_CELL = re.compile(r"<t[hd]\b[^>]*>(.*?)</t[hd]>", re.IGNORECASE | re.DOTALL)
_TAGS = re.compile(r"<[^>]+>")
_SEP_ROW = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")


def _untag(text: str) -> tuple[str, str]:
    m = _TAG.match(text)
    if not m:
        return "text", text
    return m.group(1).lower(), text[m.end() :]


def _tokens(text: str) -> int:
    return len(text.split())


def _md_rows(body: str) -> list[str]:
    return [ln.strip() for ln in body.splitlines() if ln.strip().startswith("|")]


def _html_rows(body: str) -> list[str]:
    rows: list[str] = []
    for tr in _HTML_ROW.findall(body):
        cells = [_TAGS.sub("", c).strip() for c in _HTML_CELL.findall(tr)]
        cells = [c for c in cells if c]
        if cells:
            rows.append(" | ".join(cells))
    return rows


def _table_units(body: str, heading: str, max_tokens: int) -> list[str]:
    """One retrieval unit per TABLE ROW, each carrying its header line.

    This is the chunking our data speciality forces. A measurement row read alone
    ("... 133 137 133 128") is not an answer to anything: the species is in the column header
    and the quantity is in the row label. Emitting the header with every row is how a retrieved
    row stays interpretable without a second lookup, and it is why a comparative table stops
    being one unusable 300-token blob.
    """
    rows = _md_rows(body) or _html_rows(body)
    if len(rows) < 2:
        # a "table" the reader emitted with no recoverable row structure — treat it as prose
        # rather than letting one 400-token blob into the index unsized
        return _pack([body.strip()], heading, max_tokens, 0) if body.strip() else []
    header = ""
    data: list[str] = []
    for r in rows:
        if _SEP_ROW.match(r) and header:
            continue
        if not header:
            header = r.strip("| ").strip()
            continue
        data.append(r.strip("| ").strip())
    units: list[str] = []
    caption = f"{heading} — " if heading else ""
    units.append(f"{caption}TABLE HEADER: {header}")
    for r in data:
        if not r or _SEP_ROW.match(r):
            continue
        unit = f"{caption}TABLE ROW under [{header}]: {r}"
        units.append(unit if _tokens(unit) <= max_tokens else " ".join(unit.split()[:max_tokens]))
    return units


def _pack(paragraphs: list[str], heading: str, size: int, overlap: int) -> list[str]:
    """Pack prose into ~size-token windows on PARAGRAPH boundaries, with a token overlap.

    Windows never cross a heading, and the heading is prefixed onto every window it owns, so a
    chunk retrieved out of context still says which section of which chapter it came from.
    """
    out: list[str] = []
    buf: list[str] = []
    n = 0
    prefix = f"{heading}\n" if heading else ""
    for para in paragraphs:
        t = _tokens(para)
        if t > size:
            words = para.split()
            step = max(size - overlap, 1)
            for i in range(0, len(words), step):
                out.append(prefix + " ".join(words[i : i + size]))
            continue
        if n + t > size and buf:
            out.append(prefix + "\n".join(buf))
            # carry exactly `overlap` tokens forward, never a whole trailing paragraph: a
            # paragraph-sized tail can already exceed the window and push the next chunk over it
            tail_words = " ".join(buf).split()[-overlap:] if overlap else []
            buf = [" ".join(tail_words)] if tail_words else []
            n = len(tail_words)
        buf.append(para)
        n += t
    if buf:
        out.append(prefix + "\n".join(buf))
    return out


def split(chunks: list[Chunk], cfg: dict) -> list[Chunk]:
    """Re-chunk to cfg['index'] size/overlap.

    Structure-aware rather than fixed-window (the bonus declared in A1). Three rules:

      * a TABLE becomes one chunk per row, each carrying the header row;
      * PROSE is packed to chunk_tokens on paragraph boundaries with overlap, under the
        heading that owns it;
      * a FIGURE / plate caption keeps its own chunk and inherits the nearest heading, so a
        plate legend stays retrievable on its own terms.

    A fixed 256-token window over the same text cuts across table rows, which is exactly the
    failure our speciality is about. Set index.chunking: "fixed" to get the flat baseline —
    the two are compared in notebooks/kb_demo.ipynb.
    """
    idx: dict[str, Any] = cfg["index"]
    size = int(idx["chunk_tokens"])
    overlap = int(idx["overlap"])
    mode = str(idx.get("chunking", "structure"))
    min_tokens = int(idx.get("min_chunk_tokens", 4))

    by_page: dict[str, list[Chunk]] = {}
    order: list[str] = []
    for c in chunks:
        key = c.page_ids[0] if c.page_ids else c.doc_id
        if key not in by_page:
            order.append(key)
        by_page.setdefault(key, []).append(c)

    out: list[Chunk] = []
    for page_id in order:
        blocks = by_page[page_id]
        doc_id = blocks[0].doc_id
        heading = ""
        pending: list[str] = []
        plate: list[str] = []
        units: list[str] = []

        for block in blocks:
            kind, body = _untag(block.text)
            body = body.strip()
            if not body:
                continue
            if mode != "structure":
                pending.append(body)
                continue
            if kind in ("heading", "table") and pending:
                units.extend(_pack(pending, heading, size, overlap))
                pending = []
            if kind == "heading":
                heading = body.splitlines()[0][:200]
                units.append(heading)
            elif kind == "table":
                units.extend(_table_units(body, heading, size))
            elif kind == "figure":
                # every label, callout and caption line on a plate belongs to the SAME plate;
                # emitting them one per line would put "egg" and "28" in the index as chunks
                plate.append(body)
            else:
                pending.extend(p for p in re.split(r"\n\s*\n", body) if p.strip())
        if plate:
            units.extend(_pack(["FIGURE / PLATE: " + " · ".join(plate)], heading, size, overlap))
        if pending:
            units.extend(_pack(pending, heading if mode == "structure" else "", size, overlap))

        for i, text in enumerate(units):
            text = text.strip()
            if _tokens(text) < min_tokens:
                continue
            out.append(
                Chunk(id=f"{page_id}_c{i:03d}", doc_id=doc_id, text=text, page_ids=[page_id])
            )

    sizes = [_tokens(c.text) for c in out] or [0]
    log.info(
        f"chunk: {len(out)} chunks (mode={mode}) "
        f"median={sorted(sizes)[len(sizes) // 2]} max={max(sizes)} tokens"
    )
    return out
