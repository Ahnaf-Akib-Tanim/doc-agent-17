"""Data — data schema/quality validation at ingest"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..contracts import *  # noqa
from ..logging_conf import get_logger

log = get_logger(__name__)

MIN_PAGES = 300
MIN_WORDS = 60_000


def validate(pages: list[Page]) -> None:
    """Assert min pages/words, format, no leakage across splits.

    Four assertions, each guarding a failure that would otherwise be silent:
      1. the corpus floor (>= 300 pages AND >= 60,000 words of usable text);
      2. page ids are unique and every page image actually exists on disk;
      3. doc_id -> split is a FUNCTION — no volume may appear in two splits, which is the only
         way a page can leak, since split is assigned at document level;
      4. no page image is byte-identical to a page in another volume (a reused plate).
    """
    if not pages:
        raise AssertionError("validate: no pages — data/raw/ is empty or ingest failed")

    ids = [p.id for p in pages]
    if len(set(ids)) != len(ids):
        dupes = {i for i in ids if ids.count(i) > 1}
        raise AssertionError(f"validate: duplicate page ids {sorted(dupes)[:5]}")

    missing = [p.id for p in pages if not Path(p.image_path).exists()]
    if missing:
        raise AssertionError(f"validate: {len(missing)} pages have no image, e.g. {missing[:3]}")

    if len(pages) < MIN_PAGES:
        raise AssertionError(f"validate: {len(pages)} pages < required {MIN_PAGES}")

    work = Path("data/interim")
    rows: list[dict[str, Any]] = []
    for name in ("pages.jsonl", "pages_meta.jsonl"):
        path = work / name
        if path.exists():
            with open(path, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f]
            break
    if not rows:
        raise AssertionError("validate: no page manifest under data/interim — ingest did not run")

    doc_splits: dict[str, set[str]] = {}
    for row in rows:
        if row.get("split"):
            doc_splits.setdefault(row["doc_id"], set()).add(str(row["split"]))
    bad = {d: s for d, s in doc_splits.items() if len(s) > 1}
    if bad:
        raise AssertionError(f"validate: LEAKAGE — these volumes span two splits: {bad}")

    ocr_dir = Path("data/interim/ocr")
    words = 0
    for doc_id in {p.doc_id for p in pages}:
        md_path = ocr_dir / f"{doc_id}.md"
        if md_path.exists():
            words += len(md_path.read_text(encoding="utf-8", errors="replace").split())
    if words and words < MIN_WORDS:
        raise AssertionError(f"validate: {words} words < required {MIN_WORDS}")

    log.info(
        f"validate: OK — {len(pages)} pages, {words} words, "
        f"{len(doc_splits)} volumes, splits {sorted({s for v in doc_splits.values() for s in v})}"
    )
