"""Data — corpus versioning (which corpus version -> which result)"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..contracts import *  # noqa
from ..logging_conf import get_logger

log = get_logger(__name__)

LOCK_FILE = "corpus.lock.json"


def snapshot(corpus_dir: str) -> str:
    """Hash + record a corpus version id.

    Hashes each source PDF, then hashes the sorted (name, sha256) list to get one id for the
    whole corpus. Content-addressed rather than a counter, so the id changes if and only if the
    bytes change — which is what lets an index, a metric and a form answer all be tied to the
    same corpus without anyone remembering to bump a number.

    Writes `corpus.lock.json` beside the corpus and returns e.g. "corpus_a71e3c9d".
    """
    root = Path(corpus_dir)
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist — run scripts/get_data.sh first")

    files: dict[str, str] = {}
    for pdf in sorted(root.rglob("*.pdf")):
        h = hashlib.sha256()
        with open(pdf, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        files[str(pdf.relative_to(root)).replace("\\", "/")] = h.hexdigest()

    if not files:
        raise FileNotFoundError(f"no PDFs under {root} — nothing to version")

    payload = json.dumps(files, sort_keys=True).encode("utf-8")
    version = "corpus_" + hashlib.sha256(payload).hexdigest()[:8]
    (root / LOCK_FILE).write_text(
        json.dumps({"version": version, "files": files}, indent=2), encoding="utf-8"
    )
    log.info(f"versioning: {version} over {len(files)} source documents")
    return version
