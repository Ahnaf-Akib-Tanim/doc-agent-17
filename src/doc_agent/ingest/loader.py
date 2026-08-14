"""Stage 1 — load scanned page-images"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pymupdf
from PIL import Image

from ..contracts import *  # noqa
from ..logging_conf import get_logger

log = get_logger(__name__)

PAGES_MANIFEST = "pages.jsonl"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _sources(raw_dir: Path) -> list[tuple[str, Path]]:
    """(doc_id, pdf_path) for every volume under data/raw/.

    A volume is a sub-directory named after its doc_id (data/raw/leidy_1853/*.pdf); a loose
    PDF directly in data/raw/ is accepted too and takes its doc_id from the file stem.
    """
    out: list[tuple[str, Path]] = []
    for sub in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        for pdf in sorted(sub.glob("*.pdf")):
            out.append((sub.name, pdf))
    for pdf in sorted(raw_dir.glob("*.pdf")):
        out.append((pdf.stem, pdf))
    return out


def load_pages(cfg: dict) -> list[Page]:
    """Read data/raw/ -> list[Page].

    Rasterises every page of every source PDF once into `work_dir/pages/<doc_id>/pNNNN.png`
    (greyscale, cfg['ingest']['dpi']) and records one manifest row per page. Rasterisation is
    cached: a re-run skips pages whose PNG already exists, so the pipeline is re-runnable.

    The manifest carries doc_id, page_no, pixel size, the source SHA-256 and the DOCUMENT-level
    split. Split is assigned here and nowhere else — that is the leakage guard: a page cannot
    inherit a split its volume does not have.
    """
    ing: dict[str, Any] = cfg["ingest"]
    raw_dir = Path(ing["raw_dir"])
    work = Path(ing["work_dir"])
    dpi = int(ing["dpi"])
    splits: dict[str, str] = dict(ing.get("splits", {}))
    limit = ing.get("max_pages_per_doc")

    if not raw_dir.exists():
        raise FileNotFoundError(f"{raw_dir} is missing — run scripts/get_data.sh first")

    (work / "pages").mkdir(parents=True, exist_ok=True)
    pages: list[Page] = []
    rows: list[dict[str, Any]] = []

    for doc_id, pdf_path in _sources(raw_dir):
        digest = _sha256(pdf_path)
        out_dir = work / "pages" / doc_id
        out_dir.mkdir(parents=True, exist_ok=True)
        with pymupdf.open(pdf_path) as doc:
            n = len(doc) if limit is None else min(len(doc), int(limit))
            log.info(f"loader: {doc_id} {n} pages @ {dpi}dpi from {pdf_path.name}")
            for i in range(n):
                page_no = i + 1
                page_id = f"{doc_id}_p{page_no:04d}"
                img_path = out_dir / f"p{page_no:04d}.png"
                if not img_path.exists():
                    pix = doc[i].get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
                    pix.save(str(img_path))
                    w, h = pix.width, pix.height
                else:
                    with Image.open(img_path) as im:
                        w, h = im.size
                pages.append(Page(id=page_id, image_path=str(img_path), doc_id=doc_id))
                rows.append(
                    {
                        "page_id": page_id,
                        "doc_id": doc_id,
                        "page_no": page_no,
                        "raw_path": str(img_path),
                        "width_px": w,
                        "height_px": h,
                        "source_pdf": str(pdf_path),
                        "source_sha256": digest,
                        "dpi": dpi,
                        "split": splits.get(doc_id, "train"),
                    }
                )

    manifest = work / PAGES_MANIFEST
    with open(manifest, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    n_docs = len({p.doc_id for p in pages})
    log.info(f"loader: {len(pages)} pages from {n_docs} volumes -> {manifest}")
    return pages
