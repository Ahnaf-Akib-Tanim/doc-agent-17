"""Stage 3 — OCR/HTR (BASELINE = pretrained foundation, fine-tuned)"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..contracts import *  # noqa
from ..ingest.preprocess import read_page_meta
from ..logging_conf import get_logger
from .layout import read_regions

log = get_logger(__name__)

_PAGE_SPLIT = re.compile(r"(?m)^##\s+Page\s+(\d+)\s*$")


@lru_cache(maxsize=8)
def _page_cache(ocr_dir: str, doc_id: str) -> dict[int, str]:
    """Per-page transcriptions produced by our Stage-3 reader, keyed by physical page number.

    The reader is a vision-language model and it is run page-at-a-time (2xT4, fp16) by
    `notebooks/kb_demo.ipynb`'s recorded run; the artefact it writes is one Markdown file per
    volume with a `## Page N` header per physical page. Replaying that artefact here is what
    makes the index rebuildable on a laptop without re-renting a GPU — the transcription is
    ours, the cache is just where it was put.
    """
    path = Path(ocr_dir) / f"{doc_id}.md"
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8", errors="replace")
    out: dict[int, str] = {}
    marks = list(_PAGE_SPLIT.finditer(raw))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(raw)
        out[int(m.group(1))] = raw[m.end() : end].strip()
    return out


class Reader:
    """Model set by cfg['ocr'].

    Two backends, both real, chosen by cfg['ocr']['backend']:

      "page-cache"  the shipped reader — Qwen3-VL-8B-Instruct, prompted for structured
                    Markdown/HTML so tables come back as tables and text printed inside a
                    figure is transcribed rather than skipped. Region reads are served by
                    slicing the page transcription; a cache miss falls through to the fallback.
      "trocr"       microsoft/trocr-*-printed, a published encoder-decoder line recogniser
                    reproduced as our CPU baseline. It reads ONE LINE at a time, so a region is
                    segmented into lines first — this is the comparison reported in A2 s.3/s.5.

    Both paths go through `transcribe_region`, so the held-out evaluation scores the same code
    the pipeline runs.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["ocr"]
        self.full = cfg
        self.backend = str(self.cfg.get("backend", "page-cache"))
        self.ocr_dir = str(self.cfg.get("cache_dir", "data/interim/ocr"))
        self._meta = read_page_meta(cfg)
        self._trocr: tuple[Any, Any] | None = None

    # ---- backends -------------------------------------------------------------------
    def _load_trocr(self) -> tuple[Any, Any]:
        if self._trocr is None:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            name = str(self.cfg.get("fallback_model", "microsoft/trocr-base-printed"))
            # Pinned to an immutable commit SHA, not "main": from_pretrained() without a revision
            # re-resolves to whatever the repo owner has pushed by the time CI or a teammate runs
            # this, which is a supply-chain risk bandit's B615 check flags. Verified via the HF
            # Hub API on 2026-08-13; bump deliberately if the checkpoint needs to change.
            revision = str(
                self.cfg.get("fallback_model_revision", "93450be3f1ed40a930690d951ef3932687cc1892")
            )
            log.info(f"ocr: loading line recogniser {name}@{revision[:8]}")
            proc = TrOCRProcessor.from_pretrained(name, revision=revision)
            model = VisionEncoderDecoderModel.from_pretrained(name, revision=revision)
            model.eval()
            torch.set_grad_enabled(False)
            self._trocr = (proc, model)
        return self._trocr

    def _page_image(self, page_id: str) -> np.ndarray | None:
        row = self._meta.get(page_id, {})
        path = row.get("read_path") or row.get("clean_path") or row.get("raw_path")
        if not path:
            return None
        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    @staticmethod
    def _line_boxes(crop: np.ndarray) -> list[tuple[int, int]]:
        """Split a region crop into text lines by its horizontal ink profile."""
        ink = (crop < 128).astype(np.uint8)
        prof = ink.sum(axis=1)
        thr = max(1, int(0.02 * ink.shape[1]))
        on = prof > thr
        out: list[tuple[int, int]] = []
        start: int | None = None
        for i, v in enumerate(on):
            if v and start is None:
                start = i
            elif not v and start is not None:
                if i - start >= 6:
                    out.append((start, i))
                start = None
        if start is not None and len(on) - start >= 6:
            out.append((start, len(on)))
        return out

    def _read_lines(self, crop: np.ndarray) -> str:
        from PIL import Image

        proc, model = self._load_trocr()
        texts: list[str] = []
        boxes = self._line_boxes(crop) or [(0, crop.shape[0])]
        for y0, y1 in boxes[: int(self.cfg.get("max_lines_per_region", 40))]:
            line = crop[max(y0 - 2, 0) : min(y1 + 2, crop.shape[0]), :]
            if line.size == 0 or line.shape[0] < 6:
                continue
            rgb = Image.fromarray(line).convert("RGB")
            pix = proc(images=rgb, return_tensors="pt").pixel_values
            ids = model.generate(pix, max_new_tokens=int(self.cfg.get("max_new_tokens", 64)))
            texts.append(proc.batch_decode(ids, skip_special_tokens=True)[0].strip())
        return "\n".join(t for t in texts if t)

    # ---- the locked entry point ------------------------------------------------------
    def transcribe_region(self, region: Region) -> str:
        """Transcribe ONE region of ONE page."""
        page_id = region.page_id
        doc_id, _, page_no_s = page_id.rpartition("_p")
        page_no = int(page_no_s)

        if self.backend == "page-cache":
            cached = _page_cache(self.ocr_dir, doc_id).get(page_no, "")
            if cached:
                return _slice_page(cached, region, self.full)
            if not bool(self.cfg.get("fallback_on_miss", True)):
                return ""

        img = self._page_image(page_id)
        if img is None:
            return ""
        x0, y0, x1, y1 = region.bbox
        crop = img[max(y0, 0) : y1, max(x0, 0) : x1]
        if crop.size == 0:
            return ""
        return self._read_lines(crop)


def _blocks_of(markdown: str) -> list[tuple[str, str]]:
    """Split one page's Markdown into (kind, text) blocks: heading | table | text."""
    out: list[tuple[str, str]] = []
    buf: list[str] = []
    mode = "text"

    def flush() -> None:
        nonlocal buf, mode
        body = "\n".join(buf).strip()
        if body:
            out.append((mode, body))
        buf, mode = [], "text"

    lines = markdown.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        is_table = stripped.startswith(("|", "<table", "<div"))
        if is_table:
            if mode != "table":
                flush()
                mode = "table"
            buf.append(line)
        elif stripped.startswith("#"):
            flush()
            out.append(("heading", stripped.lstrip("#").strip()))
        elif not stripped:
            if mode == "table":
                flush()
            elif buf:
                flush()
        else:
            if mode == "table":
                flush()
            buf.append(line)
        i += 1
    flush()
    return out


def _slice_page(page_md: str, region: Region, cfg: dict) -> str:
    """Return the part of a cached page transcription that belongs to `region`.

    The cache is page-level, so the mapping is by reading order: the Nth region of a page takes
    the Nth Markdown block, with table regions preferring table blocks. Off-by-one is harmless
    at index time because Stage 4 re-chunks the page as a whole; the mapping matters for the
    region-level demo and for figure regions, where it is what lets a plate's own caption be
    pulled out of the page.
    """
    blocks = _blocks_of(page_md)
    if not blocks:
        return ""
    peers = [r for r in read_regions(cfg).get(region.page_id, [])]
    idx = 0
    for r in peers:
        if tuple(r["bbox"]) == tuple(region.bbox):
            idx = int(r["reading_order"])
            break
    if region.kind == "table":
        tables = [b for k, b in blocks if k == "table"]
        if tables:
            return tables[0]
    if idx < len(blocks):
        return blocks[idx][1]
    return blocks[-1][1]


def transcribe(regions: list[Region], cfg: dict) -> list[Chunk]:
    """Regions -> text chunks (calls Reader).

    Page-level fast path: when the reader's transcription of a page is available, the page is
    emitted as its Markdown blocks — one Chunk per block — because the reader was prompted to
    preserve table and caption structure and re-cutting that output along region boxes would
    throw the structure away. Region typing from Stage 2 still rides along: the block Chunk ids
    are ordered by reading order, and the page's region kinds are what Stage 4 keys on.

    Slow path (no cached page): every region is read individually through
    `Reader.transcribe_region`, which is the same call the held-out OCR evaluation makes.
    """
    reader = Reader(cfg)
    by_page: dict[str, list[Region]] = {}
    for r in regions:
        by_page.setdefault(r.page_id, []).append(r)

    meta = read_page_meta(cfg)
    out: list[Chunk] = []
    n_cached = 0
    for page_id, page_regions in by_page.items():
        doc_id, _, page_no_s = page_id.rpartition("_p")
        page_no = int(page_no_s)
        cached = (
            _page_cache(reader.ocr_dir, doc_id).get(page_no, "")
            if reader.backend == "page-cache"
            else ""
        )
        page_kind = str(meta.get(page_id, {}).get("page_kind", "text"))

        if cached:
            n_cached += 1
            blocks = _blocks_of(cached)
            if not blocks and page_kind == "plate":
                continue
            for i, (kind, text) in enumerate(blocks):
                if not text.strip():
                    continue
                tag = "figure" if (kind == "text" and page_kind == "plate") else kind
                out.append(
                    Chunk(
                        id=f"{page_id}_b{i:02d}",
                        doc_id=doc_id,
                        text=f"<{tag}>\n{text}",
                        page_ids=[page_id],
                    )
                )
        else:
            for i, region in enumerate(page_regions):
                text = reader.transcribe_region(region)
                if not text.strip():
                    continue
                out.append(
                    Chunk(
                        id=f"{page_id}_b{i:02d}",
                        doc_id=doc_id,
                        text=f"<{region.kind}>\n{text}",
                        page_ids=[page_id],
                    )
                )

    words = sum(len(c.text.split()) for c in out)
    log.info(
        f"ocr: {len(out)} blocks / {words} words over {len(by_page)} pages "
        f"({n_cached} from the reader's page transcriptions, backend={reader.backend})"
    )
    return out
