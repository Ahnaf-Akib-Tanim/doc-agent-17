"""Stage 2 — layout detection / segmentation"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..contracts import *  # noqa
from ..ingest.preprocess import read_page_meta
from ..logging_conf import get_logger

log = get_logger(__name__)

REGIONS_MANIFEST = "regions.jsonl"

Box = tuple[int, int, int, int]

_REGION_CACHE: dict[str, dict[str, list[dict[str, Any]]]] = {}


def read_regions(cfg: dict) -> dict[str, list[dict[str, Any]]]:
    """Region side-table keyed by page_id.

    `contracts.Region` is locked to (page_id, bbox, kind). Reading order, column index and the
    recovered table grid — the things our data speciality actually needs — travel here.
    """
    path = Path(cfg["ingest"]["work_dir"]) / REGIONS_MANIFEST
    key = str(path.resolve())
    if key in _REGION_CACHE:
        return _REGION_CACHE[key]
    out: dict[str, list[dict[str, Any]]] = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out.setdefault(row["page_id"], []).append(row)
    _REGION_CACHE[key] = out
    return out


def _ink(path: str) -> np.ndarray | None:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return _trim((img < 128).astype(np.uint8))


def _tone_ink(path: str) -> np.ndarray | None:
    """Ink mask from the GREYSCALE page, for finding pictures.

    Deliberately not the Sauvola image the reader gets: adaptive thresholding thins halftone
    stipple until a lithograph stops looking like a solid object, which is precisely the
    property that identifies it as one.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    otsu, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # numpy's median() stub overloads reject the union dtype cv2.threshold's MatLike return
    # resolves to; runtime behaviour is correct (img is a plain uint8/float ndarray here).
    thr = min(float(otsu), float(np.median(img)) - 45.0)  # type: ignore[arg-type]
    return _trim((img < thr).astype(np.uint8))


def _trim(ink: np.ndarray) -> np.ndarray:
    h, w = ink.shape
    m = int(0.02 * min(h, w))
    ink[:m, :] = 0
    ink[-m:, :] = 0
    ink[:, :m] = 0
    ink[:, -m:] = 0
    return ink


def _rules(ink: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Long horizontal / vertical strokes — table rules, when a table has them."""
    h, w = ink.shape
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 12, 20), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 20, 20)))
    return cv2.morphologyEx(ink, cv2.MORPH_OPEN, hk), cv2.morphologyEx(ink, cv2.MORPH_OPEN, vk)


def _lines(ink: np.ndarray, smear: int) -> list[Box]:
    """Text lines via horizontal run-length smearing, then connected components."""
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(smear, 3), 1))
    smeared = cv2.dilate(ink, k)
    n, _, stats, _ = cv2.connectedComponentsWithStats(smeared, connectivity=8)
    boxes: list[Box] = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bw < 12 or bh < 5 or area < 40:
            continue
        boxes.append((int(x), int(y), int(x + bw), int(y + bh)))
    return sorted(boxes, key=lambda b: (b[1], b[0]))


def _columns(ink: np.ndarray, min_gutter_frac: float) -> list[tuple[int, int]]:
    """Column bounds from a vertical whitespace gutter in the central band of the page.

    Colbert 1945 is two-column on most of its text pages; reading a two-column page in raster
    order interleaves the columns and destroys the sentences, so column bounds are recovered
    before anything is grouped into blocks.
    """
    h, w = ink.shape
    col = ink.sum(axis=0).astype(np.float64)
    if col.max() <= 0:
        return [(0, w)]
    col = col / col.max()
    lo, hi = int(0.30 * w), int(0.70 * w)
    quiet = col[lo:hi] < 0.04
    best: tuple[int, int] | None = None
    start: int | None = None
    for i, q in enumerate(quiet):
        if q and start is None:
            start = i
        elif not q and start is not None:
            if best is None or (i - start) > (best[1] - best[0]):
                best = (start, i)
            start = None
    if start is not None and (best is None or (len(quiet) - start) > (best[1] - best[0])):
        best = (start, len(quiet))
    if best is None or (best[1] - best[0]) < min_gutter_frac * w:
        return [(0, w)]
    mid = lo + (best[0] + best[1]) // 2
    return [(0, mid), (mid, w)]


def _blocks(lines: list[Box], gap_factor: float) -> list[list[Box]]:
    """Merge vertically adjacent lines into blocks; a wide vertical gap starts a new block."""
    if not lines:
        return []
    heights = np.array([b[3] - b[1] for b in lines])
    med_h = float(np.median(heights))
    out: list[list[Box]] = [[lines[0]]]
    for prev, cur in zip(lines, lines[1:], strict=False):
        gap = cur[1] - prev[3]
        if gap > gap_factor * med_h:
            out.append([cur])
        else:
            out[-1].append(cur)
    return out


def _grid(ink: np.ndarray, box: Box, min_gap_frac: float) -> tuple[int, float, float, int]:
    """(n_rows, gapped_row_frac, mean_line_fill, n_cols) inside a candidate region.

    These four numbers are how a measurement table is told apart from a paragraph without
    reading a word of it. Justified body type fills its own line: between the first and last
    ink pixel of a prose line there is almost no whitespace wider than a word gap, so mean fill
    lands around 0.72 on this corpus. A table ROW is mostly whitespace between its cells —
    fill lands around 0.35-0.50, and nearly every row carries at least one wide internal gap.

    n_cols counts the vertical whitespace corridors that persist down the whole block, which is
    how these books actually rule their tables: by alignment, not by printed lines. It is what
    lets a recovered row be re-attached to the right column header.
    """
    x0, y0, x1, y1 = box
    crop = ink[y0:y1, x0:x1]
    if crop.size == 0:
        return 0, 0.0, 0.0, 0
    width = crop.shape[1]
    min_gap = max(int(min_gap_frac * width), 12)

    on = crop.sum(axis=1) > 0
    lines: list[tuple[int, int]] = []
    start: int | None = None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= 5:
                lines.append((start, i))
            start = None
    if start is not None and len(on) - start >= 5:
        lines.append((start, len(on)))
    if not lines:
        return 0, 0.0, 0.0, 0

    gapped, fills = 0, []
    for a, b in lines:
        seg = crop[a:b].sum(axis=0) > 0
        xs = np.flatnonzero(seg)
        if xs.size == 0:
            continue
        span = seg[xs[0] : xs[-1] + 1]
        fills.append(float(span.mean()))
        run, has = 0, False
        for v in span:
            if not v:
                run += 1
                if run >= min_gap:
                    has = True
                    break
            else:
                run = 0
        gapped += int(has)

    quiet = crop.sum(axis=0) <= max(1, int(0.02 * crop.shape[0]))
    corridors, run = 0, 0
    for q in quiet:
        if q:
            run += 1
        else:
            if run >= min_gap:
                corridors += 1
            run = 0
    if run >= min_gap:
        corridors += 1

    return (
        len(lines),
        gapped / len(lines),
        float(np.mean(fills)) if fills else 1.0,
        max(corridors - 1, 0),
    )


def _figures(ink: np.ndarray, cfg_layout: dict[str, Any]) -> list[Box]:
    """Pictorial regions, found BEFORE any text analysis.

    Halftone and lithograph stipple fuses into a solid block under a close-then-open; lines of
    type do not. Finding figures first and masking them out is what stops a full-page plate
    from being shredded into forty spurious "text lines", and it is what recovers the embedded
    photographs that carry their own lettering.
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    solid = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, k)
    solid = cv2.morphologyEx(solid, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    h, w = ink.shape
    page_area = h * w
    n, _, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
    out: list[Box] = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < cfg_layout["figure_min_area_frac"] * page_area:
            continue
        crop = ink[y : y + bh, x : x + bw]
        if crop.size and float(crop.mean()) < cfg_layout["figure_min_density"]:
            continue
        if bw < 0.15 * w or bh < 0.08 * h:
            continue
        out.append((int(x), int(y), int(x + bw), int(y + bh)))
    return out


def _classify(
    ink: np.ndarray,
    block: list[Box],
    box: Box,
    hrule: np.ndarray,
    med_h: float,
    cfg_layout: dict[str, Any],
) -> tuple[str, int, int]:
    x0, y0, x1, y1 = box
    ruled = int((hrule[y0:y1, x0:x1].sum(axis=1) > 0.4 * (x1 - x0)).sum())
    n_rows, gap_frac, fill, n_cols = _grid(ink, box, float(cfg_layout["table_gap_frac_width"]))
    gappy = gap_frac >= float(cfg_layout["table_gap_rows"])
    sparse = fill <= float(cfg_layout["table_max_fill"])
    is_table = n_rows >= int(cfg_layout["table_min_rows"]) and ((gappy and sparse) or ruled >= 2)
    if is_table:
        return "table", n_rows, max(n_cols, int(cfg_layout["table_min_cols"]))

    if len(block) == 1:
        bh = y1 - y0
        wide_enough = (x1 - x0) < 0.8 * ink.shape[1]
        if bh > cfg_layout["heading_height_factor"] * med_h and wide_enough:
            return "heading", 0, 0
    return "text", 0, 0


def detect(pages: list[Page], cfg: dict, *, write_manifest: bool = True) -> list[Region]:
    """Detect text/table/figure/heading regions.

    Heuristic detector, deliberately: it is deterministic, needs no GPU and no labelled layout
    data, and every threshold below is set from a distribution we measured on this corpus rather
    than inherited from a benchmark of modern born-digital documents. It does three things a
    plain region detector does not, all three forced by our table/figure-heavy speciality:

      1. recovers COLUMN bounds before grouping, so a two-column page is read column-first;
      2. recovers the table GRID (rows x columns) from whitespace corridors, so a measurement
         row stays attached to its own header;
      3. keeps full-page plates as `figure` regions instead of discarding low-text pages.

    Regions come back in reading order. The grid detail lands in `regions.jsonl`, because
    `contracts.Region` is locked to three fields.

    `write_manifest` defaults to True for the real pipeline call. `regions.jsonl` is a SHARED
    file other stages read (`read_regions()`, called from `vision/ocr.py`), and this function
    writes it with `open(..., "w")` — a full overwrite, not a merge. Calling `detect()` again on
    a subset of pages (a REPL check, an ad-hoc debug script) would otherwise silently truncate
    the real manifest down to just that subset. Pass `write_manifest=False` for any exploratory
    call that isn't the actual pipeline run.
    """
    lay: dict[str, Any] = dict(cfg["layout"])
    lay.setdefault("smear_frac", 0.020)
    lay.setdefault("block_gap_factor", 1.15)
    lay.setdefault("min_gutter_frac", 0.020)
    lay.setdefault("table_min_cols", 2)
    lay.setdefault("table_min_rows", 3)
    lay.setdefault("table_gap_rows", 0.60)
    lay.setdefault("table_max_fill", 0.55)
    lay.setdefault("table_gap_frac_width", 0.035)
    lay.setdefault("figure_min_area_frac", 0.10)
    lay.setdefault("figure_min_density", 0.22)
    lay.setdefault("heading_height_factor", 1.30)

    meta = read_page_meta(cfg)
    work = Path(cfg["ingest"]["work_dir"])
    regions: list[Region] = []
    rows: list[dict[str, Any]] = []

    for n_done, page in enumerate(pages, 1):
        row = meta.get(page.id, {})
        ink = _ink(str(row.get("read_path", page.image_path)))
        tone = _tone_ink(str(row.get("clean_path", page.image_path)))
        if ink is None or tone is None:
            continue
        h, w = ink.shape
        hrule, _vrule = _rules(ink)

        figures = _figures(tone, lay)
        if not figures and str(row.get("page_kind")) == "plate":
            # a plate whose stipple was too light to fuse — the page-level measurement already
            # says it is one, so bound the ink rather than pretend the picture is not there
            ys, xs = np.nonzero(tone)
            if ys.size:
                figures = [(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))]
        text_ink = cv2.subtract(ink, hrule)
        for fx0, fy0, fx1, fy1 in figures:
            text_ink[fy0:fy1, fx0:fx1] = 0

        cols = _columns(text_ink, float(lay["min_gutter_frac"]))
        smear = max(int(lay["smear_frac"] * w), 6)
        page_blocks: list[tuple[int, Box, list[Box] | None]] = []
        for ci, (cx0, cx1) in enumerate(cols):
            strip = np.zeros_like(text_ink)
            strip[:, cx0:cx1] = text_ink[:, cx0:cx1]
            lines = _lines(strip, smear)
            for blk in _blocks(lines, float(lay["block_gap_factor"])):
                bx0 = min(b[0] for b in blk)
                by0 = min(b[1] for b in blk)
                bx1 = max(b[2] for b in blk)
                by1 = max(b[3] for b in blk)
                page_blocks.append((ci, (bx0, by0, bx1, by1), blk))
        for fbox in figures:
            cx = (fbox[0] + fbox[2]) // 2
            ci = next((i for i, (a, b) in enumerate(cols) if a <= cx < b), 0)
            page_blocks.append((ci, fbox, None))

        if not page_blocks:
            continue
        all_lines = [b for _, _, blk in page_blocks if blk for b in blk]
        med_h = float(np.median([b[3] - b[1] for b in all_lines])) if all_lines else 12.0
        page_blocks.sort(key=lambda t: (t[0], t[1][1], t[1][0]))

        table_seq = 0
        for order, (ci, box, region_lines) in enumerate(page_blocks):
            if region_lines is None:
                kind, n_rows, n_cols = "figure", 0, 0
            else:
                kind, n_rows, n_cols = _classify(ink, region_lines, box, hrule, med_h, lay)
            table_id = ""
            if kind == "table":
                table_seq += 1
                table_id = f"{page.id}_t{table_seq:02d}"
            regions.append(Region(page_id=page.id, bbox=box, kind=kind))
            rows.append(
                {
                    "page_id": page.id,
                    "doc_id": page.doc_id,
                    "region_idx": order,
                    "reading_order": order,
                    "column_idx": ci,
                    "n_columns": len(cols),
                    "bbox": list(box),
                    "bbox_norm": [
                        round(box[0] / w, 5),
                        round(box[1] / h, 5),
                        round(box[2] / w, 5),
                        round(box[3] / h, 5),
                    ],
                    "kind": kind,
                    "n_lines": len(region_lines) if region_lines else 0,
                    "table_id": table_id,
                    "table_rows": n_rows,
                    "table_cols": n_cols,
                    "page_kind": meta.get(page.id, {}).get("page_kind", "text"),
                }
            )
        if n_done % 100 == 0:
            log.info(f"layout: {n_done}/{len(pages)} pages")

    if write_manifest:
        with open(work / REGIONS_MANIFEST, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        _REGION_CACHE.pop(str((work / REGIONS_MANIFEST).resolve()), None)

    counts: dict[str, int] = {}
    for r in regions:
        counts[r.kind] = counts.get(r.kind, 0) + 1
    log.info(f"layout: {len(regions)} regions over {len(pages)} pages {counts}")
    return regions
