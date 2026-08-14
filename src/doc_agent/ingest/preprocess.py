"""Stage 1 — deskew / denoise / binarize / augment"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..contracts import *  # noqa
from ..logging_conf import get_logger
from .loader import PAGES_MANIFEST

log = get_logger(__name__)

META_MANIFEST = "pages_meta.jsonl"


def read_page_meta(cfg: dict) -> dict[str, dict[str, Any]]:
    """Side-table of per-page measurements, keyed by page_id.

    `contracts.Page` is locked to three fields, so scan-quality measurements travel beside the
    Page objects in this manifest rather than on them. Stages 2-4 read it; nothing writes it
    except this module.
    """
    path = Path(cfg["ingest"]["work_dir"]) / META_MANIFEST
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out[row["page_id"]] = row
    return out


def _estimate_skew(binary: np.ndarray, max_deg: float = 3.0, step: float = 0.25) -> float:
    """Angle (deg) that maximises the variance of the horizontal projection profile.

    Text lines are the strongest horizontal periodicity on a book page: when the page is level,
    the row-ink profile alternates sharply between line and interline, so its variance peaks.
    Measured on a downscaled copy — the peak is a property of the line rhythm, not of resolution.
    """
    small = cv2.resize(binary, (0, 0), fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
    h, w = small.shape
    centre = (w / 2.0, h / 2.0)
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-max_deg, max_deg + 1e-9, step):
        m = cv2.getRotationMatrix2D(centre, float(angle), 1.0)
        rot = cv2.warpAffine(small, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        profile = rot.sum(axis=1, dtype=np.float64)
        score = float(profile.var())
        if score > best_score:
            best_angle, best_score = float(angle), score
    return best_angle


def _deskew(gray: np.ndarray, angle: float) -> np.ndarray:
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)


def _despeckle(binary: np.ndarray, max_px: int) -> tuple[np.ndarray, float]:
    """Drop connected components of <= max_px pixels that are not near a larger component.

    The proximity rule is what protects i/j dots, punctuation and accents: an isolated 2-px blob
    is dirt, the same blob 3 px above an 'i' stem is a tittle.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return binary, 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    is_tiny = np.concatenate([[False], areas <= max_px])
    if not is_tiny.any():
        return binary, 0.0

    big_mask = (~is_tiny[labels] & (labels > 0)).astype(np.uint8)
    near_big = cv2.dilate(big_mask, np.ones((7, 7), np.uint8)) > 0
    # per-label count of pixels that sit next to a larger component — one pass, not one per label
    near_count = np.bincount(labels.ravel(), weights=near_big.ravel(), minlength=n)
    drop = is_tiny & (near_count == 0)
    drop[0] = False
    if not drop.any():
        return binary, 0.0
    out = np.where(drop[labels], np.uint8(0), binary)
    return out, float(drop.sum()) / max(n - 1, 1)


def _sauvola(gray: np.ndarray, window: int, k: float) -> np.ndarray:
    """Sauvola adaptive threshold: T = mean * (1 + k * (std/128 - 1)).

    Adaptive rather than global because ink density varies across one page of a faded volume —
    a single Otsu threshold erases the faint half of the page to hit the dark half.
    """
    win = window if window % 2 == 1 else window + 1
    f = gray.astype(np.float32)
    mean = cv2.boxFilter(f, cv2.CV_32F, (win, win), normalize=True)
    sq = cv2.boxFilter(f * f, cv2.CV_32F, (win, win), normalize=True)
    std = np.sqrt(np.maximum(sq - mean * mean, 0.0))
    thr = mean * (1.0 + k * (std / 128.0 - 1.0))
    return np.where(f > thr, 255, 0).astype(np.uint8)


def _page_stats(gray: np.ndarray) -> dict[str, float]:
    """Scan-quality measurements. These become the slice axes the robustness NFR is scored on.

    Ink is separated with a GLOBAL Otsu threshold here on purpose. Sauvola is the right choice
    for handing an image to the reader, but it adapts away the very thing this function is
    trying to measure: how separable ink is from paper on this page at all.
    """
    h0, w0 = gray.shape
    trim = gray[int(0.06 * h0) : int(0.94 * h0), int(0.06 * w0) : int(0.94 * w0)]
    h, w = trim.shape
    otsu, _ = cv2.threshold(trim, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Otsu always returns a split, even on a blank leaf where it is splitting paper noise into
    # "ink" and "paper" and reports ~40% coverage. Floor it at 45 grey levels below the paper
    # median so a page with no ink measures as having no ink.
    paper = float(np.median(trim))
    thr = min(float(otsu), paper - 45.0)
    ink = (trim < thr).astype(np.uint8)

    coverage = float(ink.mean())
    contrast_sigma = float(trim.std() / 255.0)
    mask = ink.astype(bool)
    ink_mean = float(trim[mask].mean()) if mask.any() else 255.0
    paper_mean = float(trim[~mask].mean()) if (~mask).any() else 255.0
    gap = float(max(paper_mean - ink_mean, 0.0) / 255.0)

    n, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n > 1:
        hts = stats[1:, cv2.CC_STAT_HEIGHT]
        wds = stats[1:, cv2.CC_STAT_WIDTH]
        areas = stats[1:, cv2.CC_STAT_AREA]
        text_like = int(((hts >= 8) & (hts <= 60) & (wds >= 3) & (wds <= 80)).sum())
        speckle = float((areas <= 2).mean())
    else:
        text_like, speckle = 0, 0.0

    # Close the ink, then open it: halftone stipple fuses into one solid block, while lines of
    # type do not. This is what separates a full-page lithographic plate from a page of text.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    solid = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, k)
    solid = cv2.morphologyEx(solid, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n2, _, st2, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
    big = float(st2[1:, cv2.CC_STAT_AREA].max() / (h * w)) if n2 > 1 else 0.0

    return {
        "ink_coverage": coverage,
        "contrast_sigma": contrast_sigma,
        "ink_paper_gap": gap,
        "n_components": int(n - 1),
        "text_like_components": float(text_like),
        "speckle_frac": speckle,
        "large_region_frac": big,
    }


def _classify(st: dict[str, float], skew: float) -> tuple[str, str]:
    """(page_kind, quality_band) from the measurements — not from a hand list of page numbers.

    Plate is tested BEFORE blank. A full-page lithograph has almost no text-like components, so
    a blank test alone would throw away exactly the pages our data speciality is about — the 40
    USNM plates in Gilmore and the 12 in Leidy.
    """
    txt = st["text_like_components"]
    if (txt < 300 and st["ink_coverage"] > 0.02) or (txt < 450 and st["large_region_frac"] > 0.10):
        kind = "plate"
    elif st["ink_coverage"] < 0.004 or txt < 15:
        kind = "blank"
    else:
        kind = "text"

    if abs(skew) >= 0.5:
        band = "skewed"
    elif st["speckle_frac"] > 0.30:
        band = "noisy"
    elif st["contrast_sigma"] < 0.11 or st["ink_paper_gap"] < 0.32:
        band = "faint"
    else:
        band = "clean"
    return kind, band


def run(pages: list[Page], cfg: dict) -> list[Page]:
    """Classical preprocessing — measured first, applied conditionally.

    Per page: measure skew, contrast, ink/paper separation and speckle; deskew ONLY above the
    configured gate; despeckle text pages only (a lithograph's stipple IS its content); write a
    Sauvola-binarised copy for the reader while keeping the greyscale page for citation crops.
    Blank pages are dropped. Everything measured is written to `pages_meta.jsonl`.
    """
    ing: dict[str, Any] = cfg["ingest"]
    work = Path(ing["work_dir"])
    gate = float(ing["deskew_gate_deg"])
    max_px = int(ing["despeckle_max_px"])
    win, k = int(ing["sauvola_window"]), float(ing["sauvola_k"])
    drop_blank = bool(ing.get("drop_blank_pages", True))

    (work / "clean").mkdir(parents=True, exist_ok=True)
    (work / "read").mkdir(parents=True, exist_ok=True)

    raw_meta: dict[str, dict[str, Any]] = {}
    manifest = work / PAGES_MANIFEST
    if manifest.exists():
        with open(manifest, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                raw_meta[row["page_id"]] = row

    kept: list[Page] = []
    rows: list[dict[str, Any]] = []
    for idx, page in enumerate(pages):
        gray = cv2.imread(page.image_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            log.info(f"preprocess: unreadable {page.image_path} — skipped")
            continue

        # measure BEFORE touching anything: page kind and quality decide what may be applied
        stats = _page_stats(gray)
        step = float(ing.get("skew_step_deg", 0.25))
        skew = _estimate_skew(255 - _sauvola(gray, win, k), step=step)
        kind, band = _classify(stats, skew)

        # deskew is conditional AND kind-aware: a plate has no text lines, so the projection
        # peak that drives the estimate is noise there and rotating on it would only blur it
        applied = skew if (abs(skew) > gate and kind != "plate") else 0.0
        work_gray = _deskew(gray, applied) if applied else gray

        ink = 255 - _sauvola(work_gray, win, k)
        if kind != "plate":
            ink, removed = _despeckle(ink, max_px)
        else:
            removed = 0.0

        if kind == "blank" and drop_blank:
            rows.append(
                {
                    "page_id": page.id,
                    "doc_id": page.doc_id,
                    "page_kind": "blank",
                    "quality_band": band,
                    "kept": False,
                    "skew_deg_raw": round(skew, 3),
                    "skew_deg_applied": round(applied, 3),
                    **{k2: round(v, 5) for k2, v in stats.items()},
                    **{
                        j: raw_meta.get(page.id, {}).get(j)
                        for j in ("page_no", "split", "width_px", "height_px", "source_sha256")
                    },
                }
            )
            continue

        sub = work / "clean" / page.doc_id
        sub.mkdir(parents=True, exist_ok=True)
        clean_path = sub / Path(page.image_path).name
        cv2.imwrite(str(clean_path), work_gray)

        rsub = work / "read" / page.doc_id
        rsub.mkdir(parents=True, exist_ok=True)
        read_path = rsub / Path(page.image_path).name
        cv2.imwrite(str(read_path), 255 - ink)

        kept.append(Page(id=page.id, image_path=str(clean_path), doc_id=page.doc_id))
        rows.append(
            {
                "page_id": page.id,
                "doc_id": page.doc_id,
                "page_kind": kind,
                "quality_band": band,
                "kept": True,
                "raw_path": page.image_path,
                "clean_path": str(clean_path),
                "read_path": str(read_path),
                "skew_deg_raw": round(skew, 3),
                "skew_deg_applied": round(applied, 3),
                "despeckled_frac": round(removed, 5),
                **{k2: round(v, 5) for k2, v in stats.items()},
                **{
                    j: raw_meta.get(page.id, {}).get(j)
                    for j in ("page_no", "split", "width_px", "height_px", "source_sha256")
                },
            }
        )
        if (idx + 1) % 100 == 0:
            log.info(f"preprocess: {idx + 1}/{len(pages)} pages")

    with open(work / META_MANIFEST, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    dropped = len(pages) - len(kept)
    log.info(f"preprocess: kept {len(kept)} pages, dropped {dropped} blank/unreadable")
    return kept
