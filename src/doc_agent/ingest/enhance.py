"""Stage 1 — ENHANCEMENT (VAE / diffusion) — generative denoise / super-resolution."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..contracts import *  # noqa
from ..logging_conf import get_logger
from .preprocess import read_page_meta

log = get_logger(__name__)


class Enhancer:
    """Model set by cfg['enhance'].

    OFF by default for this corpus, and that is a measured decision rather than an omission:
    median skew is 0.00 deg and only ~4% of pages exceed the deskew gate, so a generative
    repair model would invent pixels for a defect most of our pages do not have — and we would
    then be asking a curator to cite them. Two variants live here:

      type: "classical"  — CLAHE + unsharp, applied ONLY to pages the measurements mark faint.
                           Deterministic, invents nothing, runs on CPU.
      type: "vae"/"diffusion" — the generative bonus (C18-C19). Declared deferred in A1 and
                           still deferred: it is re-opened only if Stage-3 word accuracy on the
                           faintest volume trails the cleanest by more than 10 points.

    Whatever runs here writes a separate image; the greyscale page a citation crops from is
    never overwritten.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["enhance"]
        self.full = cfg

    def train(self, pages: list[Page]) -> None:
        """Classical enhancement has no parameters to fit; the generative variant is deferred."""
        kind = str(self.cfg.get("type", "classical"))
        if kind != "classical":
            raise NotImplementedError(
                f"enhance.type={kind!r} (VAE/diffusion) is the declared A4 bonus and is not "
                "trained in A2 — see configs/design_choices.md, stage 1"
            )
        log.info(f"enhance: classical enhancer over {len(pages)} pages — nothing to train")

    def apply(self, pages: list[Page]) -> list[Page]:
        """CLAHE + unsharp on faint pages only; every other page is passed through untouched."""
        kind = str(self.cfg.get("type", "classical"))
        if kind != "classical":
            raise NotImplementedError(
                f"enhance.type={kind!r} (VAE/diffusion) is the declared A4 bonus — set "
                "enhance.type: classical or enhance.enabled: false"
            )
        meta = read_page_meta(self.full)
        work = Path(self.full["ingest"]["work_dir"]) / "enhanced"
        clip = float(self.cfg.get("clahe_clip", 2.0))
        tile = int(self.cfg.get("clahe_tile", 8))
        bands = set(self.cfg.get("apply_to_bands", ["faint"]))
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))

        out: list[Page] = []
        touched = 0
        for page in pages:
            band = str(meta.get(page.id, {}).get("quality_band", "clean"))
            if band not in bands:
                out.append(page)
                continue
            gray = cv2.imread(page.image_path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                out.append(page)
                continue
            eq = clahe.apply(gray)
            blur = cv2.GaussianBlur(eq, (0, 0), 1.2)
            sharp = np.clip(cv2.addWeighted(eq, 1.6, blur, -0.6, 0), 0, 255).astype(np.uint8)
            sub = work / page.doc_id
            sub.mkdir(parents=True, exist_ok=True)
            dst = sub / Path(page.image_path).name
            cv2.imwrite(str(dst), sharp)
            out.append(Page(id=page.id, image_path=str(dst), doc_id=page.doc_id))
            touched += 1
        log.info(f"enhance: classical repair applied to {touched}/{len(pages)} pages ({bands})")
        return out


def run(pages: list[Page], cfg: dict) -> list[Page]:
    if not cfg["enhance"]["enabled"]:
        log.info("enhance: disabled by config — pages pass through unchanged")
        return pages
    return Enhancer(cfg).apply(pages)
