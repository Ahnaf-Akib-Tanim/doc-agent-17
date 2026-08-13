"""Stage 4 — embed chunks"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import *  # noqa
from ..logging_conf import get_logger

log = get_logger(__name__)

_MODEL_CACHE: dict[str, Any] = {}


def load_model(cfg: dict) -> Any:
    """One loaded SentenceTransformer per process, keyed by checkpoint name."""
    from sentence_transformers import SentenceTransformer

    name = str(cfg["embed"]["model"])
    if name not in _MODEL_CACHE:
        device = str(cfg.get("device", "auto"))
        if device == "auto":
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"embed: loading {name} on {device}")
        _MODEL_CACHE[name] = SentenceTransformer(name, device=device)
    return _MODEL_CACHE[name]


def encode(chunks: list[Chunk], cfg: dict) -> np.ndarray:
    """Embed with cfg['embed']['model'].

    Vectors are L2-normalised so the FAISS inner-product index computes cosine similarity —
    which keeps a one-line table row comparable with a 250-token paragraph instead of letting
    length dominate the score, and it is what makes a single weak-evidence threshold meaningful
    across chunk types.

    The result is cached under `work_dir/vectors/` keyed by the model name plus a hash of the
    chunk texts, so re-running the build after a config change that does not touch the text is
    free and byte-identical.
    """
    cfg_e: dict[str, Any] = cfg["embed"]
    texts = [c.text for c in chunks]
    if not texts:
        return np.zeros((0, int(cfg_e["dim"])), dtype="float32")

    digest = hashlib.sha256(("␟".join(texts)).encode("utf-8")).hexdigest()[:16]
    tag = str(cfg_e["model"]).replace("/", "__")
    cache = Path(cfg["ingest"]["work_dir"]) / "vectors" / f"{tag}.{digest}.npy"
    if cache.exists() and bool(cfg_e.get("cache", True)):
        vecs = np.load(cache)
        log.info(f"embed: reused cached vectors {vecs.shape} from {cache.name}")
        return vecs.astype("float32")

    model = load_model(cfg)
    prefix = str(cfg_e.get("passage_prefix", ""))
    payload = [prefix + t for t in texts] if prefix else texts
    vecs = model.encode(
        payload,
        batch_size=int(cfg_e.get("batch", 64)),
        normalize_embeddings=bool(cfg_e.get("normalize", True)),
        show_progress_bar=bool(cfg_e.get("progress", False)),
        convert_to_numpy=True,
    ).astype("float32")

    dim = int(cfg_e["dim"])
    if vecs.shape[1] != dim:
        raise ValueError(
            f"embed.dim={dim} in configs/config.yaml but {cfg_e['model']} returns "
            f"{vecs.shape[1]} — fix the config, the index is built from it"
        )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, vecs)
    log.info(f"embed: {vecs.shape[0]} chunks -> {vecs.shape[1]}-d ({cfg_e['model']})")
    return vecs


def encode_queries(queries: list[str], cfg: dict) -> np.ndarray:
    """Same encoder, query-side prefix. Kept here so query and passage never drift apart."""
    cfg_e = cfg["embed"]
    prefix = str(cfg_e.get("query_prefix", ""))
    model = load_model(cfg)
    return model.encode(
        [prefix + q for q in queries],
        normalize_embeddings=bool(cfg_e.get("normalize", True)),
        convert_to_numpy=True,
    ).astype("float32")
