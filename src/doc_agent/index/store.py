"""Stage 4 — vector store"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import *  # noqa
from ..logging_conf import get_logger

log = get_logger(__name__)

INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.jsonl"
META_FILE = "meta.json"


def _dir(cfg: dict) -> Path:
    return Path(cfg["index"].get("dir", "data/index"))


def _make(kind: str, dim: int, n: int, cfg_i: dict[str, Any]) -> Any:
    import faiss

    if kind.endswith("flat"):
        return faiss.IndexFlatIP(dim)
    if kind.endswith("hnsw"):
        m = int(cfg_i.get("hnsw_m", 32))
        idx = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
        idx.hnsw.efConstruction = int(cfg_i.get("ef_construction", 200))
        idx.hnsw.efSearch = int(cfg_i.get("ef_search", 64))
        return idx
    if kind.endswith("ivf"):
        nlist = int(cfg_i.get("nlist", max(4, int(np.sqrt(max(n, 1))))))
        quant = faiss.IndexFlatIP(dim)
        # separate name from the hnsw branch's `idx`: faiss ships concrete stubs per index
        # class, so mypy treats reusing the name across branches as a redefinition
        ivf_idx = faiss.IndexIVFFlat(quant, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        return ivf_idx
    raise ValueError(f"index.type={kind!r} — expected faiss:flat, faiss:hnsw or faiss:ivf")


def build(chunks: list[Chunk], vectors: np.ndarray, cfg: dict) -> None:
    """Persist a vector index (cfg['index']['type']).

    Three index types are supported and configurable because the right one is a corpus-size
    decision, not a taste one: at ~10k chunks a flat index is exact and already sub-millisecond,
    while HNSW is what keeps that latency when the corpus grows. We ship HNSW and report its
    recall against the exact flat index in notebooks/kb_demo.ipynb, so the approximation is
    measured rather than assumed.

    Written beside the index: every chunk (so ids stay stable and citations resolve without a
    database) and a meta.json recording the embedding checkpoint, dimension, index parameters
    and the corpus version — an index whose provenance is unknown cannot be reproduced.
    """
    import faiss

    cfg_i: dict[str, Any] = cfg["index"]
    out = _dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    kind = str(cfg_i["type"]).lower()
    n, dim = vectors.shape if vectors.size else (0, int(cfg["embed"]["dim"]))
    if n != len(chunks):
        raise ValueError(f"{n} vectors vs {len(chunks)} chunks — refusing to build a skewed index")

    t0 = time.time()
    index = _make(kind, dim, n, cfg_i)
    if not index.is_trained:
        index.train(vectors)
    index.add(vectors)
    faiss.write_index(index, str(out / INDEX_FILE))
    build_s = time.time() - t0

    with open(out / CHUNKS_FILE, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(
                json.dumps(
                    {
                        "id": c.id,
                        "doc_id": c.doc_id,
                        "text": c.text,
                        "page_ids": c.page_ids,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    pages = {p for c in chunks for p in c.page_ids}
    meta = {
        "index_type": kind,
        "n_chunks": len(chunks),
        "n_pages": len(pages),
        "n_docs": len({c.doc_id for c in chunks}),
        "n_words": sum(len(c.text.split()) for c in chunks),
        "dim": int(dim),
        "embed_model": cfg["embed"]["model"],
        "chunking": cfg_i.get("chunking", "structure"),
        "chunk_tokens": cfg_i.get("chunk_tokens"),
        "overlap": cfg_i.get("overlap"),
        "hnsw_m": cfg_i.get("hnsw_m"),
        "ef_construction": cfg_i.get("ef_construction"),
        "ef_search": cfg_i.get("ef_search"),
        "ocr_model": cfg["ocr"]["model"],
        "layout_model": cfg["layout"]["model"],
        "seed": cfg.get("seed"),
        "build_seconds": round(build_s, 2),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info(f"store: {kind} index of {len(chunks)} x {dim}-d built in {build_s:.1f}s -> {out}")


def load(cfg: dict) -> dict[str, Any]:
    """Load the persisted index: {'index', 'chunks', 'meta'}."""
    import faiss

    out = _dir(cfg)
    path = out / INDEX_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing — run scripts/build_index.sh first")
    index = faiss.read_index(str(path))
    chunks: list[Chunk] = []
    with open(out / CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            chunks.append(
                Chunk(
                    id=row["id"],
                    doc_id=row["doc_id"],
                    text=row["text"],
                    page_ids=row["page_ids"],
                )
            )
    meta = json.loads((out / META_FILE).read_text(encoding="utf-8"))
    if hasattr(index, "hnsw"):
        index.hnsw.efSearch = int(cfg["index"].get("ef_search", 64))
    if index.ntotal != len(chunks):
        raise ValueError(
            f"index holds {index.ntotal} vectors but chunks.jsonl has {len(chunks)} rows — "
            "the index and its payload are out of sync; rebuild"
        )
    log.info(f"store: loaded {meta['index_type']} index, {index.ntotal} vectors, dim {meta['dim']}")
    return {"index": index, "chunks": chunks, "meta": meta}
