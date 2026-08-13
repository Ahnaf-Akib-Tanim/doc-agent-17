"""FIXED end-to-end order (Stages 0-9) + cross-cutting seams.
Do not reorder stages or remove hooks.run()/register_all() calls."""
from __future__ import annotations

from . import config, hooks, wiring  # noqa: F401
from .agent import agent
from .index import chunk, embed, store
from .ingest import enhance, loader, preprocess
from .retrieval import retriever
from .vision import layout, ocr


def build_knowledge_base(cfg: dict) -> None:
    wiring.register_all(cfg)                        # wire cross-cutting features
    pages = loader.load_pages(cfg)
    pages = preprocess.run(pages, cfg)
    pages = enhance.run(pages, cfg)                 # Stage 1 - enhancement (VAE/diffusion)
    hooks.run(hooks.AFTER_INGEST, {"pages": pages})
    regions = layout.detect(pages, cfg)             # Stage 2
    text = ocr.transcribe(regions, cfg)             # Stage 3
    hooks.run(hooks.AFTER_OCR, {"chunks": text})    # e.g. PII redaction on extracted text
    chunks = chunk.split(text, cfg)                 # Stage 4
    hooks.run(hooks.BEFORE_INDEX, {"chunks": chunks})
    vectors = embed.encode(chunks, cfg)
    store.build(chunks, vectors, cfg)

def answer(query_text: str, cfg: dict):
    wiring.register_all(cfg)
    r = retriever.Retriever(cfg)                    # Stage 5
    return agent.Agent(cfg, r).run(query_text)      # Stage 6 (seams run inside the loop)
