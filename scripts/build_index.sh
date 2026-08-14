#!/usr/bin/env bash
# A2 — build the vector index in ONE command (wraps: make seed ingest)
#
# `make ingest` and `make index` both call pipeline.build_knowledge_base — the index is built
# inside it, as run_index.py's own comment says — so this script calls the pipeline ONCE
# instead of running the whole ingest twice.
#
#   bash scripts/build_index.sh
#
# Runs pipeline.build_knowledge_base(config.load()) end to end:
#   data/raw/*.pdf -> rasterise -> preprocess (measure, deskew, despeckle, binarise)
#                  -> layout (columns, table grid, plates) -> OCR -> chunk -> embed -> FAISS
# Every stage is cached on disk, so a re-run only redoes what changed.
# Output: data/index/{index.faiss, chunks.jsonl, meta.json}
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src:${PYTHONPATH:-}"

python scripts/set_seed.py
bash scripts/get_data.sh --verify
make ingest

python - <<'PY'
import json, pathlib
meta = json.loads(pathlib.Path("data/index/meta.json").read_text())
print("\nindex built:")
for k in ("index_type","n_chunks","n_pages","n_docs","n_words","dim","embed_model","ocr_model","build_seconds"):
    print(f"  {k:14s} {meta[k]}")
PY
