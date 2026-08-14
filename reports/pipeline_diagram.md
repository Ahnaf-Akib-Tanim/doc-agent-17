# Knowledge-base pipeline (A2)

`pages → clean → (enhance) → layout → OCR → chunk → embed → store`

The order is fixed by `src/doc_agent/pipeline.py::build_knowledge_base`. What we chose is what runs
inside each box, and one thing that is *not* in the default diagram: **layout output feeds the chunker,
not just the reader.** That is our data speciality — a table row has to survive as a row all the way to
the index, or the measurement it holds is unretrievable.

```mermaid
flowchart TD
    A["data/raw/&lt;doc_id&gt;/*.pdf<br/>4 volumes · 904 page images"]

    subgraph S1["Stage 1 — ingest &amp; preprocess (CPU, classical)"]
        B["loader.load_pages()<br/>PyMuPDF raster @150dpi greyscale<br/>→ pages.jsonl (+ doc-level split)"]
        C["preprocess.run()<br/>MEASURE first: skew · contrast σ · ink/paper gap · speckle<br/>THEN apply: deskew &gt;0.5° · despeckle ≤2px · Sauvola(31, k=0.34)<br/>→ page_kind {text|plate|blank} · quality_band {clean|faint|noisy|skewed}"]
        D["enhance.run() — DISABLED<br/>generative repair not justified: only 1.0% of text pages exceed the deskew gate"]
    end

    subgraph S2["Stage 2 — layout (the data-speciality enhancement)"]
        E["layout.detect()<br/>1 figures first (close→open fuses halftone; type does not) and mask them out<br/>2 column gutters → read column-first, never raster order<br/>3 RLSA smear → lines → blocks<br/>4 table test: ≥60% of rows carry a wide internal gap AND line-fill ≤0.55<br/>→ Region(kind) + regions.jsonl (reading_order, column_idx, table grid)"]
    end

    subgraph S3["Stage 3 — OCR"]
        F["ocr.transcribe() / Reader.transcribe_region()<br/>Qwen3-VL-8B-Instruct, prompted for structured Markdown/HTML<br/>tables stay tables · text inside figures is transcribed<br/>fallback: TrOCR-base-printed, line-by-line, on CPU"]
    end

    subgraph S4["Stage 4 — index"]
        H["chunk.split() — structure-aware<br/>TABLE → one chunk per ROW, header row carried on every row<br/>PROSE → ~256 tokens on paragraph bounds, overlap 32, under its heading<br/>FIGURE/PLATE caption → its own chunk"]
        I["embed.encode()<br/>BAAI/bge-small-en-v1.5, 384-d, L2-normalised"]
        J["store.build()<br/>FAISS HNSW (M=32, efC=200, efS=64), inner product = cosine<br/>→ data/index/{index.faiss, chunks.jsonl, meta.json}"]
    end

    A --> B --> C --> D --> E --> F --> H --> I --> J
    C -. "pages_meta.jsonl<br/>(page_kind, quality_band, paths)" .-> E
    C -. .-> F
    E -. "region kinds + reading order" .-> H

    K(["hooks.AFTER_INGEST"]) -.- D
    L(["hooks.AFTER_OCR<br/>governance/pii.py redacts direct identifiers,<br/>queues person-name spans for a human"]) -.- F
    M(["hooks.BEFORE_INDEX"]) -.- H
```

## What each stage hands the next (the data contracts)

| From → To | Contract | Carried alongside (side-tables, because the contracts are locked to 3–5 fields) |
|---|---|---|
| loader → preprocess | `list[Page]` | `data/interim/pages.jsonl` — page_no, pixel size, source SHA-256, **split** |
| preprocess → layout | `list[Page]` (blanks dropped) | `pages_meta.jsonl` — page_kind, quality_band, skew raw/applied, contrast, separation, speckle, and the three image paths (raw · clean greyscale · Sauvola reader copy) |
| layout → ocr | `list[Region]` in reading order | `regions.jsonl` — reading_order, column_idx, n_columns, bbox_norm, table_id, recovered rows × columns |
| ocr → chunk | `list[Chunk]`, one per typed block | block kind is prefixed on the text (`<table>`, `<heading>`, `<figure>`) so the chunker can key on it |
| chunk → embed | `list[Chunk]` | — |
| embed → store | `np.ndarray (n, 384)` float32, L2-normalised | — |
| store → (A3) | `data/index/` | `meta.json` — embedding checkpoint, dim, index params, OCR + layout model, seed, build time |

`contracts.Page` is `(id, image_path, doc_id)` and `contracts.Region` is `(page_id, bbox, kind)`; neither
may be changed. Everything the speciality needs — reading order, column index, the recovered grid, the
scan-quality slice axes the robustness NFR is scored on — therefore travels in the JSONL side-tables,
keyed by `page_id`. That is a deliberate design decision, not a workaround: it keeps the locked contracts
locked while letting the corpus-specific work be first-class and inspectable on disk.

## Where the two non-default paths sit

- **Plates are protected at Stage 1.** The blank filter would delete a full-page lithograph (few text-like
  components), so plate detection runs *before* the blank test and 96 plates survive into the index.
- **Table rows survive to Stage 4.** A fixed 256-token window cuts a comparative table in half and
  detaches a row label from its column header. `index.chunking: fixed` switches to that flat baseline;
  the two are compared head-to-head in `notebooks/kb_demo.ipynb`.

## One command

```bash
bash scripts/build_index.sh      # rasterise → preprocess → layout → OCR → chunk → embed → FAISS
```
Every stage caches to disk, so a re-run only redoes what changed.
