# grading_kit/ — the one folder that makes this project reproducible and gradable.

Open **`manifest.yaml`** first. It declares the three axes (domain · data speciality · primary NFR),
records the corpus size and split, and names every entry point. Nothing else here needs reading to
know what this project is.

## What is in here

- **`manifest.yaml`** — the single entry point. Axes, corpus, splits, models that actually ran,
  and the commands that build and evaluate.
- **`heldout_pages/`** — 16 page-images set aside. No threshold in `configs/config.yaml` was tuned by
  looking at any of them.
- **`labels.jsonl`** — the ground-truth transcription of each held-out page. One JSON object per line:
  `page_id`, `text` (the oracle transcription), plus `doc_id`, `page_no`, `image`, `page_kind`,
  `content_kind`, `quality_band`, `split`, `n_chars`, `n_words`, `label_source`.
- **`tasks.jsonl` + `success_check.py`** — the evaluation questions and their checker. **Authored in
  A3**; the stub line is still in place.

## How the held-out slice was chosen (A2)

Reproduced by the last section of `notebooks/eda.ipynb`, seeded at 42:

1. Start from the 765 pages Stage 1 keeps.
2. **Remove the 17 pages used to calibrate the Stage-1 and Stage-2 thresholds, by name.** A threshold
   set by looking at a page cannot honestly be scored on that page, and this is the one form of
   leakage a page-image pipeline invites.
3. Stratify by volume × content kind: for each of the 4 volumes take 2 prose pages, 1 table-bearing
   page and 1 plate. Prose and table pages must carry ≥ 400 characters of ground truth so a character
   error rate means something; plates are kept short by nature.
4. Sample within each stratum with a seeded RNG and copy the **original greyscale raster** (not the
   preprocessed or binarised copy) into `heldout_pages/`.

Result: 16 pages — 4 per volume, spanning `clean` and `faint` quality bands and all three content
kinds, including 4 plates.

## Where the labels come from

The transcriptions in `labels.jsonl` were produced by an **independent second reader** — Gemini 3
Flash, a different vendor and architecture from the Qwen3-VL reader that produces our index — and then
adjudicated by the team against the page image. This matters: scoring our own reader against its own
output would measure nothing. Using a second reader makes the comparison in `notebooks/kb_demo.ipynb`
a real one, and it is also what lets `microsoft/trocr-base-printed` be scored on the same pages under
the same metric.

The labels are an oracle, not a certification of perfection: where the second reader and our reader
disagree on a faint page, the disagreement is inspected in the notebook rather than assumed to be our
error.

## Using it

```bash
bash scripts/get_data.sh --verify     # confirm the corpus matches data/provenance.md
bash scripts/build_index.sh           # rebuild data/index/ from the raw PDFs
jupyter notebook notebooks/kb_demo.ipynb   # OCR quality on these pages + a working retrieval
```
