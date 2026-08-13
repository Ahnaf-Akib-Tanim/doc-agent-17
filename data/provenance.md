# Corpus provenance

**Corpus version:** `corpus_64f400d6` (content-addressed; see `data/raw/corpus.lock.json`, written by
`data.versioning.snapshot()`). Any change to a source PDF changes this id, so an index, a metric and a
form answer can always be tied back to the exact bytes they were produced from.

---

## Source

- **Collection:** *Extinct Species*, Biodiversity Heritage Library — https://www.biodiversitylibrary.org/browse/collection/extinct
- **Scans obtained from:** the Internet Archive items BHL serves those volumes from.
- **What they are:** photographs of paper. Every page is an image before it is text; the working text
  is produced by our own Stage-3 reader, not by the Internet Archive OCR layer. (The IA text layer is
  used nowhere in this pipeline — it is a baseline to beat, not an input.)

| doc_id | Volume | Year | IA identifier | Pages | MB | SHA-256 of the copy we indexed |
|---|---|---|---|---|---|---|
| `lucas_1901` | Lucas, *Animals of the Past* | 1901 | `animalsofpast00luca` | 324 | 11.4 | `09edc73d9b33fda50ad0fe133c7fad3832df03b1fac8c2156f4abf53d306dc3b` |
| `gilmore_1914` | Gilmore, *Osteology of the Armored Dinosauria in the United States National Museum* (USNM Bulletin 89) | 1914 | `osteologyofarmor00gilm` | 234 | 14.0 | `bd7026524000220c196783322c873273816bedb6d2a431ff1113607ceb9068b0` |
| `colbert_1945` | Colbert, *The Dinosaur Book* | 1945 | `dinosauruli13colb` | 164 | 15.5 | `7c31790b76d25971671acfb84bc8c630b1bd3537c0c0937cb16d809bba166348` |
| `leidy_1853` | Leidy, *The Ancient Fauna of Nebraska* | 1853 | `ancientfaunaofne00leid` | 182 | 11.6 | `a1bf321b7f13a1628f8136ee0cbe53f826cd9713165557516ddf1e82cd9739f3` |

> The Internet Archive reports a higher `imagecount` than our page count for three of the four items
> (e.g. 330 vs 324 for Lucas). That is the scanning target, colour card and cover leaves, which the
> distributed PDF does not carry. We record our own count because that is what the pipeline ingests.
>
> **Verification status, checked against a live download from `archive.org` (2026-08-12):**
> `lucas_1901` matches the IA-hosted PDF **byte-for-byte** (same SHA-256). `leidy_1853` does **not**:
> the IA-hosted PDF is 188 pages against our 182, and its `producer` metadata reads
> `Internet Archive PDF ...` against ours reading `iLovePDF` — our working copy had already been
> re-processed (front/back-matter leaves trimmed) by whoever assembled it before it reached this
> team; page-1 text is identical, so it is the same scan, not a different edition. `gilmore_1914` and
> `colbert_1945` were not re-verified this session (connection to archive.org failed after the first
> two downloads). **Consequence:** `bash scripts/get_data.sh --verify` will legitimately report
> `MISMATCH` for `leidy_1853` and possibly the other two — that is not a corruption, and it is not
> safe to "fix" by silently accepting whatever a fresh download returns, because every page number in
> `data/interim/ocr/*.md`, `grading_kit/labels.jsonl` and the built index is pinned to **our** working
> copies' pagination. A grader needing byte-exact reproduction should ask the team for the working
> copies directly rather than re-derive them from a bare IA download.

**Changed since A1 (recorded, not quietly swapped).** A1 declared Allen 1876, *American bisons, living
and extinct*. We replaced it with Gilmore 1914 and added Lucas 1901, because both carry our declared
data speciality harder than Allen did: Gilmore is a USNM systematic monograph — dense quarry and
measurement tables plus 40 numbered plates whose legends are printed on separate leaves — and Lucas is
figure-led throughout. The swap also raised the corpus from 677 to 904 pages. The three axes (domain ·
data speciality · primary NFR) and the success metric are unchanged from A1.

---

## Licence, and whether we may re-share

**LINK-ONLY.** Three volumes are freely re-usable, one is not, so we ship `scripts/get_data.sh` rather
than the PDFs, and `data/raw/` is git-ignored.

| Volume | Rights statement (from the item's own metadata) | Re-shareable |
|---|---|---|
| Leidy 1853 | Smithsonian Institution; public domain (pre-1929 US government-published work) | yes |
| Gilmore 1914 | `NOT_IN_COPYRIGHT` — USNM Bulletin 89, US Government Printing Office | yes |
| Lucas 1901 | Harvard MCZ scan of an 1901 imprint; public domain by date | yes |
| Colbert 1945 | "Copyright American Museum of Natural History. Materials in this collection are made available for personal, non-commercial and educational use" | **no** |

All four are usable for this non-commercial coursework. If this were ever deployed publicly we would
have to drop Colbert or obtain AMNH permission — that is 164 pages, 18% of the corpus, and we are
recording the cost now rather than discovering it later.

---

## Size

**904 scanned pages · 240,893 words · 1,440,366 characters of extracted text · 52.5 MB of source PDF.**
The floor is ≥ 300 pages AND ≥ 60,000 words; we clear it by 3.0× and 4.0×. Standard (non-huge) profile.

| doc_id | Pages | Words (Stage-3 reader) | Share of words |
|---|---|---|---|
| `lucas_1901` | 324 | 51,328 | 21.3% |
| `gilmore_1914` | 234 | 66,814 | 27.7% |
| `colbert_1945` | 164 | 61,277 | 25.4% |
| `leidy_1853` | 182 | 61,474 | 25.5% |

No volume dominates: the largest share of words is 27.7%, so no single book can carry a pooled metric.

After Stage 1 drops blank versos, endpapers and scan artefacts, **765 pages are indexed** (139 dropped).
Of the kept pages, **96 are full-page plates** and are kept deliberately — they are the data speciality,
and a text-only pipeline loses them completely.

---

## Scan & script difficulty

Latin script, three eras of hot-metal printing (1853 · 1901/1914 · 1945). All numbers below are produced
by `notebooks/eda.ipynb` from `data/interim/pages_meta.jsonl`, i.e. measured on all 904 pages, not sampled.

| doc_id | contrast σ | ink/paper separation | speckle | ink coverage | faint pages |
|---|---|---|---|---|---|
| `colbert_1945` | 0.163 | 0.438 | 0.083 | 0.141 | 11 / 152 |
| `lucas_1901` | 0.126 | 0.389 | 0.052 | 0.101 | 40 / 290 |
| `gilmore_1914` | 0.115 | 0.394 | 0.139 | 0.081 | 64 / 180 |
| `leidy_1853` | 0.108 | 0.360 | 0.064 | 0.091 | 103 / 143 |

- **Contrast tracks age exactly.** Colbert 1945 is the crispest (σ 0.163, separation 0.438); Leidy 1853
  is the faintest (σ 0.108, separation 0.360) and 72% of its kept pages fall in the `faint` band. The
  corpus therefore contains its own robustness test before we perturb anything.
- **Skew is small but has a tail.** Over kept text pages: median 0.00°, mean |skew| 0.13°, p90 0.25°,
  max 2.75°; only **1.0%** exceed the 0.5° gate. This is why deskew is conditional — rotating the other
  99% would resample an already low-contrast corpus for nothing.
- **Structure, not script, is the hard part.** 97 pages carry a table (55 in Gilmore, 33 in Leidy);
  96 pages are full-page plates whose figure legends are printed on different leaves; Colbert is
  largely two-column, so raster-order extraction interleaves the columns.
- **Units are not uniform.** Leidy measures in inches and *lines* (1 line = 1/12 inch); Gilmore's USNM
  tables are in millimetres; Colbert uses feet. A number is not comparable across volumes without its
  unit, which is why the header row travels with every table row at Stage 4.

---

## Split policy (by document)

Split is assigned at **document** level in `configs/config.yaml → ingest.splits`, and nowhere else. A page
inherits its volume's split; `data.validate.validate()` asserts that `doc_id → split` is a function, so
a volume cannot appear in two splits and therefore no page can.

| Split | Volume | Pages | Why |
|---|---|---|---|
| train | `lucas_1901` + `gilmore_1914` | 558 | Gilmore carries the measurement tables the pipeline must learn to read; Lucas the figure-led pages |
| val | `colbert_1945` | 164 | typographically the most modern and the most two-column — thresholds calibrated on neither extreme |
| test | `leidy_1853` | 182 | **held out because it is the worst case on every axis we measured**: faintest ink, lowest contrast, 72% of its pages in the `faint` band, and 22 plates. A robustness claim tested on the cleanest volume would be worthless |

**The one way data could leak, and how we checked.** The live risk is the same content appearing in two
volumes — Colbert and Gilmore both describe *Stegosaurus*, and Lucas cites Leidy. Checked in
`notebooks/eda.ipynb`: cross-volume 5-gram Jaccard over content pages, long-token vocabulary overlap per
volume pair, and page-image hashing to catch a plate reused between books. Results are reported there.

---

## Preparation applied (and what it cost)

| Step | What we did | Effect |
|---|---|---|
| Deskew | Projection-profile estimate over −3°…+3° at 0.25°; rotate **only** above 0.5°, and never a plate (a plate has no text lines, so the estimate there is noise) | 0.9% of kept pages rotated; 99% never resampled |
| Despeckle | Drop connected components ≤ 2 px that are **not** within 3 px of a larger component (the proximity rule protects i/j dots, punctuation, accents); plates exempt — in a lithograph the stipple *is* the image | Removes 0.7% (Lucas) to 8.4% (Gilmore) of components |
| Binarize | Sauvola, 31-px window, k = 0.34, written to a **separate** reader image; the greyscale page a citation crops from is never overwritten | Reader gets a cleaner input on the faint volume; citation crops stay trustworthy |
| Blank removal | Ink coverage measured inside a 6% border trim, with the Otsu threshold floored 45 grey levels below the paper median so a blank leaf measures as blank | 904 → 765 pages; all 96 plates survive |
| Plate protection | Plate detected from a close-then-open of the greyscale ink (halftone fuses into a solid block, type does not), tested **before** the blank test | 96 plates kept instead of discarded as low-text pages |
