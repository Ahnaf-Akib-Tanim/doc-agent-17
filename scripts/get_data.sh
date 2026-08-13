#!/usr/bin/env bash
# A1 — fetch or recreate your scanned corpus into data/raw/
#
# Our corpus is LINK-ONLY: three of the four volumes are public domain, one (Colbert 1945) is
# American Museum of Natural History copyright released for personal / educational use only,
# so we ship this script rather than the PDFs. Every volume is fetched from its Internet
# Archive item — the same scans the Biodiversity Heritage Library serves.
#
# KNOWN, CHECKED ISSUE: the working copies this repo's OCR/index were built from are not
# guaranteed to be byte-identical to a fresh IA download. Verified 2026-08-12 against a live
# download: lucas_1901 matches byte-for-byte; leidy_1853 does NOT (IA serves 188 pages, our
# working copy is 182 — 6 fewer front/back-matter leaves — because our copy had already been
# re-processed by something reporting itself as "iLovePDF" before this team received it; same
# scan, same page-1 text, different trim). gilmore_1914 / colbert_1945 were not re-verified.
# A MISMATCH below is therefore expected for at least leidy_1853 — it does NOT mean the file is
# corrupt or that Internet Archive changed anything, and it should NOT be silently patched over,
# because every page number in data/interim/ocr/*.md, grading_kit/labels.jsonl and data/index/
# is pinned to OUR working copies' pagination, not to whatever a bare download returns today.
#
#   bash scripts/get_data.sh            # fetch + verify
#   bash scripts/get_data.sh --verify   # verify what is already on disk
set -euo pipefail
cd "$(dirname "$0")/.."

RAW=data/raw
OCR=data/interim/ocr

# doc_id | internet-archive identifier | sha256 of the PDF we built the index from
DOCS=(
  "lucas_1901|animalsofpast00luca|09edc73d9b33fda5"
  "gilmore_1914|osteologyofarmor00gilm|bd7026524000220c"
  "colbert_1945|dinosauruli13colb|7c31790b76d25971"
  "leidy_1853|ancientfaunaofne00leid|a1bf321b7f13a162"
)

verify_only=0
[[ "${1:-}" == "--verify" ]] && verify_only=1

for row in "${DOCS[@]}"; do
  IFS='|' read -r doc ia sha <<< "$row"
  dest="$RAW/$doc/$doc.pdf"
  mkdir -p "$RAW/$doc"
  if [[ ! -f "$dest" && $verify_only -eq 0 ]]; then
    url="https://archive.org/download/$ia/$ia.pdf"
    echo "fetching $doc from $url"
    curl -fL --retry 3 -o "$dest" "$url"
  fi
  if [[ -f "$dest" ]]; then
    got=$(sha256sum "$dest" | cut -c1-16)
    if [[ "$got" == "$sha" ]]; then
      echo "ok       $doc  sha256:$got"
    else
      echo "MISMATCH $doc  expected $sha got $got"
      echo "         Either this file was re-derived, OR (confirmed true for leidy_1853 as of"
      echo "         2026-08-12) our working copy was already re-processed/trimmed before this"
      echo "         team received it, so a bare IA download will not match it page-for-page."
      echo "         DO NOT just accept the downloaded file: page numbers throughout this repo"
      echo "         (Stage-3 transcriptions, grading_kit/labels.jsonl, the built index) are"
      echo "         pinned to OUR pagination, not IA's. Get the working copy from the team, or"
      echo "         re-derive the whole pipeline (rasterise -> re-OCR -> re-label -> rebuild) if"
      echo "         you intend to use the fresh download instead."
    fi
  else
    echo "missing  $doc  ($dest)"
  fi
done

# Stage-3 reader output. The transcriptions are OURS (Qwen3-VL-8B; the run is documented in
# notebooks/kb_demo.ipynb). The three public-domain volumes are committed so the index can be
# rebuilt without a GPU. colbert_1945.md is NOT committed: redistributing a full transcription of
# an AMNH-copyright volume is the same act as redistributing the scan. Re-create it by running the
# Stage-3 reader over data/raw/colbert_1945/, or build the index without it — the remaining three
# volumes are 740 pages / 179,616 words, still clear of the 300-page / 60,000-word floor.
mkdir -p "$OCR"
missing=0
for row in "${DOCS[@]}"; do
  IFS='|' read -r doc _ _ <<< "$row"
  if [[ ! -f "$OCR/$doc.md" ]]; then
    if [[ "$doc" == "colbert_1945" ]]; then
      echo "absent   $OCR/$doc.md — withheld by licence; run the Stage-3 reader to regenerate"
    else
      echo "missing  $OCR/$doc.md — run the Stage-3 reader over data/raw/$doc/"
      missing=1
    fi
  fi
done
[[ $missing -eq 0 ]] && echo "ok       Stage-3 transcriptions present for every shareable volume"

echo
echo "next: bash scripts/build_index.sh"
