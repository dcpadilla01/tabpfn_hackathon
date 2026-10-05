#!/usr/bin/env bash
# Download the six best feature tables of the reported runs (Env-1) into experiments/results/.
# Needed only for: src.analysis.transfer and the reproduction check in scripts/evaluate_test.py.
set -euo pipefail
URL="https://github.com/dcpadilla01/tabpfn_hackathon/releases/download/env1-artifacts/env1_best_feature_tables.tar"
SHA256="5b9976e26cc154b8c004d4dc8a598405654de888ab8183cdeda6099f351a1d80"
cd "$(dirname "$0")/.."
tmp="$(mktemp)"
curl -fL --retry 3 -o "$tmp" "$URL"
echo "$SHA256  $tmp" | shasum -a 256 -c -
tar -xf "$tmp"
rm -f "$tmp"
echo "extracted the six best feature tables into experiments/results/"
