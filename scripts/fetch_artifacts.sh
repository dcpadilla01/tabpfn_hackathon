#!/bin/sh
# Download the best feature tables of the reported runs into experiments/ (sha256-checked).
#   sh scripts/fetch_artifacts.sh [env1|env2|all]     (default: all)
# Needed only for the representation-transfer check and the reproduction check in scripts/evaluate_test.py.
set -eu
BASE="https://github.com/dcpadilla01/tabpfn_hackathon/releases/download"
cd "$(dirname "$0")/.."

fetch() {  # env tag  file  sha256
  tmp="$(mktemp)"
  curl -fL --retry 3 -o "$tmp" "$BASE/$2/$3"
  echo "$4  $tmp" | shasum -a 256 -c -
  tar -xf "$tmp"
  rm -f "$tmp"
  echo "$1: extracted the six best feature tables"
}
env1() { fetch env1 env1-artifacts env1_best_feature_tables.tar 5b9976e26cc154b8c004d4dc8a598405654de888ab8183cdeda6099f351a1d80; }
env2() { fetch env2 env2-artifacts env2_best_feature_tables.tar b4c80913caba3645e0ce0ee3807df9df8103135aa3639dcbf1332f37331147a2; }

case "${1:-all}" in
  env1) env1 ;;
  env2) env2 ;;
  all)  env1; env2 ;;
  *)    echo "usage: $0 [env1|env2|all]" >&2; exit 2 ;;
esac
