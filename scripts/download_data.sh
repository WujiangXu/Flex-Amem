#!/usr/bin/env bash
# Download LongMemEval-S (cleaned), SWE-bench Verified and LoCoMo into artifacts/data/ and check every file the runners
# read against the sha256 used in the paper's runs. AppWorld is installed separately (scripts/setup_appworld.sh).
# usage: scripts/download_data.sh            (run from anywhere; uses .venv/bin/python unless PYTHON is set)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-.venv/bin/python}
HF=https://huggingface.co/datasets
mkdir -p artifacts/data/longmemeval artifacts/data/swebench artifacts/data/locomo

curl -fsSL -o artifacts/data/longmemeval/longmemeval_s_cleaned.json \
  $HF/xiaowu0162/longmemeval-cleaned/resolve/98d7416c24c778c2fee6e6f3006e7a073259d48f/longmemeval_s_cleaned.json
curl -fsSL -o artifacts/data/swebench/verified_test.parquet \
  $HF/princeton-nlp/SWE-bench_Verified/resolve/c104f840cc67f8b6eec6f759ebc8b2693d585d4a/data/test-00000-of-00001.parquet
"$PY" -c "import json, pyarrow.parquet as pq; d = 'artifacts/data/swebench/'; \
json.dump(pq.read_table(d + 'verified_test.parquet').to_pylist(), open(d + 'verified_test.json', 'w'))"
# LoCoMo: a mirror of the official release (snap-research/locomo, data/locomo10.json); the runner asserts this hash
curl -fsSL -o artifacts/data/locomo/locomo10.json \
  $HF/KimmoZZZ/locomo/resolve/6e504a8184822c3b2bc1669d8cc8e05a2cfd0cb8/locomo10.json

sha256sum -c - <<'SUMS'
d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442  artifacts/data/longmemeval/longmemeval_s_cleaned.json
a45b1fe4e2f0c8390b2b2938ac83e92ed5979000856808f3679c07812e9e6dcd  artifacts/data/swebench/verified_test.parquet
b087b5dad72b3e765a6cf93a9e7d516d8796698a0fd358abb73c6627df19f66e  artifacts/data/swebench/verified_test.json
79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4  artifacts/data/locomo/locomo10.json
SUMS
