"""Run LongMemEval-S methods. One example first:  run_lme.py --run-id oneex_lme_<date> --limit 1 --allow-dirty"""
import argparse

from flexamem.bench.common import Results, write_manifest
from flexamem.bench.longmemeval import data
from flexamem.bench.longmemeval.runner import METHODS, ONLINE, REFERENCES, RETRIEVAL, results_path, run
from flexamem.llm.client import ROOT, get_llm
from flexamem.llm.ledger import set_ledger

ap = argparse.ArgumentParser()
ap.add_argument("--run-id", required=True)
ap.add_argument("--split", choices=["dev", "all", "heldout"], default="dev")
ap.add_argument("--ids", default="", help="comma-separated question ids (overrides --split)")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--methods", default="flow,amem,summary_amem_s50,readagent,truncation")
ap.add_argument("--budgets", default="8000,16000,32000")
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--seed", type=int, default=0, help="reader seed; 0 shares the reader cache with earlier runs")
ap.add_argument("--builds", default="", help="read the builds of this version (artifacts/lme_builds/<version>) and "
                "build nothing, instead of this code's version")
ap.add_argument("--allow-dirty", action="store_true")
ap.add_argument("--allow-mixed", action="store_true", help="resume over rows written by another code version")
a = ap.parse_args()

methods = a.methods.split(",")
assert all(m in METHODS or m in REFERENCES or m in RETRIEVAL or m in ONLINE for m in methods), methods
budgets = [int(x) for x in a.budgets.split(",")]
if a.ids:
    items = data.load(a.ids.split(","))
elif a.split == "dev":
    items = data.load(data.dev_ids())
else:
    items = data.load()
    if a.split == "heldout":
        dev = set(data.dev_ids())
        items = [it for it in items if it["qid"] not in dev]
if a.limit:
    items = items[:a.limit]
path = results_path(a.run_id)
write_manifest(path.parent, vars(a), a.allow_dirty)
set_ledger(ROOT / f"artifacts/ledger/lme_{a.run_id}.jsonl")
results = Results(path, ("qid", "method", "budget"), a.allow_mixed)
results.require("reader_model", get_llm().role("reader")["model"])
errors = run(items, methods, budgets, results, a.run_id,
             a.workers, a.seed, a.builds or None)
print(f"done: {len(items) - len(errors)} ok, {len(errors)} failed (API/disk; rerun to resume) {errors}")
