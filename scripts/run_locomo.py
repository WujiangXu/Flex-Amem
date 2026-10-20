"""Run LoCoMo arms (decision 2026-10-11_locomo_afm; docs/benchmarks/locomo.md). One example first:
    run_locomo.py --run-id oneex_locomo --conversations conv-26 --limit 3 \
        --methods flow_s30_ld50_noevo_zoom,rag_pieces --budgets 4000"""
import argparse

from flexamem.bench.common import Results, write_manifest
from flexamem.bench.locomo import data
from flexamem.bench.locomo.runner import ARMS, REFERENCES, known, results_path, run
from flexamem.llm.client import ROOT, get_llm
from flexamem.llm.ledger import set_ledger

ap = argparse.ArgumentParser()
ap.add_argument("--run-id", required=True)
ap.add_argument("--conversations", default="", help="comma-separated sample ids (default: all 10)")
ap.add_argument("--limit", type=int, default=0, help="first n questions of each conversation (one example)")
ap.add_argument("--methods", default=",".join(ARMS + REFERENCES))
ap.add_argument("--budgets", default="2000,4000,8000", help="ignored for full (always uncapped)")
ap.add_argument("--workers", type=int, default=32, help="parallel render-and-read jobs")
ap.add_argument("--allow-dirty", action="store_true")
ap.add_argument("--allow-mixed", action="store_true", help="resume over rows written by another code version")
a = ap.parse_args()

methods = a.methods.split(",")
assert all(known(m) for m in methods), [m for m in methods if not known(m)]
budgets = [int(x) for x in a.budgets.split(",")]
convs = data.load(a.conversations.split(",") if a.conversations else None)
assert convs, "no conversations"
if a.limit:
    for c in convs:
        c["questions"] = c["questions"][:a.limit]
path = results_path(a.run_id)
write_manifest(path.parent, vars(a), a.allow_dirty)
set_ledger(ROOT / f"artifacts/ledger/locomo_{a.run_id}.jsonl")
results = Results(path, ("qid", "method", "budget"), a.allow_mixed)
results.require("reader_model", get_llm().role("reader")["model"])
results.require("memory_model", get_llm().role("memory")["model"])
errors = run(convs, methods, budgets, results, a.run_id, a.workers)
print(f"done: {len(errors)} failed (API/disk; rerun to resume) {errors[:20]}")
