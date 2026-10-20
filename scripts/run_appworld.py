"""Run AppWorld arms on Test-Normal (decision 2026-10-07_appworld), Test-Challenge, or the pre-registered confirmation
set train,dev (docs/protocol/confirmation_appworld.md; only as frozen in configs/splits/manifest_confirm.json). One
example first:
    run_appworld.py --run-id oneex_aw_<date> --ids <task_id> --methods flow_noevo_zoom --budgets 4000
Needs the AppWorld venv and data under artifacts/appworld (docs/benchmarks/appworld.md)."""
import argparse
import hashlib
import json


from flexamem.bench.appworld.runner import AGENT, ARMS, REFERENCES, load_tasks, run, run_dir_for
from flexamem.bench.swe.runner import MEMORY
from flexamem.bench.common import Results, write_manifest
from flexamem.llm.client import ROOT, get_llm, load_model_config
from flexamem.llm.ledger import set_ledger

ap = argparse.ArgumentParser()
ap.add_argument("--run-id", required=True)
ap.add_argument("--split", default="test_normal", help="test_normal | test_challenge (the 417 harder tasks) | "
                "train,dev (the confirmation set)")
ap.add_argument("--ids", default="", help="comma-separated task ids (overrides --split's full list)")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--methods", default=",".join(ARMS + REFERENCES))
ap.add_argument("--budgets", default="4000,6000", help="ignored for passthrough (always uncapped)")
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--step-limit", type=int, default=AGENT["step_limit"], help="60 registered; higher only for the "
                "step-limit sensitivity run")
ap.add_argument("--variant", type=int, default=0, help="1-3: only that variant of each scenario (task ids end in _<n>)")
ap.add_argument("--shard", default="0/1", help="i/n: every n-th task from the i-th, so n processes can share one run id "
                "(each skips the rows already in results.jsonl; appends are locked)")
ap.add_argument("--allow-dirty", action="store_true")
ap.add_argument("--allow-mixed", action="store_true", help="resume over rows written by another code version")
a = ap.parse_args()

methods = a.methods.split(",")
assert all(m in MEMORY or m in REFERENCES for m in methods), methods
budgets = [int(x) for x in a.budgets.split(",")]
splits = a.split.split(",")
assert splits in (["test_normal"], ["test_challenge"], ["train", "dev"]), a.split
full = [t for s in splits for t in load_tasks(s)]
if splits == ["train", "dev"]:  # the confirmation set: the frozen list or nothing
    frozen = json.loads((ROOT / "configs/splits/manifest_confirm.json").read_text())["files"]["appworld_confirm.txt"]
    listed = (ROOT / "configs/splits/appworld_confirm.txt").read_bytes()
    if hashlib.sha256(listed).hexdigest() != frozen["sha256"] or full != listed.decode().split():
        raise SystemExit("train + dev differ from the frozen confirmation list")
tasks = [t for t in full if t in set(a.ids.split(","))] if a.ids else full
shard, n_shards = map(int, a.shard.split("/"))
assert 0 <= shard < n_shards, a.shard
if a.variant:
    tasks = [t for t in tasks if t.endswith(f"_{a.variant}")]
tasks = tasks[shard::n_shards]
assert tasks, "no tasks"
if a.limit:
    tasks = tasks[:a.limit]
jobs = [(t, m, b) for t in tasks for m in methods for b in ([0] if m == "passthrough" else budgets)]
run_dir = run_dir_for(a.run_id)
results = Results(run_dir / "results.jsonl", ("task_id", "method", "budget"), a.allow_mixed)
results.require("agent_model", get_llm().role("agent")["model"])
results.require("memory_model", get_llm().role("memory")["model"],  # rows from before the field: the default writer
                missing=load_model_config()["roles"]["memory"]["model"])
results.require("step_limit", a.step_limit)
write_manifest(run_dir, vars(a), a.allow_dirty)
set_ledger(ROOT / f"artifacts/ledger/appworld_{a.run_id}.jsonl")
errors = run(jobs, results, run_dir, a.run_id, a.workers, a.step_limit)
print(f"done: {len(jobs) - len(errors)} ok, {len(errors)} infra failures (rerun on resume) {errors}")
