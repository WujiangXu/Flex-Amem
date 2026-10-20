"""Run SWE-bench Verified arms. One example first:
    run_swe.py --run-id oneex_swe_<date> --ids django__django-11095 --methods flow --budgets 8000 --allow-dirty
Requires a running Docker daemon and the instance images (scripts/pull_swe_images.sh)."""
import argparse

from flexamem.bench.common import Results, write_manifest
from flexamem.bench.swe.grade import docker_client
from flexamem.bench.swe.runner import EXEC, MEMORY, REFERENCES, load_instances, regrade, run, run_dir_for
from flexamem.llm.client import ROOT, get_llm
from flexamem.llm.ledger import set_ledger

ap = argparse.ArgumentParser()
ap.add_argument("--run-id", required=True)
ap.add_argument("--split", choices=["dev20", "eval100", "eval95", "rest385", "all"], default="dev20",
                help="eval95: Eval-100 without gold-patch failures; rest385: the other evaluable issues (P1)")
ap.add_argument("--ids", default="", help="comma-separated instance ids (overrides --split)")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--methods", default="flow")
ap.add_argument("--budgets", default="8000", help="ignored for passthrough (always uncapped)")
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--snapshots", action="store_true", help="commit checkpoint containers in the passthrough arm")
ap.add_argument("--no-egress", action="store_true", help="start agent containers with --network=none")
ap.add_argument("--allow-dirty", action="store_true")
ap.add_argument("--allow-mixed", action="store_true", help="resume over rows written by another code version")
ap.add_argument("--regrade", action="store_true", help="only re-grade rows whose grading had no verdict")
ap.add_argument("--reserve", type=float, default=0.95, help="cap = reserve x B (0.95 registered; lower only for the "
                "API-over-B sensitivity rerun)")
a = ap.parse_args()
docker_client()  # fail fast if the Docker daemon is down, before any episode is spent

if a.reserve != 0.95:  # the API-over-B sensitivity rerun: a tighter cap for every policy of this run
    import flexamem.adapters.mswea as MSW
    MSW.RESERVE = a.reserve
methods = a.methods.split(",")
assert all(m in MEMORY or m in REFERENCES for m in methods), methods
budgets = [int(x) for x in a.budgets.split(",")]
if a.ids:
    ids = a.ids.split(",")
elif a.split == "all":
    ids = None
else:
    ids = (ROOT / f"configs/splits/swe_verified_{a.split}.txt").read_text().split()
instances = load_instances(ids)
if a.limit:
    instances = instances[:a.limit]
jobs = [(inst, m, b) for inst in instances for m in methods for b in ([0] if m == "passthrough" else budgets)]
run_dir = run_dir_for(a.run_id)
results = Results(run_dir / "results.jsonl", ("iid", "method", "budget"), a.allow_mixed)
results.require("agent_model", get_llm().role("agent")["model"])
write_manifest(run_dir, vars(a), a.allow_dirty, exec_backend=EXEC)  # a regrade gets its own manifest (args.regrade)
if a.regrade:
    print(f"regraded {regrade(results, {i['instance_id']: i for i in load_instances()}, run_dir, a.run_id, a.workers)}")
    raise SystemExit
set_ledger(ROOT / f"artifacts/ledger/swe_{a.run_id}.jsonl")
errors = run(jobs, results, run_dir, a.run_id, a.workers, a.snapshots, egress=not a.no_egress)
print(f"done: {len(jobs) - len(errors)} ok, {len(errors)} infra failures (rerun on resume) {errors}")
