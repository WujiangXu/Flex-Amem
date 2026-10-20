"""Build the paper tables and the P1-P3 endpoint tests (markdown; LaTeX with --latex) into --out, which is the
run's folder in the results tree (results/README.md), e.g.:
    make_tables.py --lme-runs dev_gate --out results/longmemeval_s/runs/dev_gate/tables
    make_tables.py --diag-runs p0_diag --out results/checkpoint_diag/runs/p0_diag/tables
The endpoint statuses need every run of the fixed sequence: build the combined P1-P3 table from all SWE and
LongMemEval runs, e.g.
    make_tables.py --swe-runs p0_swe_flow,p0_swe_sa,p1_swe_rest385,p0_swe_base,p0_swe_passthrough \
        --lme-runs p0_lme_flow,p0_lme_noevo,p0_lme_amem,p0_lme_readagent,p0_lme_refs --only endpoints --out <dir>/tables
in a tables/ folder of its own: --out is cleared first, so --only refuses a folder that holds other tables.
The Summary+A-MEM split and the Flow arm of the endpoints are frozen per benchmark in configs/exp/endpoints.yaml (while
the split is null, the dev rule picks it from the loaded rows); --summary-split-{swe,lme} / --flow-arm-{swe,lme} override
them.
No LLM calls: reads result rows, judge labels, ledgers and trajectories only."""
import argparse
import sys

from flexamem.analysis.tables import build_all, code_stamp
from flexamem.llm.client import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--swe-runs", default="", help="comma-separated SWE run ids (artifacts/runs/swe/<id>)")
ap.add_argument("--lme-runs", default="", help="comma-separated LongMemEval run ids (artifacts/runs/lme/<id>)")
ap.add_argument("--diag-runs", default="", help="comma-separated checkpoint-diagnostic run ids (artifacts/runs/diag/<id>)")
for _b in ("swe", "lme"):
    ap.add_argument(f"--summary-split-{_b}", type=int, choices=(30, 50, 70),
                    help=f"override the {_b} Summary+A-MEM split of configs/exp/endpoints.yaml")
    ap.add_argument(f"--flow-arm-{_b}", help=f"override the {_b} Flow arm of configs/exp/endpoints.yaml, e.g. flow")
ap.add_argument("--out", required=True, help="output folder relative to the repo root, e.g. results/<bench>/runs/<id>/tables")
ap.add_argument("--latex", action="store_true")
ap.add_argument("--only", default="", help="comma-separated table names to write (e.g. endpoints); default all")
a = ap.parse_args()
stamp = code_stamp()
only = {x for x in a.only.split(",") if x}

out = ROOT / a.out
if out.name != "tables":  # the folder is cleared below; refuse anything that is not a tables/ folder
    ap.error("--out must end in /tables (e.g. results/<benchmark>/runs/<run_id>/tables)")
old = [f for f in [*out.glob("*.md"), *out.glob("*.tex")] if f.name != "README.md"]
if only and (other := sorted({f.stem for f in old} - only)):  # clearing would delete tables this call does not rebuild
    ap.error(f"--only: {a.out} holds other tables ({', '.join(other)}); "
             "write the selected tables to a tables/ folder of their own")
tables = build_all([x for x in a.swe_runs.split(",") if x], [x for x in a.lme_runs.split(",") if x],
                   {"swe": a.summary_split_swe, "lme": a.summary_split_lme},
                   {"swe": a.flow_arm_swe, "lme": a.flow_arm_lme}, [x for x in a.diag_runs.split(",") if x])
if unknown := sorted(only - {t.name for t in tables}):
    ap.error(f"--only: unknown table(s) {', '.join(unknown)}; choose from {', '.join(t.name for t in tables)}")
out.mkdir(parents=True, exist_ok=True)
for f in [*out.glob("*.md"), *out.glob("*.tex")]:  # a rebuild replaces the whole folder, so no stale table survives
    f.unlink()
index = [f"# Tables for {a.out}", "", f"Generated at code {stamp} by", "", "```",
         "python scripts/make_tables.py " + " ".join(sys.argv[1:]), "```", "",
         "Do not edit by hand; rerun the command. Tables with no rows (or no evaluable endpoint) for these runs are not written.", ""]
for t in tables:
    if (only and t.name not in only) or not t.rows or all(r[-1] == "not evaluable" for r in t.rows):
        continue
    (out / f"{t.name}.md").write_text(t.md())
    if a.latex:
        (out / f"{t.name}.tex").write_text(t.tex())
    index.append(f"- [{t.title}]({t.name}.md)")
    print(t.md())
(out / "README.md").write_text("\n".join(index) + "\n")
if only and not index[-1].startswith("- "):
    print(f"warning: none of --only {a.only} has rows for these runs; no table written", file=sys.stderr)
print(f"wrote {out}")
