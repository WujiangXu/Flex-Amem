"""LoCoMo tables for one or more runs (src/flexamem/analysis/locomo.py), after scripts/judge_locomo.py:
    locomo_tables.py --runs p_locomo --out results/locomo/runs/p_locomo/tables"""
import argparse
import sys

from flexamem.analysis.locomo import OURS, contrasts, load, main_table
from flexamem.analysis.tables import code_stamp
from flexamem.llm.client import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True, help="comma-separated run ids")
ap.add_argument("--out", required=True)
ap.add_argument("--ours", default=OURS, help="the arm the contrasts compare every other arm against")
ap.add_argument("--exclude", default="", help="comma-separated conversations left out (e.g. a method's tuning conversation)")
a = ap.parse_args()
out = ROOT / a.out
assert out.name == "tables", "--out must end in /tables"
rows = load(a.runs.split(","))
if a.exclude:
    rows = [r for r in rows if r["cid"] not in set(a.exclude.split(","))]
ts = [main_table(rows), contrasts(rows, a.ours)]
out.mkdir(parents=True, exist_ok=True)
for f in out.glob("*.md"):
    f.unlink()
index = [f"# Tables for {a.out}", "", f"Generated at code {code_stamp()} by", "", "```",
         "python scripts/locomo_tables.py " + " ".join(sys.argv[1:]), "```", "", "Do not edit by hand; rerun the command.", ""]
for t in ts:
    (out / f"{t.name}.md").write_text(t.md())
    index.append(f"- [{t.title}]({t.name}.md)")
    print(t.md())
(out / "README.md").write_text("\n".join(index) + "\n")
