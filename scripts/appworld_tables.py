"""AppWorld tables for one or more runs (src/flexamem/analysis/appworld.py):
    appworld_tables.py --runs p_aw_testnormal --out results/appworld/runs/p_aw_testnormal/tables"""
import argparse
import sys

from flexamem.analysis.appworld import contrasts, load, main_table, render_mix
from flexamem.analysis.tables import code_stamp
from flexamem.llm.client import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True, help="comma-separated run ids")
ap.add_argument("--out", required=True)
ap.add_argument("--ours", default="flow_noevo_zoom", help="the arm the contrasts compare every other arm against")
ap.add_argument("--render-mix", action="store_true", help="add what the memory renders showed (the render controls)")
a = ap.parse_args()
out = ROOT / a.out
assert out.name == "tables", "--out must end in /tables"
rows = load(a.runs.split(","))
ts = [main_table(rows), contrasts(rows, a.ours)] + ([render_mix(rows)] if a.render_mix else [])
out.mkdir(parents=True, exist_ok=True)
for f in out.glob("*.md"):
    f.unlink()
index = [f"# Tables for {a.out}", "", f"Generated at code {code_stamp()} by", "", "```",
         "python scripts/appworld_tables.py " + " ".join(sys.argv[1:]), "```", "", "Do not edit by hand; rerun the command.", ""]
for t in ts:
    (out / f"{t.name}.md").write_text(t.md())
    index.append(f"- [{t.title}]({t.name}.md)")
    print(t.md())
(out / "README.md").write_text("\n".join(index) + "\n")
