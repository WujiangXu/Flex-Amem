"""Judge LoCoMo answers (bench/locomo/judge.py) and print J and F1 by method x budget x category.

J uses the judge template of bench/locomo/prompts.py (JUDGE with per-category overrides; LongMemEval's prompts until
replaced: the temporal-reasoning prompt for category 2, the default prompt otherwise) with the primary judge (role
"judge_locomo": the primary judge, Claude Sonnet 4.6 at temperature 0, with room for Mem0's one-sentence reason and
JSON label; LongMemEval's "judge" role caps output at 10 tokens for its yes/no prompt). --second adds the second judge (role "judge2") and their agreement. F1 is LoCoMo's token
F1 on the last line of the answer, computed without an API call. Labels go to judged.jsonl next to results.jsonl.
    judge_locomo.py --run-id <run_id> [--second]"""
import argparse
import collections

from flexamem.bench.common import Results, read_rows
from flexamem.bench.locomo import data
from flexamem.bench.locomo.judge import f1, judge
from flexamem.bench.locomo.runner import results_path
from flexamem.llm.client import ROOT, get_llm
from flexamem.llm.ledger import ledger_context, pmap, set_ledger

KEY = ("qid", "method", "budget")
ap = argparse.ArgumentParser()
ap.add_argument("--run-id", required=True)
ap.add_argument("--workers", type=int, default=32)
ap.add_argument("--second", action="store_true", help="also ask the second judge")
ap.add_argument("--allow-mixed", action="store_true", help="judge new rows next to rows judged at another code version")
a = ap.parse_args()
src = results_path(a.run_id)
out = Results(src.parent / "judged.jsonl", KEY, allow_mixed=a.allow_mixed)
JUDGES = {"judge_model": get_llm().role("judge_locomo")["model"],
          **({"judge2_model": get_llm().role("judge2")["model"]} if a.second else {})}
for k, v in JUDGES.items():  # a resume under another judge is refused
    out.require(k, v)
set_ledger(ROOT / f"artifacts/ledger/locomo_{a.run_id}_judge.jsonl")
rows = read_rows(src, KEY)
convs = data.load(sorted({r["cid"] for r in rows}))
question = {q["qid"]: q for c in convs for q in c["questions"]}
gold = {qid: g for c in convs for qid, g in c["gold"].items()}
judged = {tuple(x[k] for k in KEY): x for x in out.rows()}


def one(r):
    done = judged.get(tuple(r[k] for k in KEY))
    if done is not None and done.get("attempt") == r.get("attempt") and (not a.second or "j2" in done):
        return
    q, g = question[r["qid"]], gold[r["qid"]]
    row = {**{k: r[k] for k in ("qid", "cid", "category", "qtype", "method", "budget")}, **JUDGES,
           "attempt": r.get("attempt")}  # a rewritten result row (rerun) is judged again
    if r.get("error"):  # the method produced no answer: wrong, without asking the judges
        out.add({**row, "j1": False, **({"j2": False} if a.second else {}), "f1": 0.0, "error": r["error"]})
        return
    with ledger_context(stage="judge", method=r["method"], budget=r["budget"], qid=r["qid"], run_id=a.run_id):
        j1 = judge("judge_locomo", q, g["answer"], r["answer"])
        j2 = judge("judge2", q, g["answer"], r["answer"]) if a.second else None
    out.add({**row, "j1": j1["label"], "j1_raw": j1["raw"], "f1": f1(r["answer"], g["answer"], q["category"]),
             **({"j2": j2["label"], "j2_raw": j2["raw"]} if j2 else {})})


pmap(one, rows, a.workers)
acc = collections.defaultdict(list)
for r in out.rows():
    acc[(r["method"], r["budget"], "all")].append(r)
    acc[(r["method"], r["budget"], r["qtype"])].append(r)
print(f"{'method':28s} {'B':>6s} {'category':12s} {'n':>5s} {'J':>6s} {'F1':>6s}" + (f" {'J2':>6s} {'agree':>6s}" if a.second else ""))
for k in sorted(acc):
    v = acc[k]
    n = len(v)
    line = f"{k[0]:28s} {k[1]:>6d} {k[2]:12s} {n:>5d} {sum(x['j1'] for x in v) / n:6.3f} {sum(x['f1'] for x in v) / n:6.3f}"
    if a.second:
        line += f" {sum(x.get('j2', False) for x in v) / n:6.3f} {sum(x['j1'] == x.get('j2') for x in v) / n:6.3f}"
    print(line)
