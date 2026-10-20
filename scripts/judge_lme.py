"""Judge LongMemEval answers with both judges and print accuracy by method x budget x type.
--relabel (offline, no API call): set every judged label to the cached verdict of its judge prompt, so identical
answers carry one verdict. Before the client's single-flight cache, concurrent judge calls on the same prompt could
each keep their own reply (journal: "Concurrent identical judge calls kept different verdicts")."""
import argparse
import collections

from flexamem.bench.common import Results, read_rows
from flexamem.bench.longmemeval import data
from flexamem.bench.longmemeval.judge import judge, judge_prompt, verdict
from flexamem.bench.longmemeval.runner import results_path
from flexamem.llm.client import ROOT, get_llm
from flexamem.llm.ledger import ledger_context, pmap, set_ledger

ap = argparse.ArgumentParser()
ap.add_argument("--run-id", required=True)
ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--relabel", action="store_true")
ap.add_argument("--allow-mixed", action="store_true", help="judge new rows next to rows judged at another code version")
a = ap.parse_args()
src = results_path(a.run_id)
out = Results(src.parent / "judged.jsonl", ("qid", "method", "budget"), allow_mixed=a.relabel or a.allow_mixed)
set_ledger(ROOT / f"artifacts/ledger/lme_{a.run_id}_judge.jsonl")
rows = read_rows(src, ("qid", "method", "budget"))
gold = {it["qid"]: it for it in data.load(sorted({r["qid"] for r in rows}))}


def one(r):
    if out.has(**r):
        return
    it = gold[r["qid"]]
    if r.get("error"):  # the method produced no answer: wrong, without asking the judges
        out.add({"qid": r["qid"], "qtype": r["qtype"], "method": r["method"], "budget": r["budget"],
                 "abstention": r["abstention"], "j1": False, "j2": False, "error": r["error"]})
        return
    with ledger_context(stage="judge", method=r["method"], budget=r["budget"], qid=r["qid"], run_id=a.run_id):
        j1 = judge("judge", it["qtype"], it["question"], it["gold"]["answer"], r["answer"], it["gold"]["abstention"])
        j2 = judge("judge2", it["qtype"], it["question"], it["gold"]["answer"], r["answer"], it["gold"]["abstention"])
    out.add({"qid": r["qid"], "qtype": r["qtype"], "method": r["method"], "budget": r["budget"],
             "abstention": r["abstention"], "j1": j1["label"], "j2": j2["label"], "j1_raw": j1["raw"], "j2_raw": j2["raw"]})


def relabel():
    answer = {(r["qid"], r["method"], r["budget"]): r.get("answer") for r in rows}
    n = 0
    for r in out.rows():
        if r.get("error"):
            continue
        it = gold[r["qid"]]
        p = judge_prompt(it["qtype"], it["question"], it["gold"]["answer"], answer[(r["qid"], r["method"], r["budget"])],
                         it["gold"]["abstention"])
        new = dict(r)
        for j, role in (("j1", "judge"), ("j2", "judge2")):
            raw = get_llm().cached(role, p)
            if raw is None:
                raise SystemExit(f"no cached {role} verdict for {r['qid']} {r['method']} {r['budget']}")
            if verdict(raw) != r[j]:
                new[j], new[f"{j}_raw"] = verdict(raw), raw
        if new != r:
            new["relabel_sha"] = out.git_sha
            print("relabel", r["qid"], r["method"], r["budget"], {j: (r[j], new[j]) for j in ("j1", "j2")})
            out.add(new)
            n += 1
    print(f"{n} rows relabelled")


if a.relabel:
    relabel()
else:
    pmap(one, rows, a.workers)
judged = out.rows()
acc = collections.defaultdict(list)
for r in judged:
    acc[(r["method"], r["budget"], "all")].append(r)
    acc[(r["method"], r["budget"], r["qtype"])].append(r)
print(f"{'method':22s} {'B':>6s} {'type':26s} {'n':>4s} {'j1':>6s} {'j2':>6s} {'agree':>6s}")
for k in sorted(acc):
    v = acc[k]
    n = len(v)
    print(f"{k[0]:22s} {k[1]:>6d} {k[2]:26s} {n:>4d} {sum(x['j1'] for x in v) / n:6.3f} "
          f"{sum(x['j2'] for x in v) / n:6.3f} {sum(x['j1'] == x['j2'] for x in v) / n:6.3f}")
