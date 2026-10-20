"""LoCoMo tables (scripts/locomo_tables.py): J and F1 by arm and budget, per category, tokens, and paired contrasts.

Rows are results.jsonl joined with judged.jsonl (scripts/judge_locomo.py) on (qid, method, budget); a key in two runs is
an error. Cost per arm:
- reader prompt tokens per question (our o200k count of the reader prompt, which B bounds);
- build tokens per conversation: the billed tokens of the arm's builds. A LongMemEval build kind's rows are found by
  its build id in the ledger its build file names, since a build is reused across runs. An online method's rows (AFM)
  are its run ledger's build rows for that conversation;
- render tokens per question: billed tokens of model calls made while rendering (merges, summary compression, AFM's
  compressions), from the run ledger.
Contrasts pair FlexA-MEM (unit recovery) with each arm on the same questions: exact McNemar on J, and a bootstrap CI
that resamples whole conversations (10 clusters, so the CIs are rough)."""
import gzip
import json
from collections import defaultdict
from functools import lru_cache
from statistics import mean

from flexamem.analysis import stats as S
from flexamem.analysis.cost import LEDGER, read_ledger, tokens
from flexamem.analysis.tables import Table, ci, ktok, label, pct
from flexamem.bench.common import read_rows
from flexamem.bench.locomo.data import CATEGORIES
from flexamem.bench.locomo.runner import BUILD_ROOT, ONLINE, results_path
from flexamem.bench.longmemeval.runner import METHODS

KEY = ("qid", "method", "budget")
OURS = "flow_s30_ld50_noevo_zoom"
ORDER = (OURS, "flow_s30_ld50_noevo_noexp", "flow_s30_ld50_noevo_bm25", "rag_pieces", "amem", "amem_zoom",
         "summary_amem_s50", "afm", "truncation", "full")


def load(runs) -> list[dict]:
    rows = {}
    for rid in runs:
        src = results_path(rid)
        judged = {tuple(r[k] for k in KEY): r for r in read_rows(src.parent / "judged.jsonl", KEY)} \
            if (src.parent / "judged.jsonl").exists() else {}
        for r in read_rows(src, KEY):
            k = tuple(r[f] for f in KEY)
            assert k not in rows, f"{k} in two runs"
            j = judged.get(k)
            rows[k] = {**r, "j1": j["j1"] if j else None, "f1": j["f1"] if j else None}
    return list(rows.values())


def arms(rows):
    order = {m: i for i, m in enumerate(ORDER)}
    keys = sorted({(r["method"], r["budget"]) for r in rows}, key=lambda k: (k[1] == 0, k[1], order.get(k[0], 99), k[0]))
    return [((m, b), [r for r in rows if (r["method"], r["budget"]) == (m, b)]) for m, b in keys]


@lru_cache(maxsize=None)
def _ledger(path: str) -> tuple:
    return tuple(read_ledger(path))


@lru_cache(maxsize=None)
def _build_file(version: str, kind: str, cid: str) -> dict:
    with gzip.open(BUILD_ROOT / version / kind / f"{cid}.json.gz", "rt") as f:
        d = json.load(f)
    return d.get("extra", d)  # NoteStore files keep build_id and ledger in "extra", summary files at the top


@lru_cache(maxsize=None)
def _by_build(path: str) -> dict:
    """A ledger's billed tokens summed per build id, (stage == build rows) per (build id, conversation) too: one pass."""
    out = defaultdict(int)
    for x in _ledger(path):
        if x.get("build_id"):
            out[x["build_id"]] += tokens(x)
            if x.get("stage") == "build":
                out[(x["build_id"], x.get("qid"))] += tokens(x)
    return out


def build_tokens(r: dict) -> int:
    """Billed tokens of the builds row r's arm read, for its conversation."""
    if r["method"] in ONLINE or r["method"] == "afm":  # the build the row's arm read (a resume rebuilds)
        bid = r.get("build_ids", {}).get(r["method"])
        return _by_build(str(LEDGER / f"locomo_{r['run_id']}.jsonl")).get((bid, r["cid"]), 0)
    total = 0
    for kind in METHODS.get(r["method"], ((),))[0]:
        if kind not in r.get("build_ids", {}):  # the build failed: no file and no cost recorded for it
            continue
        meta = _build_file(r["build_version"], kind, r["cid"])
        total += _by_build(meta["ledger"]).get(meta["build_id"], 0)
    return total


def render_tokens(rows) -> dict:
    """(method, budget) -> billed render-time tokens per question, from the rows' run ledgers."""
    spent, att = defaultdict(int), {r["attempt"] for r in rows}
    for rid in {r["run_id"] for r in rows}:
        for x in _ledger(str(LEDGER / f"locomo_{rid}.jsonl")):
            if x.get("stage") == "render" and x.get("attempt") in att:
                spent[(x["method"], x["budget"])] += tokens(x)
    return spent


def main_table(rows) -> Table:
    qs = {(m, b): {r["qid"] for r in rs} for (m, b), rs in arms(rows)}
    assert len({frozenset(v) for v in qs.values()}) <= 1, "arms answered different question sets (unresolved failures?)"
    rend, out = render_tokens(rows), []
    for (m, b), rs in arms(rows):
        jr = [r for r in rs if r["j1"] is not None]
        k = sum(r["j1"] for r in jr)
        per = {c: [r["j1"] for r in jr if r["category"] == c] for c in CATEGORIES}
        conv = {r["cid"]: build_tokens(r) for r in rs}
        out.append([label(m), f"{b // 1000}k" if b else "uncapped", len(rs), ci(*S.wilson(k, len(jr))) if jr else "–",
                    *[pct(mean(per[c])) if per[c] else "–" for c in CATEGORIES],
                    pct(mean(r["f1"] for r in jr)) if jr else "–",
                    f"{mean(r['prompt_tokens'] for r in rs if r.get('prompt_tokens')):.0f}"
                    if any(r.get("prompt_tokens") for r in rs) else "–",
                    ktok(mean(conv.values())) if any(conv.values()) else "0", f"{rend[(m, b)] / len(rs):.0f}",
                    sum(bool(r.get("error")) for r in rs)])
    return Table("locomo_main", "LoCoMo (categories 1-4) by arm and budget",
                 ["Method", "B", "n", "J % [95% CI]", *[f"{n} %" for n in CATEGORIES.values()], "F1 %",
                  "reader prompt tokens", "build tokens / conversation", "render tokens / question", "method errors"],
                 out,
                 ["J: the primary judge with the judge template of bench/locomo/prompts.py (the run card names it); "
                  "Wilson interval. F1: LoCoMo's token F1 on the last line of the answer, without stemming "
                  "(bench/locomo/judge.py). Unjudged rows count in n but not in J or F1.",
                  "Build tokens are billed (prompt + completion) over the arm's builds of one conversation; render "
                  "tokens are model calls made while rendering, per question."])


def contrasts(rows, ours: str = OURS) -> Table:
    by = {(r["qid"], r["method"], r["budget"]): r for r in rows if r["j1"] is not None}
    out = []
    for (m, b), rs in arms(rows):
        if m == ours:
            continue
        for bo in sorted({r["budget"] for r in rows if r["method"] == ours}):
            if b and b != bo:
                continue
            pairs = [(by[(r["qid"], ours, bo)], r) for r in rs
                     if r["j1"] is not None and (r["qid"], ours, bo) in by]
            if not pairs:
                continue
            a, o = [p[0]["j1"] for p in pairs], [p[1]["j1"] for p in pairs]
            t = S.mcnemar_exact(a, o)
            bs = S.paired_bootstrap([x - y for x, y in zip(a, o)], [p[0]["cid"] for p in pairs])
            out.append([f"{label(ours)} vs {label(m)}", f"{bo // 1000}k" + ("" if b else " vs uncapped"), len(pairs),
                        pct(mean(a)), pct(mean(o)),
                        f"{100 * bs['est']:+.1f} [{100 * bs['lo']:+.1f}, {100 * bs['hi']:+.1f}]",
                        f"{t['n10']}/{t['n01']}", f"{t['p']:.3g}"])
    return Table("locomo_contrasts", f"LoCoMo: {label(ours)} against each arm on the same questions (J)",
                 ["Comparison", "B", "n pairs", "first %", "other %", "Δ (pts) [95% CI, by conversation]",
                  "wins/losses", "McNemar p"], out,
                 ["Exact McNemar on paired J; the CI resamples whole conversations (10 clusters; 10k, seeded). Not "
                  "corrected for multiplicity."])
