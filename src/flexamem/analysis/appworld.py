"""AppWorld tables (decision 2026-10-07_appworld). Per arm and budget:
- task goal completion (TGC): the share of tasks whose unit tests all pass;
- scenario goal completion (SGC): the share of scenarios whose three variants all succeed, AppWorld's two metrics;
- the share of tests passed;
- steps, compacted steps and billed tokens per episode (ledger, by attempt; agent and memory calls);
- how often a compacted step's API-billed prompt exceeded B: the cap holds on our o200k count, and the provider's
  tokenizer counts more (design deviation 4; trajectories' context logs).
Paired contrasts of our method (A-MEM Zoom) against every arm on the same tasks: exact McNemar on success and a 95%
bootstrap CI of the difference that resamples whole scenarios (the three variants share a world and a solution)."""
import json
from collections import defaultdict
from statistics import mean

from flexamem.analysis import stats as S
from flexamem.analysis.cost import LEDGER, read_ledger, tokens
from flexamem.analysis.tables import Table, ci, ktok, label, pct
from flexamem.bench.appworld.env import APPWORLD_ROOT
from flexamem.bench.common import read_rows
from flexamem.llm.client import ROOT

RUNS = ROOT / "artifacts/runs/appworld"
KEY = ("task_id", "method", "budget")
OURS = "flow_noevo_zoom"


def load(runs) -> list[dict]:
    """Result rows of the runs (a key in two runs is an error), each with its episode's billed tokens."""
    rows, spent = {}, defaultdict(lambda: defaultdict(int))
    for rid in runs:
        for r in read_ledger(LEDGER / f"appworld_{rid}.jsonl"):
            if r.get("attempt"):
                spent[r["attempt"]]["agent" if r["phase"] == "agent" else "memory"] += tokens(r)
        for r in read_rows(RUNS / rid / "results.jsonl", KEY):
            k = tuple(r[f] for f in KEY)
            assert k not in rows, f"{k} in two runs"
            rows[k] = {**r, "tok_agent": spent[r["attempt"]]["agent"], "tok_memory": spent[r["attempt"]]["memory"],
                       **api_over_b(r)}
    return list(rows.values())


SPLITS = {"test_normal": "Test-Normal", "test_challenge": "Test-Challenge", "train": "Train", "dev": "Dev"}


def split_name(rows) -> str:
    """The AppWorld split(s) the rows' tasks come from, by AppWorld's own lists (Train + Dev: the confirmation set)."""
    have = {r["task_id"] for r in rows}
    names = [n for s, n in SPLITS.items()
             if have & set((APPWORLD_ROOT / "data/datasets" / f"{s}.txt").read_text().split())]
    return " + ".join(names) or "AppWorld"


def render_mix(rows) -> Table:
    """What the renders of the memory arms showed, from the episodes' context logs: per compacted step with a memory
    render, raw and excerpted units, render tokens; per episode, merges; for the after-timeline control, the share of
    renders whose header changed the selection (deviation 39); for fixed-length recovery, its demand against the raw
    demand."""
    out = []
    for (m, b), rs in arms(rows):
        if not b or not m.startswith(("flow_noevo_zoom", "amem", "afm")):
            continue
        rend, merges = [], []
        for r in rs:
            log = json.loads((ROOT / r["traj"]).read_text())["info"]["context"]["log"]
            rend += [x["render"] for x in log if x["mode"] != "pass" and x.get("render")]
            merges.append(r["memory"].get("merges", 0))
        if not rend:
            continue
        tip = [x["after_selection_differs"] for x in rend if x.get("after_selection_differs") is not None]
        dem = [(x["demand"], x["demand_raw"]) for x in rend if "demand" in x]
        out.append([label(m), f"{b // 1000}k", len(rend),
                    f"{mean(x['full'] if 'full' in x else len(x.get('expanded', x.get('retrieved', []))) for x in rend):.2f}",
                    f"{mean(x['compressed'] if 'compressed' in x else len(x.get('excerpted', [])) for x in rend):.2f}", f"{mean(x.get('tokens', 0) for x in rend):.0f}",
                    f"{mean(merges):.1f}", pct(mean(tip)) if tip else "–",
                    f"{mean(d for d, _ in dem):.0f} / {mean(d for _, d in dem):.0f}" if dem else "–",
                    f"{mean(x['stub'] for x in rend):.2f} / {mean(x['dropped'] for x in rend):.2f}" if "stub" in rend[0] else "–"])
    return Table("appworld_render_mix", f"AppWorld {split_name(rows)}: what the memory renders showed",
                 ["Method", "B", "renders", "raw units / render", "excerpts / render", "render tokens",
                  "merges / episode", "header changed the selection %", "demand / raw demand (tokens)",
                  "stubs / dropped units per render"], out,
                 ["From the context log of every compacted step with a memory render. Merges are the episode's "
                  "permanent merges at its end. A-MEM arms: raw units are the retrieved notes shown whole. AFM: raw units are "
                  "its full tier, excerpts its compressed tier.",
                  "A-MEM + fit-or-degrade ranks excerpt pieces against the newest unit on AppWorld (as FlexA-MEM does "
                  "there) and against the question on LongMemEval."])


def api_over_b(r: dict) -> dict:
    """Compacted steps of the episode, those whose API-billed prompt exceeded B, and the largest billed prompt / B."""
    if not r["budget"]:
        return {"n_comp_api": 0, "n_over_b": 0, "max_api_b": None}
    log = json.loads((ROOT / r["traj"]).read_text())["info"]["context"]["log"]
    api = [x["api_prompt_tokens"] for x in log if x["mode"] != "pass" and x.get("api_prompt_tokens")]
    return {"n_comp_api": len(api), "n_over_b": sum(a > r["budget"] for a in api),
            "max_api_b": max(api) / r["budget"] if api else None}


def arms(rows):
    order = {m: i for i, m in enumerate((OURS, "flow_noevo_zoom_noexp", "flow_noevo_zoom_nolinks",
                                         "flow_noevo_zoom_nofloor", "flow_noevo_zoom_randrec",
                                         "flow_noevo_zoom_shufcand", "flow_noevo_zoom_after", "flow_noevo_zoom_fixed",
                                         "flow_noevo_bm25", "flow_noevo", "flow_noevo_noexp", "flow_noevo_nolinks", "flow_noevo_index",
                                         "flow_noevo_digest", "flow_noevo_docsig", "trace",
                                         "summary_amem_s70", "amem", "amem_zoom", "afm", "afm_t30_15", "afm_t20_10", "afm_t15_07",
                                         "rag_pieces", "rag_raw", "masking", "masking_trace", "truncation", "passthrough"))}
    keys = sorted({(r["method"], r["budget"]) for r in rows}, key=lambda k: (k[1] == 0, k[1], order.get(k[0], 99), k[0]))
    return [((m, b), [r for r in rows if (r["method"], r["budget"]) == (m, b)]) for m, b in keys]


def main_table(rows) -> Table:
    out = []
    for (m, b), rs in arms(rows):
        k = sum(r["success"] for r in rs)
        sc = defaultdict(list)
        for r in rs:
            sc[r["scenario"]].append(r["success"])
        full = [all(v) for v in sc.values() if len(v) == 3]
        out.append([label(m), f"{b // 1000}k" if b else "uncapped", len(rs), ci(*S.wilson(k, len(rs))),
                    f"{pct(mean(full))} ({len(full)})" if full else "–",
                    pct(mean(r["passes"] / r["num_tests"] for r in rs if r["num_tests"])),
                    f"{mean(r['steps'] for r in rs):.1f}", f"{mean(r['compacted_steps'] for r in rs):.1f}",
                    ktok(mean(r["tok_agent"] + r["tok_memory"] for r in rs)), ktok(mean(r["tok_memory"] for r in rs)),
                    pct(sum(r["n_over_b"] for r in rs) / n) if (n := sum(r["n_comp_api"] for r in rs)) else "–",
                    f"{max(x):.2f}" if (x := [r["max_api_b"] for r in rs if r["max_api_b"]]) else "–",
                    sum(bool(r.get("error")) for r in rs)])
    return Table("appworld_main", f"AppWorld {split_name(rows)} by arm and budget",
                 ["Method", "B", "n", "TGC % [95% CI]", "SGC % (scenarios)", "tests passed %", "steps", "compacted",
                  "tokens/episode", "memory tokens", "API > B % (compacted)", "max API/B", "method errors"], out,
                 ["TGC = tasks with every unit test passed (Wilson interval); SGC = scenarios whose 3 variants all "
                  "succeed, over scenarios with all 3 run. Tokens are billed (prompt + completion) per episode from the "
                  "ledger, agent and memory calls; memory tokens are the part spent writing and rendering memory. The "
                  "cap holds on our o200k count; API > B is the share of compacted steps whose API-billed prompt "
                  "exceeded B (deviation 4)."])


def contrasts(rows, ours: str = OURS) -> Table:
    by = {(r["task_id"], r["method"], r["budget"]): r for r in rows}
    out = []
    for (m, b), rs in arms(rows):
        if m == ours:
            continue
        for bo in sorted({r["budget"] for r in rows if r["method"] == ours}):
            if b and b != bo:
                continue
            pairs = [(by[(r["task_id"], ours, bo)], r) for r in rs if (r["task_id"], ours, bo) in by]
            if not pairs:
                continue
            a, o = [p[0]["success"] for p in pairs], [p[1]["success"] for p in pairs]
            t = S.mcnemar_exact(a, o)
            bs = S.paired_bootstrap([x - y for x, y in zip(a, o)], [p[0]["scenario"] for p in pairs])
            out.append([f"{label(ours)} vs {label(m)}", f"{bo // 1000}k" + ("" if b else " vs uncapped"), len(pairs),
                        pct(mean(a)), pct(mean(o)), f"{100 * bs['est']:+.1f} [{100 * bs['lo']:+.1f}, {100 * bs['hi']:+.1f}]",
                        f"{t['n10']}/{t['n01']}", f"{t['p']:.3g}"])
    return Table("appworld_contrasts", f"AppWorld: {label(ours)} against each arm on the same tasks (success)",
                 ["Comparison", "B", "n pairs", "first %", "other %", "Δ (pts) [95% CI, by scenario]", "wins/losses",
                  "McNemar p"], out,
                 ["Exact McNemar on paired task success; the CI resamples whole scenarios (10k, seeded). Not corrected "
                  "for multiplicity."])
