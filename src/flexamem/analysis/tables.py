"""Paper tables and the P1-P3 endpoint tests (docs/protocol/experiment_protocol.md) from result rows (last row per
key), judge labels, ledgers and SWE trajectories. SWE instances whose gold patch fails
(results/swe_verified/gold_validation/README.md) and the two instances without an image are excluded for every arm;
ungraded SWE episodes (awaiting --regrade) are counted but have no outcome. LongMemEval outcomes use the primary
judge (j1) for tests; j2 is reported alongside. Loaded runs must not share an episode key. The Summary+A-MEM split
of the endpoints comes from configs/exp/endpoints.yaml (or the dev rule while it is not frozen), and so does the Flow
arm that the endpoints and comparisons use (the original `flow` while it is not frozen)."""
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
import yaml

from flexamem.analysis import cost as C
from flexamem.analysis import stats as S
from flexamem.bench.common import git_state, read_rows
from flexamem.llm.client import ROOT

UNAVAILABLE = ("django__django-11138", "sympy__sympy-24661")  # images cannot be pulled (protocol)
N_GOLD_FAILED = 18  # the paper's gold validation (configs/splits/manifest_post_gold.json: excluded minus UNAVAILABLE)
N_FULL = 500  # questions (LongMemEval-S) / instances (SWE-bench Verified)
ENDPOINT_BUDGET = {"swe": 8000, "lme": 16000}  # P1 at SWE 8k; P2, P3 at LongMemEval 16k
DEV = {"swe": "swe_verified_dev20.txt", "lme": "lme_dev60.txt"}
SPLITS = {"swe": (("Dev20", DEV["swe"]), ("Eval-100", "swe_verified_eval100.txt")), "lme": (("dev60", DEV["lme"]),)}
BENCH = {"swe": "SWE", "lme": "LongMemEval"}
def NO_CALL(r) -> bool:  # SWE method error before any agent call succeeded: the fixed prompt exceeds the cap
    # (adapters/mswea.py) or the first prompt was rejected REJECT_SENDS times (llm/agent_model.py)
    return bool(r.get("error")) and not r["cost"]["agent_calls"]


LME_TYPES = ("single-session-user", "single-session-assistant", "single-session-preference", "multi-session",
             "knowledge-update", "temporal-reasoning")
ABL_SUFFIX = {"_noexp": "w/o expansion", "_nolinks": "w/o links", "_stacked": "stacked", "_noevo": "w/o evolution",
              "_links": "w/o rewrites", "_noevo_fullemb": "w/o evolution, full-text embedding",
              "_fullemb": "full-text embedding", "_views": "graded views", "_zoom": "+ fit-or-degrade"}
# "_noevo_fullemb" before "_fullemb": ablation_of takes the first match
# added after P0 (design deviations 28-30, 32): paired with their Flow arm (A-MEM Zoom, "_zoom", with the main arm Flow
# w/o evolution), but outside the Holm family
EXPLORATORY = ("_links", "_fullemb", "_views", "_zoom")
ABLATIONS = tuple(f"flow{x}" for x in ABL_SUFFIX)
LABELS = {
    "trace": "Action trace (rule-based records, no LLM)",
    "flow_noevo_index": "Flow w/o evolution + explored-so-far floor",
    "flow_noevo_digest": "Flow w/o evolution + exact-name digests",
    "flow_noevo_docsig": "Flow w/o evolution + doc signatures",
    "flow": "Flow", "amem": "A-MEM (retrieval-only)", "memgas": "MemGAS", "sculptor": "Sculptor",
          "readagent": "ReadAgent", "masking": "Obs. masking", "truncation": "Truncation", "passthrough": "Passthrough", "full": "Full history",
          "oracle": "Oracle",
          "flow_noexp": "Flow w/o expansion", "flow_nolinks": "Flow w/o links", "flow_stacked": "Flow-stacked",
          "flow_noevo": "Flow w/o evolution", "flow_links": "Flow w/o rewrites", "summary_amem": "Summary+A-MEM (split undecided)",
          "flow_s30": "Flow (history 0.3)", "flow_s30_ld50": "Flow (history 0.3, link disc. 0.5)",
          "flow_noevo_zoom": "FlexA-MEM",
          "rag_raw": "Raw retrieval (MiniLM chunks, no build)", "rag_pieces": "Piece retrieval (BM25 pieces, no build)",
          "amem_zoom": "A-MEM + fit-or-degrade", "afm": "AFM (2025, reproduced)", "masking_trace": "Obs. masking + rule-based trace",
          "afm_t30_15": "AFM, thresholds 0.30/0.15", "afm_t20_10": "AFM, thresholds 0.20/0.10",
          "afm_t15_07": "AFM, thresholds 0.15/0.07", "afm_t08_05": "AFM, thresholds 0.08/0.05", "amem_zoom_e150": "A-MEM + fit-or-degrade (E 150)",
          "flow_s30_ld50_noevo_zoom_e150_nofloor": "A-MEM Zoom, E 150, no lines", "summary_flow_s50": "Summary + Flow notes retrieval (split 0.50)", "flow_s30_ld50_zoom": "Flow (with evolution) + fit-or-degrade",
          "flow_s30_ld50_noevo_noexp": "Flow w/o evolution, lines only",
          "flow_s30_ld50_noevo_zoom_nofloor": "A-MEM Zoom w/o lines", "flow_s30_ld50_noevo_zoom_nolinks": "A-MEM Zoom w/o links",
          "flow_noevo_zoom_noexp": "A-MEM Zoom w/o expansion (records only)", "flow_noevo_zoom_nolinks": "A-MEM Zoom w/o links",
          "flow_noevo_zoom_nofloor": "A-MEM Zoom w/o lines",
          "flow_noevo_bm25": "Timeline + BM25 pieces (A-MEM Zoom's records, zero-build recovery)",
          "flow_s30_ld50_noevo_bm25": "Timeline + BM25 pieces (A-MEM Zoom's records, zero-build recovery)",
          "flow_noevo_zoom_randrec": "A-MEM Zoom, random recovery",
          "flow_noevo_zoom_shufcand": "A-MEM Zoom, same candidates in random order",
          "flow_noevo_zoom_after": "A-MEM Zoom, same selection shown after the timeline",
          "flow_noevo_zoom_fixed": "A-MEM Zoom, fixed-length recovery (excerpts only)",
          "flow_s30_ld50_noevo_zoom_randrec": "A-MEM Zoom, random recovery",
          "flow_s30_ld50_noevo_zoom_shufcand": "A-MEM Zoom, same candidates in random order",
          "flow_s30_ld50_noevo_zoom_after": "A-MEM Zoom, same selection shown after the timeline",
          "flow_s30_ld50_noevo_zoom_fixed": "A-MEM Zoom, fixed-length recovery (excerpts only)", "flow_s30_ld50_noevo_zoom_s50": "A-MEM Zoom, history 0.5",
          "flow_s30_ld50_noevo_zoom_rawfirst": "A-MEM Zoom, raw first (E 150)", "flow_s30_ld50_noevo_zoom_e150": "A-MEM Zoom, E 150",
          "flow_s30_ld50_noevo_zoom_e600": "A-MEM Zoom, E 600", "flow_s30_ld50_noevo_zoom_lead": "A-MEM Zoom, first pieces",
          "flow_s30_ld50_noevo_zoom": "FlexA-MEM (history 0.3, link discount 0.5)"}
ORDER = ("flow", "flow_s30", "flow_s30_ld50", "amem", "summary_amem_s30", "summary_amem_s50", "summary_amem_s70", "memgas",
         "sculptor", "readagent", "masking", "truncation", "passthrough", "full", "oracle", *ABLATIONS, "flow_noevo_zoom",
         "flow_s30_ld50_noevo_zoom")
SUB = {"Δ": r"$\Delta$", "κ": r"$\kappa$", "δ": r"$\delta$", "–": "--", "≤": r"$\le$", "·": r"$\cdot$", "%": r"\%",
       "&": r"\&", "_": r"\_", "#": r"\#"}


@dataclass
class Table:
    name: str
    title: str
    header: list
    rows: list
    notes: list = field(default_factory=list)

    def md(self) -> str:
        out = [f"### {self.title}", "", "| " + " | ".join(self.header) + " |", "|" + "---|" * len(self.header)]
        out += ["| " + " | ".join(map(str, r)) + " |" for r in self.rows] or ["| (no data) |" + " |" * (len(self.header) - 1)]
        return "\n".join(out + [""] + [f"- {n}" for n in self.notes]) + "\n"

    def tex(self) -> str:
        esc = lambda s: "".join(SUB.get(ch, ch) for ch in str(s))  # noqa: E731
        return "\n".join([f"% {self.title}", r"\begin{tabular}{" + "l" * len(self.header) + "}", r"\toprule",
                          " & ".join(map(esc, self.header)) + r" \\", r"\midrule",
                          *[" & ".join(map(esc, r)) + r" \\" for r in self.rows], r"\bottomrule", r"\end{tabular}", ""])


def code_stamp() -> str:
    """Provenance of generated tables: the short HEAD sha, marked when the code that generates them (src/, scripts/,
    configs/) differs from it. The tables themselves, the cards and docs/ do not count."""
    g = git_state("src", "scripts", "configs")
    return f"`{(g['sha'] or 'no commit')[:7]}`" + (" (uncommitted changes)" if g["dirty"] else "")


# ---------------------------------------------------------------- formatting

def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def pct(x, d=1) -> str:
    return "–" if _nan(x) else f"{100 * x:.{d}f}"


def ci(p, lo, hi) -> str:
    return "–" if _nan(p) else f"{pct(p)} [{pct(lo)}, {pct(hi)}]"


def ktok(x) -> str:
    return "–" if _nan(x) else f"{x / 1000:.1f}k"


def pv(p) -> str:
    return "–" if _nan(p) else "<0.001" if p < 0.001 else f"{p:.3f}"


def num(x, d=2) -> str:
    return "–" if _nan(x) else f"{x:.{d}f}"


def mean(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else math.nan


def ablation_of(m: str) -> str | None:
    """The Flow arm an ablation arm belongs to (flow_noexp -> flow, flow_s30_ld50_noevo -> flow_s30_ld50), else None."""
    return next((m[:-len(x)] for x in ABL_SUFFIX if m.startswith("flow") and m.endswith(x)), None)


def label(m: str) -> str:
    if m in LABELS:
        return LABELS[m]
    if m.startswith("summary_amem_s"):
        return f"Summary+A-MEM (split 0.{m[-2:]})"
    base = ablation_of(m)
    return f"{label(base)} {ABL_SUFFIX[m[len(base):]]}" if base else m


def budget(b: int) -> str:
    return f"{b // 1000}k" if b else "uncapped"


def arm_key(k: tuple):
    return ORDER.index(k[0]) if k[0] in ORDER else len(ORDER), k[0], k[1]


def arms(rows) -> list[tuple[tuple, list]]:
    g = defaultdict(list)
    for r in rows:
        g[(r["method"], r["budget"])].append(r)
    return sorted(g.items(), key=lambda kv: arm_key(kv[0]))


def rate(rs, f="y"):
    v = [r[f] for r in rs if r.get(f) is not None]
    return S.wilson(sum(v), len(v)), len(v)


def paired(rows, a: tuple, b: tuple, f="y", keep=lambda r: True):
    """Items where both arms (method, budget) have a value for f: (values a, values b, clusters, rows a, rows b)."""
    ix = {(r["method"], r["budget"], r["item"]): r for r in rows if keep(r) and r.get(f) is not None}
    items = sorted(i for (m, bb, i) in ix if (m, bb) == a and (*b, i) in ix)
    ra, rb = [ix[(*a, i)] for i in items], [ix[(*b, i)] for i in items]
    return [r[f] for r in ra], [r[f] for r in rb], [r["cluster"] for r in ra], ra, rb


# ---------------------------------------------------------------- loading

def gold_failures(path=ROOT / "configs/splits/manifest_post_gold.json") -> set[str]:
    """Instance ids whose gold patch failed in the paper's gold validation: the excluded list of the post-gold split
    manifest, without the instances whose image was unavailable."""
    out = set(json.loads(path.read_text())["excluded"]) - set(UNAVAILABLE)
    assert len(out) == N_GOLD_FAILED, f"{path}: {len(out)} gold-patch failures, expected {N_GOLD_FAILED}"
    return out


def swe_excluded() -> set[str]:
    return gold_failures() | set(UNAVAILABLE)


def _split(name: str) -> set[str]:
    return set((ROOT / "configs/splits" / name).read_text().split())


def reacquisitions(messages: list[dict]) -> int:
    """Actions whose command and output both repeat an earlier action of the episode: re-reading an unchanged
    file or re-running an identical test with an unchanged result."""
    cmd = {a.get("tool_call_id"): a.get("command") for m in messages if m.get("role") == "assistant"
           for a in (m.get("extra") or {}).get("actions", [])}
    seen, n = set(), 0
    for m in messages:
        if m.get("role") == "tool" and m.get("tool_call_id") in cmd:
            k = (cmd[m["tool_call_id"]], m.get("content"))
            n += k in seen
            seen.add(k)
    return n


HIDES = ("suffix", "dip", "gist", "last")


def hide_kinds(x: dict) -> set[str]:
    """How one step (a context-log record) lost history: "suffix" (the shared overflow rule of adapters/mswea.py
    dropped the oldest whole steps: truncation's policy, masking's last resort), "dip" (Flow or Sculptor hid its
    oldest history for this render because the allowance fell below the permanent reference), "gist" (ReadAgent hid
    the gists beyond its history share) and "last" (a memory render hid history outside those cases: Flow or Sculptor
    outside a dip, or the renderer's final fit). Flow and Sculptor log all their hides in one counter, so final-fit
    drops inside a dip render count as "dip"; renders logged before the `dip` flag count as "last"."""
    rd, out = x.get("render") or {}, set()
    if x.get("dropped_steps"):
        out.add("suffix")
    if "n_pages" in rd:  # ReadAgent: overflow_dropped = gists hidden + final-fit drops (n_shown - n_hist_shown)
        fit = rd["n_shown"] - rd.get("n_hist_shown", rd["n_shown"]) if "n_shown" in rd else 0
        out |= {"gist"} if (rd.get("overflow_dropped") or 0) > fit else set()
        out |= {"last"} if fit else set()
    elif rd.get("overflow_dropped"):
        out.add("dip" if rd.get("dip") is True else "last")
    return out


def _claim(src: dict, k: tuple, rid: str):
    if src.setdefault(k, rid) != rid:
        raise ValueError(f"episode {k} has result rows in runs {src[k]} and {rid}; load runs with disjoint keys")


def load_swe(runs) -> tuple[list[dict], dict]:
    """Result rows (exclusions applied) with outcome y, cost and trajectory statistics; failures: errors.jsonl
    entries per arm by type (infra / other) and the errored episodes without a result row per arm (missing)."""
    excl, dev = swe_excluded(), _split(DEV["swe"])
    rows, src, errs = {}, {}, defaultdict(Counter)
    for rid in runs:
        d = ROOT / "artifacts/runs/swe" / rid
        for r in read_rows(d / "results.jsonl", ("iid", "method", "budget")):
            k = (r["iid"], r["method"], r["budget"])
            _claim(src, k, rid)
            rows[k] = r
        for line in (d / "errors.jsonl").read_text().splitlines() if (d / "errors.jsonl").exists() else []:
            e = json.loads(line)
            if e["key"][0] not in excl:
                errs[tuple(e["key"])]["infra" if e.get("infra", True) else "other"] += 1
    fail = {"errors": defaultdict(Counter), "missing": defaultdict(list)}
    for (iid, m, b), c in sorted(errs.items()):
        fail["errors"][(m, b)].update(c)
        if (iid, m, b) not in rows:
            fail["missing"][(m, b)].append(iid)
    by_attempt, by_key = C.swe_ledger(runs)
    out = []
    for r in rows.values():
        if r["iid"] in excl:
            continue
        p = ROOT / r["traj"]
        traj = json.loads(p.read_text()) if p.exists() else None
        log = traj["info"]["context"]["log"] if traj else []
        hides = Counter(k for x in log for k in hide_kinds(x))
        # API-billed prompt per step against B (the cap 0.95 B applies to our o200k count, not to the API's)
        api = [(x["api_prompt_tokens"], x["mode"] != "pass") for x in log if x.get("api_prompt_tokens") is not None]
        out.append({**r, "item": r["iid"], "cluster": r["repo"], "dev": r["iid"] in dev,
                    "y": r["resolved"] if r.get("graded") else None, "cost": C.swe_cost(r, by_attempt, by_key, traj),
                    "reacq": reacquisitions(traj["messages"]) if traj else None,
                    "hides": {k: hides[k] for k in HIDES} if traj else None,
                    "shrunk": sum(x.get("shrunk_observations", 0) for x in log) if traj else None,
                    "api_comp": sum(c for _, c in api), "api_max": max((t for t, _ in api), default=None),
                    "api_comp_over_b": sum(c and t > r["budget"] for t, c in api) if r["budget"] else None,
                    "our_prompt": sum(x["tokens"] for x in log) if traj else None})
    return out, fail


# Runs whose memory build shared Flow's writer outputs but raced with Flow's build on some questions: the two
# processes missed the shared LLM cache together and wrote those questions' notes independently, so on them the
# ablation also resamples the writer (results/longmemeval_s/runs/p0_lme_noevo/README.md, Issues). The committed
# question ids, one per line; a missing list stops the build.
WRITER_RACE = {}  # the paper's p0_lme_noevo flagged its raced questions; fresh runs share no builds across processes


def raced_builds() -> set[str]:
    """Build ids of the writer-raced builds. A run that renders them without building (A-MEM Zoom over the main arm's P0
    builds) carries the flag on the same questions."""
    key = ("qid", "method", "budget")
    return {b for rid in WRITER_RACE for r in read_rows(ROOT / "artifacts/runs/lme" / rid / "results.jsonl", key)
            if r["qid"] in writer_race_qids(rid) for b in (r.get("build_ids") or {}).values()}


def writer_race_qids(rid: str) -> set[str]:
    if rid not in WRITER_RACE:
        return set()
    p = ROOT / WRITER_RACE[rid]
    if not p.exists():
        raise FileNotFoundError(f"{rid}: writer-race question list missing ({WRITER_RACE[rid]})")
    return set(p.read_text().split())


def load_lme(runs) -> list[dict]:
    """Result rows joined with both judges' labels of their own run (None if not judged yet) and cost; rows of a
    question whose build raced on the writer (WRITER_RACE), or that render such a build, carry writer_raced."""
    rows, src, dev, raced = {}, {}, _split(DEV["lme"]), raced_builds()
    key = ("qid", "method", "budget")
    for rid in runs:
        d = ROOT / "artifacts/runs/lme" / rid
        judged = ({tuple(j[f] for f in key): j for j in read_rows(d / "judged.jsonl", key)}
                  if (d / "judged.jsonl").exists() else {})
        race = writer_race_qids(rid)
        for r in read_rows(d / "results.jsonl", key):
            k = tuple(r[f] for f in key)
            _claim(src, k, rid)
            rows[k] = {**r, "y": judged.get(k, {}).get("j1"), "y2": judged.get(k, {}).get("j2"),
                       "writer_raced": r["qid"] in race or bool(raced & set((r.get("build_ids") or {}).values()))}
    builds, query = C.lme_ledger(runs)
    return [{**r, "item": r["qid"], "cluster": r["qid"], "dev": r["qid"] in dev, "cost": C.lme_cost(r, builds, query)}
            for r in rows.values()]


def expected_n(bench: str, heldout: bool) -> int:
    """Items of an endpoint population as pre-registered (docs/protocol/experiment_protocol.md, Splits and
    Endpoints): all 500 of the benchmark whatever runs are loaded, minus the SWE exclusions and, for the sensitivity
    populations (SWE w/o Dev20, LongMemEval held-out), the dev split."""
    excl, dev = swe_excluded() if bench == "swe" else set(), _split(DEV[bench])
    return N_FULL - len(excl) - (len(dev - excl) if heldout else 0)


def loaded_split(rows, bench: str) -> str | None:
    """The smallest named split below all 500 holding every loaded item of the benchmark (e.g. Eval-100 when only
    P0 SWE runs are loaded), else None: a result on it is not the endpoint's population."""
    items = {r["item"] for r in rows}
    return next((name for name, f in SPLITS[bench] if items and items <= _split(f)), None)


def split_dev(rows, bench: str) -> tuple[list, int | None]:
    """Summary+A-MEM splits on the dev items where every split has an outcome at that budget:
    [(budget, split, n, correct, mean uncached cost)] and the pick of the rule of
    results/decisions/2026-09-24_summary_amem_split.md at the endpoint budget: highest dev accuracy, ties -> lower
    mean uncached cost, remaining ties -> 50."""
    g = defaultdict(dict)
    for r in rows:
        if r["dev"] and r["method"].startswith("summary_amem_s") and r["y"] is not None:
            g[r["budget"]].setdefault(int(r["method"][-2:]), {})[r["item"]] = r
    out, pick = [], None
    for b in sorted(g):
        common = set.intersection(*map(set, g[b].values()))
        st = {s: (sum(v[i]["y"] for i in common), mean(v[i]["cost"]["total"] for i in common))
              for s, v in sorted(g[b].items())}
        out += [(b, s, len(common), k, c) for s, (k, c) in st.items()]
        if b == ENDPOINT_BUDGET[bench] and common:
            pick = min(st, key=lambda s: (-st[s][0], st[s][1], s != 50))
    return out, pick


def flow_choice(lme=(), swe=(), override: dict | None = None) -> dict:
    """bench -> (Flow arm of the endpoints and comparisons, source): the CLI override, else configs/exp/endpoints.yaml,
    else `flow`. Runs that predate the frozen arm (dev runs of `flow` only) fall back to `flow`, said in the source."""
    cfg = yaml.safe_load((ROOT / "configs/exp/endpoints.yaml").read_text()).get("flow_arm") or {}
    out = {}
    for bk, rows in (("swe", swe), ("lme", lme)):
        loaded = {r["method"] for r in rows}
        if (override or {}).get(bk):
            out[bk] = (override[bk], "CLI override")
        elif not cfg.get(bk):
            out[bk] = ("flow", "default, not frozen")
        elif cfg[bk] not in loaded and "flow" in loaded:
            out[bk] = ("flow", f"{cfg[bk]} is frozen in configs/exp/endpoints.yaml but has no rows in these runs")
        else:
            out[bk] = (cfg[bk], "frozen in configs/exp/endpoints.yaml")
    return out


def summary_choice(lme, swe, override: dict | None = None) -> dict:
    """bench -> (Summary+A-MEM split or None, source): the CLI override, else configs/exp/endpoints.yaml, else the
    dev rule on the loaded rows."""
    cfg = yaml.safe_load((ROOT / "configs/exp/endpoints.yaml").read_text())["summary_split"]
    out = {}
    for bench, rows in (("swe", swe), ("lme", lme)):
        if (override or {}).get(bench) is not None:
            out[bench] = int(override[bench]), "CLI override"
        elif cfg.get(bench) is not None:
            out[bench] = int(cfg[bench]), "frozen in configs/exp/endpoints.yaml"
        elif (pick := split_dev(rows, bench)[1]) is not None:
            out[bench] = pick, "split by rule on loaded dev rows, not frozen"
        else:
            out[bench] = None, f"undecided: not frozen, no Summary+A-MEM dev rows at {budget(ENDPOINT_BUDGET[bench])}"
    return out


# ---------------------------------------------------------------- LongMemEval tables

def race_note(rows) -> list[str]:
    """Note for LongMemEval tables that show a writer-raced arm (WRITER_RACE) on all its questions."""
    raced = [r for r in rows if r.get("writer_raced")]
    return ([f"{', '.join(sorted({label(r['method']) for r in raced}))}: all questions, including "
             f"{len({r['item'] for r in raced})} whose build raced on Flow's writer outputs (see the ablations "
             f"table, which tests the others)."] if raced else [])


def lme_accuracy(rows) -> Table:
    out = []
    for (m, b), rs in arms(rows):
        (p1, l1, h1), n1 = rate(rs)
        (p2, l2, h2), _ = rate(rs, "y2")
        both = [r for r in rs if r["y"] is not None and r["y2"] is not None]
        agree, kappa = S.agreement([r["y"] for r in both], [r["y2"] for r in both])
        ho, ab = rate([r for r in rs if not r["dev"]])[0], rate([r for r in rs if r["abstention"]])[0]
        out.append([label(m), budget(b) if b else "–", len(rs), n1, ci(p1, l1, h1), ci(p2, l2, h2), pct(agree),
                    num(kappa), pct(ho[0]), pct(ab[0]), sum(bool(r.get("error")) for r in rs)])
    return Table("lme_accuracy", "LongMemEval-S accuracy (%) by method and budget",
                 ["Method", "B", "n", "judged", "J1 acc [95% CI]", "J2 acc [95% CI]", "agree %", "κ",
                  "J1 held-out", "J1 abstention", "method errors"], out,
                 ["J1 = Claude 4.6 Sonnet (primary), J2 = GPT-5.4; Wilson intervals. Held-out = all questions "
                  "outside dev60. Method-error rows are scored wrong by both judges."] + race_note(rows))


def lme_types(rows, f="y") -> Table:
    j = "J1" if f == "y" else "J2"
    out = []
    for (m, b), rs in arms(rows):
        cells = [ci(*rate([r for r in rs if r["qtype"] == t], f)[0]) for t in LME_TYPES]
        out.append([label(m), budget(b) if b else "–", *cells, ci(*rate([r for r in rs if r["abstention"]], f)[0])])
    return Table(f"lme_types_{j.lower()}", f"LongMemEval-S accuracy (%) per question type, judge {j}",
                 ["Method", "B", *LME_TYPES, "abstention"], out,
                 ["Wilson 95% intervals. Abstention questions are also counted in their type."] + race_note(rows))


def lme_cost(rows) -> Table:
    ph = ("write", "link", "merge", "summarize", "gist", "lookup", "filter", "reader")
    out = []
    for (m, b), rs in arms(rows):
        c = [r["cost"] for r in rs]
        # Sculptor reads over several turns: its API mean is over the turns, so compare it with our mean over them
        ours = [(r.get("render") or {}).get("mean_prompt") or r.get("prompt_tokens") for r in rs]
        api = [x["api_prompt"] / o for o, x in zip(ours, c) if x["api_prompt"] and o]
        rend = [r.get("render") or {} for r in rs]
        out.append([label(m), budget(b) if b else "–", len(rs), ktok(mean(x["build"] for x in c)),
                    num(mean(x["build_calls"] for x in c), 1), ktok(mean(x["query"] for x in c)),
                    ktok(mean(x["total"] for x in c)), *[ktok(mean(x["phase"].get(p, 0) for x in c)) for p in ph],
                    num(mean(x["phase_calls"].get("embed", 0) for x in c), 1),
                    num(mean(s.get("merges") for s in rend), 1), num(mean(s.get("merge_depth") for s in rend), 1),
                    max((s.get("merge_depth") or 0 for s in rend), default=0), num(mean(api), 3)])
    miss = sum(r["cost"]["missing_builds"] for r in rows)
    bf = arms([r for r in rows if r["cost"]["build_failed"]])
    old = sum(not r["cost"]["attempt_matched"] for r in rows)
    rerun = sum(r["cost"]["rerun_tokens"] for r in rows)
    return Table("lme_cost", "LongMemEval-S cost per question (tokens, uncached)",
                 ["Method", "B", "n", "build", "build calls", "query", "total", *ph, "embed batches", "merges",
                  "depth", "max depth", "API/ours prompt"], out,
                 ["Tokens = prompt + completion (incl. reasoning), cached calls at their original usage. Build = the "
                  "question's memory builds, charged in full to every method that uses them (identical across "
                  "budgets); query = render-time calls + reader. Phase columns are means per question. API/ours = "
                  "reader prompt tokens billed by the API over our tiktoken count, cache-hit reader calls excluded "
                  "(questions whose reader calls were all cache hits are skipped)."]
                 + ([f"{miss} build ids referenced by result rows have no ledger rows (build cost undercounted)."]
                    if miss else [])
                 + ([f"{sum(len(rs) for _, rs in bf)} result rows failed in the build (method error, no build ids): "
                     f"their build cost is not counted (build undercounted): "
                     + ", ".join(f"{label(m)} {budget(b)}: {len(rs)}" for (m, b), rs in bf) + "."] if bf else [])
                 + ([f"{old} of {len(rows)} result rows predate the attempt id: their render and reader rows are "
                     f"matched on (run, question, method, budget) only, so a question rerun after an API or disk "
                     f"error also counts its earlier attempt's rows."] if old else [])
                 + ([f"Render and reader rows of other attempts (reruns after API or disk errors), not counted: "
                     f"{ktok(rerun)} tokens in total."] if rerun else [])
                 + race_note(rows))


# ---------------------------------------------------------------- SWE tables

def swe_main(rows) -> Table:
    out = []
    for (m, b), rs in arms(rows):
        (p, lo, hi), n = rate(rs)
        comp = [r for r in rs if r["compacted_steps"]]
        tr = [r for r in rs if r.get("api_prompt_tokens") is not None and r["our_prompt"]]
        hide = lambda k: sum((r["hides"] or {}).get(k, 0) for r in rs)  # noqa: E731
        n_comp, api_max = sum(r["api_comp"] for r in rs), max((r["api_max"] or 0 for r in rs), default=0)
        out.append([label(m), budget(b), len(rs), n, ci(p, lo, hi), pct(rate([r for r in rs if not r["dev"]])[0][0]),
                    pct(len(comp) / len(rs)), num(mean(r["compacted_steps"] for r in rs), 1),
                    num(mean(r["steps"] for r in rs), 1), *[hide(k) for k in HIDES],
                    sum(r["shrunk"] or 0 for r in rs), num(mean(r["reacq"] for r in rs), 2),
                    pct(sum(r["api_comp_over_b"] for r in rs) / n_comp) if b and n_comp else "–",
                    num(api_max / b) if b and api_max else "–",
                    num(sum(r["api_prompt_tokens"] for r in tr) / sum(r["our_prompt"] for r in tr) if tr else None, 3)])
    shas = defaultdict(list)
    for (m, b), rs in arms(rows):
        shas["+".join(sorted({(r.get("git_sha") or "?")[:7] for r in rs}))].append(f"{label(m)} {budget(b)}")
    return Table("swe_main", "SWE-bench Verified: resolved rate (%) and context statistics per arm",
                 ["Method", "B", "n", "graded", "resolved [95% CI]", "w/o Dev20", "compacted %", "compacted steps",
                  "steps", "suffix-rule steps", "dip-hide steps", "gist-hide steps", "last-resort hide steps",
                  "shrunk obs.", "reacq./issue", "API > B % (compacted)", "max API/B",
                  "API/ours prompt (ratio of sums)"], out,
                 ["Excludes gold-patch failures (results/swe_verified/gold_validation/README.md) and the two instances without an image. "
                  "Wilson intervals over graded episodes. Compacted steps and steps are means per issue.",
                  "Steps that lost history, summed over the arm's episodes, by logged cause: suffix rule = the shared "
                  "overflow rule dropped the oldest whole steps (truncation's policy; masking's last resort once its "
                  "window reaches 1); dip hide = Flow or Sculptor hid its oldest history for one render because the "
                  "allowance fell below the permanent reference (by design); gist hide = ReadAgent hid the gists "
                  "beyond its history share (by design); last-resort hide = a memory render hid history in any other "
                  "case (Flow or Sculptor outside a dip, or the renderer's final fit; final-fit drops inside a dip "
                  "are logged with the dip hides). Shrunk obs.: observations cut by the last-resort shrink.",
                  "Reacquisition: an action whose command and output repeat an earlier one.",
                  "The cap 0.95·B bounds our o200k count of each prompt, not the API's count. API > B %: compacted "
                  "steps whose API-billed prompt exceeds B, over compacted steps with an API count; max API/B: the "
                  "largest API-billed prompt of any step, over B. API/ours: sum of API-billed over sum of our prompt "
                  "tokens, over episodes with both counts (a ratio of sums, which hides the per-step tail)."]
                 + (["Code versions (git_sha) per arm: " + "; ".join(f"{v} ({', '.join(a)})" for v, a in shas.items())
                     + "."] if shas else []))


def swe_cost(rows) -> Table:
    ph = ("agent", "write", "link", "merge", "summarize", "gist", "lookup", "filter")
    out = []
    for (m, b), rs in arms(rows):
        c = [r["cost"] for r in rs]
        tot = sum(x["total"] for x in c)
        sim = [x for x in c if x["sim_cached"] is not None]
        out.append([label(m), budget(b), len(rs), ktok(mean(x["total"] for x in c)),
                    ktok(mean(x["cache_adjusted"] for x in c)), ktok(mean(x["no_build"] for x in c)),
                    pct(sum(x["memory"] for x in c) / tot if tot else math.nan),
                    pct(sum(x["sim_cached"] for x in sim) / max(1, sum(x["api_prompt"] for x in sim)) if sim else None),
                    pct(sum(x["api_cached"] for x in c) / max(1, sum(x["api_prompt"] for x in c))),
                    *[ktok(mean(x["phase"].get(p, 0) for x in c)) for p in ph],
                    ktok(mean(x["build"] for x in c)), ktok(mean(x["render"] for x in c))])
    lb = arms([r for r in rows if r["cost"]["sim_mode"] == "lower_bound"])
    unknown = [r for r in rows if not r["cost"]["agent_calls"] or not r["cost"]["total"]]
    name = lambda r: f"{r['iid']} ({label(r['method'])} {budget(r['budget'])})"  # noqa: E731
    nc, nl = [name(r) for r in unknown if NO_CALL(r)], [name(r) for r in unknown if not NO_CALL(r)]
    return Table("swe_cost", "SWE-bench Verified: cost per issue (tokens)",
                 ["Method", "B", "n", "uncached", "cache-adj.", "w/o build", "memory share %", "sim. cached %",
                  "API cached %", *ph, "build", "render"], out,
                 [f"Uncached: every call at its full usage. Cache-adjusted: simulated cached agent-prompt tokens "
                  f"billed at {C.CACHED_WEIGHT} (offline longest-common-prefix simulation over consecutive agent "
                  f"prompts, extended into the unchanged start of the memory message, analysis/cost.py); memory "
                  f"calls uncached. W/o build: agent + render-time memory calls. Sim./API cached %: share of agent "
                  f"prompt tokens cached in the simulation / as reported by the API."]
                 + ([f"Sim. cached and cache-adj. are lower bounds for {sum(len(rs) for _, rs in lb)} episodes logged "
                     f"before the memory-prefix log, whose memory message counts as changed at every compacted step: "
                     + ", ".join(f"{label(m)} {budget(b)}: {len(rs)}" for (m, b), rs in lb) + "."] if lb else [])
                 + ([f"{len(nc)} episodes stopped with a method error before any agent call succeeded (0 tokens here; "
                     f"scored unresolved; the P1 cost test leaves them out of its pairs, as their log cost ratio is "
                     f"undefined): " + ", ".join(nc) + "."] if nc else [])
                 + ([f"{len(nl)} other episodes have no agent ledger rows or zero tokens (cost unknown, 0 here; the P1 "
                     f"cost test stops on them): " + ", ".join(nl) + "."] if nl else []))


def swe_failures(rows, fail) -> Table:
    out, g = [], dict(arms(rows))
    for k in sorted(set(g) | set(fail["errors"]), key=arm_key):
        m, b = k
        rs, e, miss = g.get(k, []), fail["errors"].get(k, {}), fail["missing"].get(k, [])
        ga = lambda x: sum(r.get("graded_as") == x for r in rs)  # noqa: E731
        out.append([label(m), budget(b), len(rs), sum(bool(r.get("error")) for r in rs),
                    sum(r.get("format_errors") or 0 for r in rs), ga("timeout"), ga("apply_failed"),
                    sum(r.get("graded_as") == "empty_patch" and not r.get("error") for r in rs),
                    sum(r.get("graded") is False for r in rs), e.get("infra", 0), e.get("other", 0), len(miss),
                    sum(r["cost"]["reruns"] for r in rs), ktok(sum(r["cost"]["rerun_tokens"] for r in rs))])
    miss = [f"{label(m)} {budget(b)}: {', '.join(v)}" for (m, b), v in sorted(fail["missing"].items(),
                                                                             key=lambda kv: arm_key(kv[0]))]
    return Table("swe_failures", "SWE-bench Verified: failures per arm",
                 ["Method", "B", "n", "method errors", "format-error calls", "grading timeouts", "apply failed",
                  "empty patch", "ungraded", "infra errors", "other errors", "missing", "other attempts",
                  "other-attempt tokens"], out,
                 ["Method errors, timeouts, failed applies and empty patches are scored unresolved (timeouts flagged); "
                  "empty patch excludes method-error episodes. Ungraded episodes await --regrade and have no outcome. "
                  "Infra / other errors: errors.jsonl entries with an infrastructure / any other exception (rerun on "
                  "resume); missing: errored episodes with no result row (no outcome; counted against the endpoints' "
                  "expected n). Other attempts: ledger attempts of the same episode key and run that are not the "
                  "result row's attempt (not counted in the arm's cost)."]
                 + (["Missing episodes: " + "; ".join(miss) + "."] if miss else []))


# ---------------------------------------------------------------- comparisons

def _compare(rows, a, b, f="y", keep=lambda r: True):
    ya, yb, cl, *_ = paired(rows, a, b, f, keep)
    return ya, yb, S.mcnemar_exact(ya, yb)


def _holm_last(rows):
    """Replaces the raw p-value in the last column by its Holm-adjusted value (one family: the finite ones)."""
    ok = [r for r in rows if not _nan(r[-1])]
    for r, h in zip(ok, S.holm([r[-1] for r in ok])):
        r[-1] = pv(h)
    for r in rows:
        r[-1] = "–" if _nan(r[-1]) else r[-1]
    return rows


def ablations(lme, swe, flow: dict) -> Table:
    """Each ablation against the Flow arm whose render settings it keeps, on paired items. Holm families: per
    benchmark, the ablations of the endpoints' Flow arm (flow_choice); an ablation whose build raced on the writer
    for some questions (writer_raced) enters with its clean questions, and its all-question row is flagged."""
    out = []
    for bk, rows in (("lme", lme), ("swe", swe)):
        part = []
        for (m, b), rs in arms([r for r in rows if ablation_of(r["method"])]):
            raced = {r["item"] for r in rs if r.get("writer_raced")}
            variants = ([(f" [{len({r['item'] for r in rs}) - len(raced)} writer-clean]",
                          lambda r, x=raced: r["item"] not in x, True),
                         (f" [all, incl. {len(raced)} writer-raced; flagged]", lambda r: True, False)]
                        if raced else [("", lambda r: True, True)])
            for tag, keep, fam in variants:
                ya, yb, _, ra, _ = paired(rows, (m, b), (ablation_of(m), b), keep=keep)
                t = S.mcnemar_exact(ya, yb)
                p = t["p"] if t["n"] else math.nan
                part.append([BENCH[bk], label(m) + tag, budget(b), t["n"], pct(mean(ya)),
                             pct(mean(r["y2"] for r in ra)) if bk == "lme" else "–", pct(mean(yb)),
                             pct(mean(ya) - mean(yb)), f"{t['n10']}/{t['n01']}",
                             ktok(mean(r["cost"]["total"] for r in rs if keep(r))), pv(p),
                             p if fam and ablation_of(m) == flow[bk] and not m.endswith(EXPLORATORY) else math.nan])
        out += _holm_last(part)
    raced = {r["item"] for r in lme if r.get("writer_raced")}
    return Table("ablations", "Ablations against their Flow arm at the same budget (paired items)",
                 ["Benchmark", "Ablation", "B", "n pairs", "acc %", "J2 acc %", "Flow acc %", "Δ (pts)",
                  "abl./Flow wins", "cost/item", "McNemar p", "Holm p"], out,
                 ["LongMemEval accuracy with J1 (J2 descriptive, over the same paired items); SWE resolved rate. Each "
                  "ablation is compared with the Flow arm whose render settings it keeps (named in its label). Exact "
                  "McNemar on paired items. Holm correction per benchmark over the ablations of the endpoints' Flow "
                  "arm (" + "; ".join(f"{BENCH[bk]}: {label(flow[bk])}" for bk in ("lme", "swe")) + "); other rows, "
                  "and the exploratory rows added after P0 (" + ", ".join(ABL_SUFFIX[x] for x in EXPLORATORY) + "), have no Holm p."]
                 + ([f"Flow w/o evolution shares Flow's writer outputs by design, but on {len(raced)} questions its "
                     f"build raced with Flow's and wrote its own notes, so there the ablation also resamples the "
                     f"writer. Its test uses the other questions ('writer-clean'); the row over all questions is kept "
                     f"for reference, flagged and outside the Holm family."] if raced else []))


def secondary(lme, swe, summary: dict, flow: dict) -> Table:
    """Descriptive Flow-vs-baseline comparisons (everything outside P1-P3), one Holm family; references
    (truncation, passthrough, full history, oracle) are not tested."""
    tests = []
    primary = {("swe", ENDPOINT_BUDGET["swe"], summary["swe"]), ("lme", ENDPOINT_BUDGET["lme"], summary["lme"]),
               ("lme", ENDPOINT_BUDGET["lme"], "amem")}
    for bk, rows in (("lme", lme), ("swe", swe)):
        bench = BENCH[bk]
        for b in sorted({r["budget"] for r in rows if r["method"] == flow[bk]}):
            for o in ("amem", summary[bk], "memgas", "sculptor", "readagent", "masking"):
                subsets = [("all", lambda r: True)] if (bk, b, o) not in primary else []
                if bench == "LongMemEval":
                    subsets += [(t, lambda r, t=t: r["qtype"] == t) for t in LME_TYPES]
                    subsets += [("abstention", lambda r: r["abstention"])]
                for name, keep in subsets:
                    ya, yb, t = _compare(rows, (flow[bk], b), (o, b), keep=keep)
                    if t["n"]:
                        tests.append([bench, f"{label(flow[bk])} vs {label(o)}", budget(b), name, t["n"], pct(mean(ya)),
                                      pct(mean(yb)), pct(mean(ya) - mean(yb)), pv(t["p"]), t["p"]])
    return Table("secondary", "Descriptive comparisons, Flow vs baselines (Holm-corrected)",
                 ["Benchmark", "Comparison", "B", "subset", "n pairs", "Flow %", "other %", "Δ (pts)", "McNemar p",
                  "Holm p"], _holm_last(tests),
                 ["Exact McNemar on paired items (LongMemEval: J1); one Holm family over all rows of this table. The "
                  "P1-P3 comparisons themselves (subset all of Flow vs Summary+A-MEM at SWE 8k and LongMemEval 16k, "
                  "Flow vs A-MEM at LongMemEval 16k) are in the endpoints table, not in this family. References "
                  "(truncation, passthrough, full history, oracle) are descriptive only and not tested."])


# Exploratory paired comparisons at SWE 8k (swe_paired): the main arm chosen after P1 and the registered Flow, each
# against the arms that bound it (results/decisions/2026-10-04_main_arm_noevo.md). P1's own row (Flow vs
# Summary+A-MEM over all issues) is left to the endpoints table, so none of them is an endpoint.
PAIRED_ARMS = ("flow_noevo", "flow", "flow_noevo_zoom")  # A-MEM Zoom (deviation 32) last: its first pair is the main arm
PAIRED_POPS = (("Eval-95", "swe_verified_eval95.txt"), ("rest385", "swe_verified_rest385.txt"), ("all 480", None))
MAIN_ARM_ROLE = {"Eval-95": "selection set", "rest385": "check set", "all 480": "pooled"}  # of flow_noevo


def swe_paired(swe, summary: dict, flow: dict) -> Table:
    """Each arm of PAIRED_ARMS against Flow (for the other arm), Summary+A-MEM (the endpoint split), observation
    masking and truncation at SWE 8k, per population of PAIRED_POPS with paired outcomes: non-inferiority at the
    P1 margin (3 points) on the 90% CI both repo-clustered and over items, exact McNemar, and the paired cost ratio
    as geometric mean and as ratio of means. Post hoc and exploratory: no Holm family; the endpoint Flow arm against
    Summary+A-MEM over all issues is P1 and stays in the endpoints table, as in secondary()."""
    b, out = ENDPOINT_BUDGET["swe"], []
    for m in PAIRED_ARMS:
        for o in [x for x in PAIRED_ARMS if x != m][:1] + [summary["swe"], "masking", "truncation"]:
            if m == "flow" and o == "flow_noevo":
                continue
            seen = set()
            for pop, f in PAIRED_POPS:
                if f is None and (m, o) == (flow["swe"], summary["swe"]):
                    continue  # = P1
                ids = _split(f) if f else None
                keep = (lambda r, ids=ids: r["item"] in ids) if ids else (lambda r: True)
                ya, yb, cl, ra, _ = paired(swe, (m, b), (o, b), keep=keep)
                items = frozenset(r["item"] for r in ra)
                if not ya or items in seen:  # e.g. the pooled row while an arm lacks rest385
                    continue
                seen.add(items)
                mc = S.mcnemar_exact(ya, yb)
                nc, ni = S.noninferiority(ya, yb, 0.03, cl), S.noninferiority(ya, yb, 0.03)
                verdict = lambda t: ("not evaluable" if not t["evaluable"]  # noqa: E731
                                     else "yes" if t["noninferior"] else "no")
                ca, cb, _ = swe_cost_pairs(swe, (m, b), (o, b), keep)
                out.append([f"{label(m)} vs {label(o)}",
                            f"{pop} ({MAIN_ARM_ROLE[pop]})" if m == "flow_noevo" else pop,
                            f"{len(ya)} ({len(set(cl))})",
                            pct(mean(ya)), pct(mean(yb)), f"{mc['n10']}/{mc['n01']}", pct(nc["est"]),
                            ci(nc["est"], nc["lo"], nc["hi"]), ci(ni["est"], ni["lo"], ni["hi"]),
                            f"{verdict(nc)} / {verdict(ni)}", pv(mc["p"]), ratios(log_cost_ratio(ca, cb))])
    return Table("swe_paired", f"Exploratory paired comparisons at SWE {budget(b)} (post hoc, not endpoints)",
                 ["Comparison", "Population", "n pairs (repos)", "first %", "second %", "discordant (first/second)",
                  "Δ (pts)", "90% CI, repos", "90% CI, items", "NI δ=3: repos / items", "McNemar p",
                  "cost ratio first/second (uncached)"], out,
                 ["Added after P1. Flow w/o evolution became the main arm on Eval-95 and LongMemEval "
                  "(results/decisions/2026-10-04_main_arm_noevo.md): Eval-95 is its selection set, rest385 its "
                  "check set, and its pooled rows mix the two. None of its comparisons was registered.",
                  f"{label(flow['swe'])} vs {label(summary['swe'])} over all issues is P1 and is in the endpoints "
                  f"table, not here. {label(flow['swe'])} vs Obs. masking over all issues is a registered "
                  "descriptive comparison, with its Holm-corrected p in the secondary table; reading it as "
                  "non-inferiority was not registered. Truncation is a reference and was never tested. No row is "
                  "an endpoint, and the p-values here are not corrected for multiplicity.",
                  "NI δ=3: the lower limit of the 90% paired bootstrap CI above -3 points, with repositories or items "
                  f"resampled; not evaluable below {S.MIN_DISCORDANT} discordant pairs.",
                  "Cost ratio over issues with both episodes (an episode that ended before any agent call leaves the "
                  "pairs): the geometric mean of the paired ratios with its 95% repo-clustered CI, then the ratio of "
                  "the mean tokens per issue, which long episodes dominate."])


def summary_split_dev(lme, swe, choice: dict) -> Table:
    out = []
    for bk, rows in (("lme", lme), ("swe", swe)):
        tab, pick = split_dev(rows, bk)
        s0, src = choice[bk]
        for b, s, n, k, c in tab:
            at = b == ENDPOINT_BUDGET[bk]
            mark = [x for x, ok in (("rule pick", at and s == pick), (f"endpoints ({src})", at and s == s0)) if ok]
            out.append([BENCH[bk], budget(b), f"0.{s}", n, pct(k / n if n else None),
                        "–" if _nan(c) else f"{c:,.0f}", "; ".join(mark) or "–"])
    return Table("summary_split_dev", "Summary+A-MEM split on dev (rule of results/decisions/"
                 "2026-09-24_summary_amem_split.md)",
                 ["Benchmark", "B", "split", "n dev", "dev acc %", "uncached tokens/item", "choice"], out,
                 ["Dev accuracy (LongMemEval: J1 on dev60; SWE: resolved on Dev20 without exclusions) and mean "
                  "uncached tokens per item, over the dev items where every split has an outcome at that budget. "
                  "Rule: highest dev accuracy at the endpoint budget (SWE 8k, LongMemEval 16k), ties -> lower mean "
                  "uncached cost, remaining ties -> 0.50. The split used by the endpoints comes from "
                  "configs/exp/endpoints.yaml; while that is null, from this rule on the loaded dev rows."])


def swe_cost_pairs(rows, a: tuple, b: tuple, keep=lambda r: True) -> tuple[list, list, list]:
    """SWE episodes of arms a and b on the same issues for a paired cost comparison: (rows a, rows b, issues left
    out). An episode that ended in a method error before any agent call succeeded (e.g. an issue whose fixed prompt
    exceeds the cap in every capped arm) is unresolved for accuracy, but its cost ratio is undefined, so it
    leaves the pairs; any other episode without agent ledger rows or tokens stops the build."""
    _, _, _, ca, cb = paired(rows, a, b, "attempt", keep)  # both episodes exist
    out = [x["iid"] for x, y in zip(ca, cb) if NO_CALL(x) or NO_CALL(y)]
    kept = [(x, y) for x, y in zip(ca, cb) if not (NO_CALL(x) or NO_CALL(y))]
    ca, cb = [x for x, _ in kept], [y for _, y in kept]
    bad = [f"{r['iid']} ({label(r['method'])})" for r in ca + cb if not r["cost"]["agent_calls"] or not r["cost"]["total"]]
    if bad:
        raise ValueError(f"SWE cost: {len(bad)} episodes have no agent ledger rows or zero tokens (ledger missing or "
                         f"not matched on the attempt id), so their log cost ratio is undefined: " + ", ".join(bad))
    return ca, cb, out


def log_cost_ratio(ra, rb, k="total") -> dict:
    """Paired bootstrap (95%, clustered) of log(cost a / cost b) over the pairs where both costs are positive, with
    the ratio of the mean costs over the same pairs (ratio_of_means). exp(est) is the geometric mean of the paired
    ratios: every pair weighs the same, so it is not the ratio of the mean costs, which long episodes dominate."""
    v = [(x, y) for x, y in zip(ra, rb) if (x["cost"][k] or 0) > 0 and (y["cost"][k] or 0) > 0]
    bs = S.paired_bootstrap([math.log(x["cost"][k] / y["cost"][k]) for x, y in v], [x["cluster"] for x, _ in v])
    return {**bs, "ratio_of_means": sum(x["cost"][k] for x, _ in v) / sum(y["cost"][k] for _, y in v) if v else math.nan}


def ratios(bs: dict) -> str:
    """'0.40× [0.35, 0.44] geo. mean; 0.22× of means' from log_cost_ratio()."""
    if not bs["n"]:
        return "–"
    return (f"{math.exp(bs['est']):.2f}× [{math.exp(bs['lo']):.2f}, {math.exp(bs['hi']):.2f}] geo. mean; "
            f"{bs['ratio_of_means']:.2f}× of means")


def endpoints(swe, lme, summary: dict, choice: dict, flow_src: dict) -> Table:
    """P1-P3 as a fixed sequence P1 -> P2 -> P3 at alpha = 0.05: an endpoint is confirmatory only if every earlier
    gating endpoint passed on its pre-registered population (expected_n); a gate stays unknown while an earlier
    endpoint is not loaded or provisional, and a result on a smaller loaded split is only provisional. P1 (SWE,
    Flow vs Summary+A-MEM): TOST non-inferiority, delta = 3 points, on the 90% repo-clustered bootstrap CI, and
    the 95% clustered CI of the mean paired log uncached-cost ratio below 0.
    P2 (LongMemEval, Flow vs Summary+A-MEM) mirrors P1 with questions as clusters. P3 (LongMemEval, Flow vs A-MEM
    retrieval-only): non-inferiority, delta = 5 points, on the 90% paired bootstrap CI. Non-inferiority needs
    S.MIN_DISCORDANT discordant pairs. LongMemEval endpoints use J1."""
    out, sb, lb, no_call, p1_repos = [], ENDPOINT_BUDGET["swe"], ENDPOINT_BUDGET["lme"], [], Counter()
    flow = {bk: m for bk, (m, _) in flow_src.items()}
    for bk, rows in (("swe", swe), ("lme", lme)):
        s, src = choice[bk]
        if any((r["method"], r["budget"]) == (flow[bk], ENDPOINT_BUDGET[bk]) for r in rows):
            assert any((r["method"], r["budget"]) == (summary[bk], ENDPOINT_BUDGET[bk]) for r in rows), (
                f"{BENCH[bk]}: Summary+A-MEM split {s} ({src}) has no rows at {budget(ENDPOINT_BUDGET[bk])}")

    logratio = log_cost_ratio  # pairs with both costs positive (P1 stops on a zero-token episode before this)

    def ratio_str(bs, both=False):
        return ("–" if not bs["n"] else f"{bs['est']:+.3f} [{bs['lo']:+.3f}, {bs['hi']:+.3f}]"
                + (f" = {ratios(bs)}" if both else ""))

    gates = []  # (endpoint, state) of the gating endpoints so far: pass, fail, not loaded or provisional

    def row(ep, comp, pop, bk, heldout, ya, yb, cl, test, crit, passed, stage, cost=None, split=False):
        n_exp, sub = expected_n(bk, heldout), loaded_split(swe if bk == "swe" else lme, bk)
        d95, mc = S.paired_bootstrap(np.asarray(ya, float) - np.asarray(yb, float), cl), S.mcnemar_exact(ya, yb)
        # SWE: issues resampled instead of repositories; on LongMemEval this would repeat the registered interval
        it = S.noninferiority(ya, yb, test["delta"]) if bk == "swe" and "delta" in test else {}
        prov = (f"provisional (not the {ep[:2]} population: {sub} only)" if sub else
                f"provisional ({n_exp - len(ya)} items without an outcome)" if len(ya) < n_exp else "")
        status = "not evaluable"
        if ya:
            gate = "descriptive"
            if stage == "gating":
                shut = [e for e, st in gates if st == "fail"]
                unknown = [f"{e} {st}" for e, st in gates if st in ("not loaded", "provisional")]
                gate = (f"descriptive (gate closed: {', '.join(shut)} did not pass)" if shut else
                        f"gate unknown ({', '.join(unknown)})" if unknown else "" if sub else "confirmatory")
            status = ", ".join(x for x in (gate, prov, "split not frozen" if split else "") if x)
        if stage == "gating":
            gates.append((ep, "not loaded" if not ya else "provisional" if prov else
                          "pass" if passed is True else "fail"))
        out.append([ep, comp, pop, f"{len(ya)}/{n_exp} ({len(set(cl))})",
                    f"{mc['n10'] + mc['n01']} ({mc['n10']}/{mc['n01']})" if ya else "–",
                    ci(d95["est"], d95["lo"], d95["hi"]),
                    ci(test.get("est"), test.get("lo"), test.get("hi")) if "delta" in test else "–",
                    ci(it.get("est"), it.get("lo"), it.get("hi")) if it else "–", crit,
                    pv(mc["p"]) if ya else "–", cost or "–",
                    "–" if not ya or (passed is None and test.get("evaluable", True)) else
                    "not evaluable (discordant < 5)" if passed is None else "pass" if passed else "fail", status])

    for pop, heldout, stage in (("all", False, "gating"), ("w/o Dev20", True, "sensitivity")):
        keep = (lambda r: not r["dev"]) if heldout else (lambda r: True)
        a, b = (flow["swe"], sb), (summary["swe"], sb)
        ya, yb, cl, *_ = paired(swe, a, b, keep=keep)
        ca, cb, left = swe_cost_pairs(swe, a, b, keep)
        if stage == "gating":
            no_call, p1_repos = no_call + left, Counter(cl)
        ni = S.noninferiority(ya, yb, 0.03, cl)
        lr, nb, ad = logratio(ca, cb, "total"), logratio(ca, cb, "no_build"), logratio(ca, cb, "cache_adjusted")
        row("P1" if stage == "gating" else "P1 (sensitivity)",
            f"{label(flow['swe'])} vs {label(summary['swe'])}, SWE {budget(sb)}",
            pop, "swe", heldout, ya, yb, cl, ni, "NI δ=3 & log cost CI < 0",
            (ni["noninferior"] and lr["n"] > 0 and lr["hi"] < 0) if ni["evaluable"] else None, stage,
            f"{ratio_str(lr, True)} (w/o build {ratio_str(nb)}; cache-adj. {ratio_str(ad)})",
            choice["swe"][1].startswith("split by rule"))
    for pop, heldout, stage in (("all", False, "gating"), ("held-out", True, "sensitivity")):
        keep = (lambda r: not r["dev"]) if heldout else (lambda r: True)
        a, b = (flow["lme"], lb), (summary["lme"], lb)
        ya, yb, cl, *_ = paired(lme, a, b, keep=keep)
        _, _, _, ca, cb = paired(lme, a, b, "qid", keep)  # cost pairs: both rows exist, judged or not
        ni, lr = S.noninferiority(ya, yb, 0.03), logratio(ca, cb, "total")
        row("P2" if stage == "gating" else "P2 (sensitivity)",
            f"{label(flow['lme'])} vs {label(summary['lme'])}, LME {budget(lb)}", pop, "lme", heldout, ya, yb, cl, ni,
            "NI δ=3 & log cost CI < 0", (ni["noninferior"] and lr["n"] > 0 and lr["hi"] < 0) if ni["evaluable"] else None,
            stage, ratio_str(lr, True), split=choice["lme"][1].startswith("split by rule"))
    for pop, heldout, stage in (("all", False, "gating"), ("held-out", True, "sensitivity")):
        keep = (lambda r: not r["dev"]) if heldout else (lambda r: True)
        ya, yb, cl, *_ = paired(lme, (flow["lme"], lb), ("amem", lb), keep=keep)
        ni = S.noninferiority(ya, yb, 0.05)
        row("P3" if stage == "gating" else "P3 (sensitivity)", f"{label(flow['lme'])} vs {label('amem')}, LME {budget(lb)}",
            pop, "lme", heldout, ya, yb, cl, ni, "NI δ=5", ni["noninferior"] if ni["evaluable"] else None, stage)
    return Table("endpoints", "Pre-registered endpoints P1-P3 (gatekeeping order)",
                 ["Endpoint", "Comparison", "Population", "n paired/expected (clusters)", "discordant (Flow/other)",
                  "Δ acc pts [95% CI]", "90% CI (TOST)", "90% CI, items", "criterion", "McNemar p",
                  "log cost ratio [95% CI]",
                  "result", "status"], out,
                 ["Fixed sequence P1 -> P2 -> P3 at α = 0.05: an endpoint is confirmatory only if every earlier gating one "
                  "passed on its pre-registered population. P1: TOST non-inferiority on the 90% repo-clustered "
                  "bootstrap CI (lower limit > -3 pts) and the 95% clustered CI of the log cost ratio below 0. P2: the "
                  "same rule on LongMemEval (questions as clusters; cost over questions where both arms logged "
                  "tokens). P3: non-inferiority on the 90% paired bootstrap CI (lower limit > -5 pts). LongMemEval "
                  "uses J1.",
                  f"Non-inferiority is not evaluable with fewer than {S.MIN_DISCORDANT} discordant pairs (with none, "
                  "the bootstrap CI is [0, 0] and passes any margin); the endpoint then does not pass.",
                  "n paired/expected: items with an outcome in both arms / items of the pre-registered population "
                  "(docs/protocol/experiment_protocol.md), whatever runs are loaded: all 500 instances (SWE) or "
                  "questions (LongMemEval), minus the SWE exclusions (gold-patch failures, two instances without an "
                  "image) and, for w/o Dev20 / held-out, the dev split. 'provisional (not the Pk population: S "
                  "only)': every loaded item of the benchmark is in split S (Dev20, Eval-100 or dev60), so the row is "
                  "not the endpoint's result; 'provisional (k items without an outcome)': items of the population "
                  "without an outcome in an arm (no result row, errored, ungraded or not judged). 'gate unknown': an "
                  "earlier gating endpoint is not loaded or provisional; 'gate closed': one did not pass.",
                  "Clusters: repositories (SWE), questions (LongMemEval). '90% CI, items' (SWE only) resamples "
                  "issues instead of repositories: a sensitivity interval, not the registered test."
                  + (" P1's paired issues span {} repositories, {} holding {} of {}, so the clustered bootstrap can "
                     "understate the spread; read the result against both.".format(
                         len(p1_repos), *p1_repos.most_common(1)[0], p1_repos.total()) if p1_repos else "")
                  + " On LongMemEval each question is its own "
                  "cluster, so the registered interval already resamples items. Log cost ratio = mean over items of "
                  "log(cost Flow / cost other), uncached tokens; its exponential is the geometric mean of the paired "
                  "ratios ('geo. mean'), next to the ratio of the mean costs ('of means'), which long episodes "
                  "dominate. SWE: over issues with both episodes; an episode that "
                  "ended in a method error before any agent call succeeded (e.g. its fixed prompt exceeds the cap) is "
                  "unresolved and leaves the cost pairs, and any other episode without agent "
                  "ledger rows or tokens stops the build; w/o build and cache-adjusted are descriptive. LongMemEval: "
                  "over questions where both arms logged tokens. Sensitivity rows (SWE without Dev20, LongMemEval "
                  "held-out 440) are descriptive."
                  + (f" P1 cost pairs left out for that reason: {len(no_call)} ({', '.join(no_call)})." if no_call
                     else ""),
                  "Summary+A-MEM split: " + "; ".join(
                      f"{BENCH[bk]} {'–' if s is None else f'0.{s}'} ({src})" for bk, (s, src) in choice.items())
                  + " (table summary_split_dev). Flow arm: " + "; ".join(
                      f"{BENCH[bk]} {m} ({src})" for bk, (m, src) in flow_src.items()) + "."])


# ---------------------------------------------------------------- checkpoint diagnostic (recall branch)

# sha of a tool output that shows nothing, as the question generator hashes outputs (recall_questions._sha)
EMPTY_OUTPUT = {hashlib.sha1(f"<returncode>{c}</returncode>\n<output>\n</output>".encode()).hexdigest()[:16]
                for c in range(256)}


def load_diag(runs) -> list[dict]:
    """Checkpoint-diagnostic rows (scripts/run_diag.py) without a method error, last row per (checkpoint, method).
    An answer to a membership question with gold "yes" whose display commands all printed nothing (by the output
    hashes in the run's frozen questions.jsonl) carries empty_display."""
    rows, src = {}, {}
    for rid in runs:
        d = ROOT / "artifacts/runs/diag" / rid
        empty = {q["qid"] for q in read_rows(d / "questions.jsonl", ("qid",)) if q["template"] == "membership"
                 and q["gold"] == "yes" and all(x["sha"] in EMPTY_OUTPUT for x in q["sources"])}
        for r in read_rows(d / "results.jsonl", ("cid", "method")):
            k = (r["cid"], r["method"])
            _claim(src, k, rid)
            rows[k] = {**r, "answers": [{**a, "empty_display": a["qid"] in empty} for a in r["answers"]]}
    return [r for r in rows.values() if not r.get("error")]


def diag(rows) -> Table:
    """Per method, pooled over checkpoints: recall accuracy (every template except membership and abstain) with its
    Wilson 95% CI, and on the facts outside truncation's context (its answers' fact_visible); file-read membership
    hit / false-alarm rate and balanced accuracy, also without "yes" questions on empty displays (empty_display);
    abstention false alarms."""
    out, g = [], defaultdict(list)
    for r in rows:
        g[r["method"]].append(r)
    recall = lambda a: a["template"] not in ("membership", "abstain")  # noqa: E731
    seen = {a["qid"]: a["fact_visible"] for r in g.get("truncation", []) for a in r["answers"]
            if recall(a) and a.get("fact_visible") is not None}
    member = lambda ans, gold, drop=False: [a["said_yes"] for a in ans if a["template"] == "membership"  # noqa: E731
                                            and a["gold"] == gold and not (drop and a["empty_display"])]
    for m in sorted(g, key=lambda m: arm_key((m, 0))):
        ans = [a for r in g[m] for a in r["answers"]]
        rec = [a["correct"] for a in ans if recall(a)]
        hid = [a["correct"] for a in ans if recall(a) and seen.get(a["qid"]) is False]
        hit, fa, hit2 = mean(member(ans, "yes")), mean(member(ans, "no")), mean(member(ans, "yes", True))
        ab = [not a["correct"] for a in ans if a["template"] == "abstain"]
        out.append([label(m), len(g[m]), f"{sum(rec)}/{len(rec)}", ci(*S.wilson(sum(rec), len(rec))),
                    f"{sum(hid)}/{len(hid)}" if seen else "–", pct(hit), pct(fa), num((hit + 1 - fa) / 2),
                    num((hit2 + 1 - fa) / 2), f"{sum(ab)}/{len(ab)}", ktok(mean([r["context_tokens"] for r in g[m]]))])
    n_empty = len({a["qid"] for r in rows for a in r["answers"] if a["empty_display"]})
    return Table("diag", "Checkpoint diagnostic: recall of earlier work from each method's rendered context",
                 ["Method", "checkpoints", "recall", "recall % [95% CI]", "recall, facts not in truncation's context",
                  "membership hit %", "false alarm %", "balanced acc.", "balanced acc. w/o empty displays",
                  "abstain false alarms", "mean context"], out,
                 ["Questions are generated by a program from the uncapped trajectory before any render and hashed. "
                  "Each fact rests only on steps that every memory method has already written to memory, so none of "
                  "them can read it off its recent window; truncation and masking keep windows of their own, which "
                  "can still show it."
                  + (f" Truncation's context shows {sum(seen.values())} of the {len(seen)} recall facts; the column "
                     f"'facts not in truncation's context' scores each method on the other "
                     f"{len(seen) - sum(seen.values())} only." if seen else ""),
                  (f"Membership: {n_empty} \"yes\" questions rest only on display commands that printed nothing "
                   f"(the generator counts them as displays); 'w/o empty displays' leaves them out of the hit rate."
                   if n_empty else "Membership: no \"yes\" question rests only on empty displays, so 'w/o empty "
                   "displays' equals balanced acc."),
                  "Checkpoints where the method's render, reader or judge call failed (error rows) are left out of "
                  "that method's row; the checkpoints column counts the rest. Descriptive."])


def build_all(swe_runs, lme_runs, summary_split: dict | None = None, flow_arm: dict | None = None,
              diag_runs=()) -> list[Table]:
    """summary_split / flow_arm: optional per-benchmark overrides {"swe": ..., "lme": ...} of the endpoint config
    (Summary+A-MEM split 30|50|70, Flow method name)."""
    swe, fail = load_swe(swe_runs) if swe_runs else ([], {"errors": {}, "missing": {}})
    lme = load_lme(lme_runs) if lme_runs else []
    choice = summary_choice(lme, swe, summary_split)
    summary = {bk: "summary_amem" if s is None else f"summary_amem_s{s}" for bk, (s, _) in choice.items()}
    flow_src = flow_choice(lme, swe, flow_arm)
    flow = {bk: m for bk, (m, _) in flow_src.items()}
    return [endpoints(swe, lme, summary, choice, flow_src), summary_split_dev(lme, swe, choice), lme_accuracy(lme),
            lme_types(lme, "y"), lme_types(lme, "y2"), lme_cost(lme), swe_main(swe), swe_cost(swe),
            swe_failures(swe, fail), ablations(lme, swe, flow), secondary(lme, swe, summary, flow),
            swe_paired(swe, summary, flow),
            diag(load_diag(diag_runs) if diag_runs else [])]
