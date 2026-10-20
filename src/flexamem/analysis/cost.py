"""Token cost per SWE episode and per LongMemEval question from the phase-tagged ledgers (artifacts/ledger/), and
the offline prompt-cache simulation for SWE agent calls.

The cost unit is tokens: prompt + completion (completion includes reasoning) summed over calls; embedding rows
(local model) count as calls with no tokens. Cached memory calls keep their original usage (`cache_hit`), so every
total is the cost as if no call had been cached ("uncached"). Memory-side cost is everything except the agent
(SWE) or reader (LongMemEval) calls; build cost is the memory calls made while writing (SWE stage `memory_write`,
LongMemEval stage `build`), render cost those made while rendering (`memory_render` / `render`; for Sculptor on SWE
also its context tools, `memory_tool`).

SWE rows are matched on the result row's `attempt`, so rows of crashed or rerun attempts are not counted in the
episode; they are reported separately (`rerun_tokens`: other attempts of the same key in the result row's run).
LongMemEval builds are matched on `build_ids` across all LongMemEval ledgers (a build is reused by later runs, and
each method is charged its full build); render and read rows on (run_id, qid, method, budget) and the result row's
`attempt`, rows of other attempts (a question rerun after an API or disk error) being reported as `rerun_tokens`.
Result rows written before the attempt id was logged fall back to the key alone (`attempt_matched` False), which
counts an earlier attempt's rows too."""
import json
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

from flexamem.adapters.mswea import EARLIER_HEADER, TOOLS
from flexamem.llm.client import ROOT
from flexamem.llm.tokens import MSG_OVERHEAD, count, count_message, count_messages
from flexamem.memory.segmenter import parse_steps, swe_units

LEDGER = ROOT / "artifacts/ledger"
PHASES = ("agent", "write", "link", "merge", "summarize", "gist", "lookup", "filter", "embed", "reader")
CACHED_WEIGHT = 0.1                  # billed share of a cached prompt token (OpenAI GPT-5 family: 90% discount)
CACHE_MIN, CACHE_BLOCK = 1024, 128   # OpenAI prompt caching: prefixes of >= 1024 tokens, in 128-token steps


def read_ledger(path: Path) -> list[dict]:
    out = []
    for line in Path(path).read_text().splitlines() if Path(path).exists() else []:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:  # a partial last line of a ledger still being written
            pass
    return out


def tokens(r: dict) -> int:
    """Billed tokens of a ledger row. Gemini rows logged before the client normalised usage (llm/tokens.billed_usage,
    2026-10-06) left reasoning out of completion_tokens; their reasoning_tokens are added back. Claude rows logged
    before it under-report the prompt and cannot be repaired (prompt_tokens_reported marks the repaired later rows)."""
    extra = (r.get("reasoning_tokens") or 0) if str(r.get("model", "")).startswith("gemini") and not r.get("usage_v") else 0
    return (r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0) + extra


def summarize(rows: list[dict]) -> dict:
    """Totals and the per-phase breakdown of a set of ledger rows."""
    phase, calls = defaultdict(int), defaultdict(int)
    for r in rows:
        phase[r["phase"]] += tokens(r)
        calls[r["phase"]] += 1
    return {"total": sum(phase.values()), "calls": sum(v for k, v in calls.items() if k != "embed"),
            "phase": dict(phase), "phase_calls": dict(calls)}


# ---------------------------------------------------------------- SWE

def swe_ledger(run_ids) -> tuple[dict, dict]:
    """(rows by attempt, attempts by (run_id, iid, method, budget)) over artifacts/ledger/swe_<run>.jsonl."""
    by_attempt, by_key = defaultdict(list), defaultdict(set)
    for rid in run_ids:
        for r in read_ledger(LEDGER / f"swe_{rid}.jsonl"):
            if r.get("attempt"):
                by_attempt[r["attempt"]].append(r)
                by_key[(rid, r["iid"], r["method"], r["budget"])].add(r["attempt"])
    return by_attempt, by_key


def episode_layout(traj: dict, tools: list | None = None) -> dict:
    """Token counts and boundaries of a saved trajectory: pin (system + task + the tool schemas + reply priming), tok
    (tokens per message), step_start and unit_start (first message index of every step and unit, unit_start ending
    with one past the last step message; a finished trajectory's trailing exit message belongs to no step, so there
    that is len(messages) - 1; unit starts never move, so the full trajectory gives every start), and ends: per
    context-log record, the length k of the raw message prefix the call was built from (the agent saw
    policy.build(messages[:k]); messages[k] is its reply, except for a format_error record, whose rejected reply was
    discarded: there messages[k] is the format-error message that replaced it, and the next record starts at k + 1)."""
    msgs, log = traj["messages"], traj["info"]["context"]["log"]
    pin = count_messages(msgs[:2], tools or TOOLS)  # AppWorld episodes pass their python tool
    tok = [count_message(m) for m in msgs]
    cum = [pin]
    for t in tok[2:]:
        cum.append(cum[-1] + t)
    steps = parse_steps(msgs[2:])
    step_start, unit_start = [2], [2]
    for s in steps:
        step_start.append(step_start[-1] + len(s))
    for u in swe_units(steps):
        unit_start.append(unit_start[-1] + len(u.messages))
    step_start.pop()
    ends = []
    for x in log:
        j = bisect_left(cum, x["raw_tokens"])
        assert j < len(cum) and cum[j] == x["raw_tokens"], f"step {x['step']}: no message prefix has {x['raw_tokens']} tokens"
        ends.append(2 + j)
    return {"pin": pin, "tok": tok, "step_start": step_start, "unit_start": unit_start, "ends": ends}


def agent_prompts(traj: dict) -> list[tuple[int, list[tuple]]]:
    """Reconstructs the layout of every agent prompt of an episode from the saved trajectory (complete messages)
    and the per-call context log: (step, blocks), a block being (identity, tokens). Identities are trajectory
    message indices, masked observations, the memory message (identified by the sha1 of its text, or by its render
    record in logs that predate `memory_sha`) and one opaque block per call for anything shortened by the
    last-resort shrink. The pinned block holds system, task, the tool
    schemas sent at that call (`pin_tokens`; Sculptor adds its context tools once it compacts) and reply priming."""
    msgs, log = traj["messages"], traj["info"]["context"]["log"]
    lay = episode_layout(traj)
    pin, tok, step_start, unit_start = lay["pin"], lay["tok"], lay["step_start"], lay["unit_start"]
    out = []
    for x, k in zip(log, lay["ends"]):  # the raw prompt is msgs[:k]
        p = x.get("pin_tokens", pin)  # logs before pin_tokens: every policy sent TOOLS only
        head, first = [("pin", p)], 2
        if x["mode"] == "pass":
            body = [(("m", i), tok[i]) for i in range(2, k)]
        elif "recent_start" in x:  # memory policies: [system, task, memory message, recent units]
            first = unit_start[x["recent_start"]]
            body = [(("m", i), tok[i]) for i in range(first, k)]
            mem = (x["tokens"] - p - sum(t for _, t in body) if not x.get("shrunk_observations")
                   else x.get("header_tokens", count(EARLIER_HEADER)) + x["memory_tokens"] + MSG_OVERHEAD)
            if x["memory_tokens"]:
                head.append((("mem", x.get("memory_sha") or json.dumps(
                    [x.get("written"), x["memory_tokens"], x.get("render")], sort_keys=True)), mem))
        else:  # truncation / masking: the whole steps kept after `dropped_steps`, observations masked by window
            starts = [s for s in step_start if s < k]
            ends = starts[1:] + [k]
            first = starts[x.get("dropped_steps", 0)] if x.get("dropped_steps", 0) < len(starts) else k
            w = x.get("window", len(starts))
            body = []
            for n, (s, e) in enumerate(zip(starts, ends)):
                for i in range(max(s, first), e):
                    if x["mode"] == "masking" and n < len(starts) - w and msgs[i].get("role") in ("tool", "user"):
                        mm = {**msgs[i], "content": f"Previous {len((msgs[i].get('content') or '').splitlines())} "
                                                    f"lines omitted for brevity."}
                        body.append((("masked", i), count_message(mm)))
                    else:
                        body.append((("m", i), tok[i]))
        if x.get("shrunk_observations"):
            body = [(("shrunk", x["step"]), x["tokens"] - sum(t for _, t in head))]
        out.append((x["step"], head + body))
    return out


def simulate_cache(traj: dict) -> tuple[dict[int, int], str]:
    """(step -> simulated cached prompt tokens of that agent call, mode). Offline longest-common-prefix simulation
    over consecutive agent prompts of one episode. Assumptions: the provider caches the previous prompt of the
    episode and nothing else (no cross-episode hits, no eviction or routing misses between consecutive calls);
    prefixes are compared at block level (a message, or the memory message), except that a changed memory message
    still shares its role header and its unchanged start (`memory_prefix_tokens`, the tokens of the longest common
    character prefix with the previous memory message; may over-count by the token straddling the point where the
    texts diverge); a hit needs >= 1024 common tokens and is rounded down to 128-token steps; token counts are
    ours (tiktoken). Mode "exact" (under these assumptions) or, for logs that predate `memory_prefix_tokens`,
    "lower_bound": a memory message whose render record changed breaks the prefix at its start. In both modes a
    step cut by the last-resort shrink is one opaque block, so it and the step after it are lower bounds."""
    log = traj["info"]["context"]["log"]
    exact = all("memory_prefix_tokens" in x for x in log if "recent_start" in x)
    out, prev = {}, []
    for x, (step, blocks) in zip(log, agent_prompts(traj)):
        lcp = 0
        for a, b in zip(prev, blocks):
            if a != b:
                if exact and a[0][0] == b[0][0] == "mem":  # role header (all overhead but the end token) + prefix
                    lcp += min(MSG_OVERHEAD - 1 + x["memory_prefix_tokens"], a[1], b[1])
                break
            lcp += b[1]
        out[step] = lcp // CACHE_BLOCK * CACHE_BLOCK if lcp >= CACHE_MIN else 0
        prev = blocks
    return out, "exact" if exact else "lower_bound"


def swe_cost(row: dict, by_attempt: dict, by_key: dict, traj: dict | None = None) -> dict:
    """Cost of one SWE episode (result row). `cache_adjusted` bills simulated cached agent-prompt tokens at
    CACHED_WEIGHT (memory calls stay uncached; None without a trajectory); `no_build` = agent + render-time
    memory calls; `api_cached` = cached tokens reported by the API for the agent calls (for comparison);
    `sim_mode` = simulate_cache's mode; `agent_calls` = agent ledger rows of the attempt (0: cost unknown)."""
    rows = by_attempt.get(row["attempt"], [])
    s = summarize(rows)
    agent = [r for r in rows if r["phase"] == "agent"]
    render = sum(tokens(r) for r in rows if r.get("stage") in ("memory_render", "memory_tool"))
    a_tok = sum(map(tokens, agent))
    cached = adj = mode = None
    if traj is not None:
        sim, mode = simulate_cache(traj)
        cached = sum(min(sim.get(r.get("step"), 0), r["prompt_tokens"]) for r in agent)
        adj = s["total"] - (1 - CACHED_WEIGHT) * cached
    others = by_key.get((row["run_id"], row["iid"], row["method"], row["budget"]), set()) - {row["attempt"]}
    return {**s, "agent": a_tok, "agent_calls": len(agent), "memory": s["total"] - a_tok,
            "build": s["total"] - a_tok - render,
            "render": render, "no_build": a_tok + render, "cache_adjusted": adj, "sim_cached": cached, "sim_mode": mode,
            "api_prompt": sum(r["prompt_tokens"] for r in agent), "api_cached": sum(r["cached_tokens"] for r in agent),
            "reruns": len(others), "rerun_tokens": sum(tokens(r) for a in others for r in by_attempt[a])}


# ---------------------------------------------------------------- LongMemEval

def lme_ledger(run_ids) -> tuple[dict, dict]:
    """(build rows by build_id from every LongMemEval ledger, render/read rows by (run_id, qid, method, budget))."""
    builds, query = defaultdict(list), defaultdict(list)
    run_ids = set(run_ids)
    for p in sorted(LEDGER.glob("lme_*.jsonl")):
        if p.stem.endswith("_judge"):
            continue
        for r in read_ledger(p):
            if r.get("stage") == "build":
                builds[r["build_id"]].append(r)
            elif r.get("stage") in ("render", "read") and r.get("run_id") in run_ids:
                query[(r["run_id"], r["qid"], r["method"], r["budget"])].append(r)
    return builds, query


def lme_cost(row: dict, builds: dict, query: dict) -> dict:
    """Cost of one (question, method, budget): its builds (full cost, uncached) plus render and reader calls.
    `api_prompt` is the reader prompt as billed by the API (mean over parse retries, or over Sculptor's turns; cache
    hits excluded, None if every reader call was one), to cross-check our count. `build_failed`: a method error in the build left the row
    without build ids, so its build cost is not counted.
    Render and reader rows are those of the result row's attempt; rows of the same key without it (earlier
    attempts, including any logged before the attempt id existed) are `rerun_tokens`. A result row without an
    attempt id keeps every row of its key (`attempt_matched` False)."""
    b = [r for bid in (row.get("build_ids") or {}).values() for r in builds.get(bid, [])]
    q, other = query.get((row["run_id"], row["qid"], row["method"], row["budget"]), []), []
    if row.get("attempt"):
        other = [r for r in q if r.get("attempt") != row["attempt"]]
        q = [r for r in q if r.get("attempt") == row["attempt"]]
    sb, s = summarize(b), summarize(b + q)
    reader = [r for r in q if r["phase"] == "reader" and not r.get("cache_hit")]
    return {**s, "build": sb["total"], "build_calls": sb["calls"], "query": s["total"] - sb["total"],
            "memory": s["total"] - s["phase"].get("reader", 0),
            "missing_builds": sum(bid not in builds for bid in (row.get("build_ids") or {}).values()),
            "build_failed": "build_ids" not in row and bool(row.get("error")),
            "attempt_matched": bool(row.get("attempt")), "rerun_tokens": sum(map(tokens, other)),
            "api_prompt": sum(r["prompt_tokens"] for r in reader) / len(reader) if reader else None}
