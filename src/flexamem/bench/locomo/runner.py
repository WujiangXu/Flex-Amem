"""LoCoMo runner (decision 2026-10-11_locomo_afm). LongMemEval's machinery with one memory per conversation.

Each conversation's memory is built once over its sessions, before and without its questions: every build call runs
under the client's guard against all of that conversation's question texts (those that do not occur in the
conversation itself). The build kinds and the arms' factories are LongMemEval's (bench/longmemeval/runner.py:
METHODS, RETRIEVAL, _build, under that build version's hash), stored per conversation in artifacts/locomo_builds/.
Methods outside LongMemEval's build kinds (ONLINE, e.g. AFM) are built online, one fresh instance per conversation
writing all its units, and kept in memory.

Every question then renders each arm under each budget and is read with bench/locomo/reader.py: the template in
bench/locomo/prompts.py (LongMemEval's Chain-of-Note prompt unless replaced), the conversation's last session date as
the current date, and LongMemEval's budget semantics (B bounds the whole reader prompt at RESERVE x B; the memory gets
what the rest of the prompt leaves, counted on the template actually set). A conversation's questions share its
builds, so their renders run under the conversation's lock (renders mutate shared caches: the store's embedding matrix,
the piece index, merge and compression caches); reader calls run in parallel. Memory is never shared across
conversations.

Failures as in LongMemEval: an API, cache or disk failure fails the job and is retried on resume; a method failure
becomes a row with an `error` field and no answer, scored as wrong."""
import contextvars
import copy
import hashlib
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from flexamem.bench.locomo import reader
from flexamem.bench.longmemeval.runner import BUILD_GROUPS, METHODS, RESERVE, RETRIEVAL, _build, build_version, is_infra
from flexamem.interface import Query
from flexamem.llm.client import ROOT, forbid_text, get_llm
from flexamem.llm.embed import embed_one
from flexamem.llm.ledger import ledger_context
from flexamem.memory.segmenter import lme_units
from flexamem.memory.store import NoteStore
from flexamem.methods.afm import CALIBRATED, AFMMethod
from flexamem.render import format as F
from flexamem.render.renderer import render_recent_raw
from flexamem.types import Note

BUILD_ROOT = ROOT / "artifacts/locomo_builds"
# FlexA-MEM with unit recovery, timeline only and piece recovery (LongMemEval's settings: history 0.3, link discount
# 0.5, excerpts of 300 tokens), zero-build BM25 pieces, A-MEM with and without fit-or-degrade, Summary+A-MEM at the split
# LongMemEval froze (configs/exp/endpoints.yaml: 50)
ARMS = ("flow_s30_ld50_noevo_zoom", "flow_s30_ld50_noevo_noexp", "flow_s30_ld50_noevo_bm25", "rag_pieces", "amem",
        "amem_zoom", "summary_amem_s50", "afm")
REFERENCES = ("truncation", "full")
# method -> factory() of a MemoryMethod built online per conversation (write(all units)): AFM (Cruz 2025), reproduced
# from the paper (methods/afm.py)
ONLINE: dict = {"afm": lambda: AFMMethod(),
                **{k: lambda t=t: AFMMethod(tau_high=t[0], tau_mid=t[1]) for k, t in CALIBRATED.items()}}


def locomo_build_version() -> str:
    """LongMemEval's build version (memory code, prompts, memory model) plus LoCoMo's loader, which decides the units'
    text."""
    h = hashlib.sha256((build_version() + (ROOT / "src/flexamem/bench/locomo/data.py").read_text()).encode())
    return h.hexdigest()[:12]


def units_of(conv: dict):
    return lme_units(conv["sessions"], starter=conv["speaker_a"])


def known(m: str) -> bool:
    return m in METHODS or m in RETRIEVAL or m in REFERENCES or m in ONLINE


class Prepared:
    """One conversation's memory: units, the raw store, builds (LongMemEval kinds and ONLINE instances), the questions'
    query embeddings, and the jobs still to run."""

    def __init__(self, conv, todo, units, raw, builds, build_ids, failed, queries):
        self.conv, self.todo, self.units, self.raw = conv, todo, units, raw
        self.builds, self.build_ids, self.failed, self.queries = builds, build_ids, failed, queries
        self.lock = threading.Lock()


def prepare(conv: dict, methods: list[str], budgets: list[int], results, run_id: str, build_dir: Path):
    """Builds what the conversation's remaining jobs need; None when nothing is left."""
    cid = conv["cid"]
    todo = [(q, m, B) for q in conv["questions"] for m in methods for B in (budgets if m != "full" else [0])
            if not results.has(qid=q["qid"], method=m, budget=B)]
    if not todo:
        return None
    units = units_of(conv)
    haystack = "\n".join(u.text for u in units)
    guard = tuple(t for t in conv["guard"] if t.strip() not in haystack)  # every question, whatever --limit took
    left = {m for _, m, _ in todo}
    kinds = sorted({k for m in left if m in METHODS for k in METHODS[m][0]})
    groups = [[k for k in g if k in kinds] for g in BUILD_GROUPS if set(g) & set(kinds)]
    assert sum(map(len, groups)) == len(kinds), kinds
    builds, build_ids, failed = {}, {}, {}

    def build(ks):
        out = {}
        for k in ks:
            try:
                with forbid_text(guard):
                    out[k] = _build(k, cid, units, build_dir, run_id)
            except Exception as e:
                out[k] = e
        return out

    def build_online(m):
        bid = uuid.uuid4().hex[:12]  # this build's ledger rows: a resume rebuilds (from the cache) and must not count twice
        try:
            with forbid_text(guard), ledger_context(stage="build", build=m, qid=cid, run_id=run_id, build_id=bid):
                inst = ONLINE[m]()
                inst.write(units)
            return {m: (inst, bid)}
        except Exception as e:
            return {m: e}

    online = sorted(m for m in left if m in ONLINE)
    with ThreadPoolExecutor(max_workers=max(1, len(groups) + len(online))) as ex:
        futs = [ex.submit(contextvars.copy_context().run, build, g) for g in groups]
        futs += [ex.submit(contextvars.copy_context().run, build_online, m) for m in online]
    for k, r in (kr for f in futs for kr in f.result().items()):
        if not isinstance(r, Exception):
            builds[k], build_ids[k] = r
        elif is_infra(r):
            raise r
        else:
            failed[k] = f"build {k}: {type(r).__name__}: {r}"
    queries = {}
    for q in conv["questions"]:
        if any(x is q for x, _, _ in todo):
            with ledger_context(stage="query_embed", qid=q["qid"], run_id=run_id):
                queries[q["qid"]] = Query(text=q["question"], emb=embed_one(q["question"], phase="embed"),
                                          date=q["question_date"])
    raw = NoteStore()
    for u in units:
        raw.add(Note.from_unit(u))
    return Prepared(conv, todo, units, raw, builds, build_ids, failed, queries)


def _render(p: Prepared, q: dict, m: str, cap: int | None, run_id: str, B: int, attempt: str) -> tuple[str, dict]:
    with ledger_context(stage="render", method=m, budget=B, qid=q["qid"], run_id=run_id, attempt=attempt):
        if m == "full":
            return F.join([F.raw_block(n) for n in p.raw.notes.values()]), {}
        query, allowance = p.queries[q["qid"]], reader.history_allowance(q, cap)
        if m == "truncation":
            r = render_recent_raw(p.raw, allowance)
        elif m in RETRIEVAL:
            r = RETRIEVAL[m](p.raw).render(query, allowance)
        elif m in ONLINE:
            r = p.builds[m].render(query, allowance)
        else:
            blds = {k: copy.copy(p.builds[k]) if k == "summary" else p.builds[k] for k in METHODS[m][0]}
            r = METHODS[m][1](blds).render(query, allowance)
        return r.text, {**r.stats, "allowance": allowance}


def run_job(p: Prepared, q: dict, m: str, B: int, results, run_id: str, build_dir: Path) -> None:
    attempt = uuid.uuid4().hex[:10]  # tags this attempt's ledger rows, so a rerun is not double-counted
    cap = int(B * RESERVE) if B else None
    deps = METHODS[m][0] if m in METHODS else (m,) if m in ONLINE else ()
    base = {"qid": q["qid"], "cid": q["cid"], "category": q["category"], "qtype": q["qtype"], "method": m,
            "budget": B, "cap": cap, "n_units": len(p.units), "run_id": run_id, "attempt": attempt,
            "reader_model": get_llm().role("reader")["model"], "memory_model": get_llm().role("memory")["model"],
            "build_version": build_dir.name, "build_ids": {k: p.build_ids[k] for k in deps if k in p.build_ids}}
    bad = [p.failed[k] for k in deps if k in p.failed]
    if bad:
        results.add({**base, "answer": None, "error": bad[0]})
        return
    try:
        with p.lock:
            history, stats = _render(p, q, m, cap, run_id, B, attempt)
        with ledger_context(stage="read", method=m, budget=B, qid=q["qid"], run_id=run_id, attempt=attempt):
            answer, n_prompt = reader.read(q, history, cap)
        row = {"prompt_tokens": n_prompt, "answer": answer, "error": None, "render": stats}
    except Exception as e:
        if is_infra(e):
            raise
        row = {"answer": None, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()}
    results.add({**base, **row})


def run(convs: list[dict], methods: list[str], budgets: list[int], results, run_id: str, workers: int = 32) -> list:
    """Conversations are prepared in parallel; each one's jobs start as soon as it is ready. Returns the failed
    conversations and jobs (API or disk failures; rerun to resume)."""
    assert all(known(m) for m in methods), [m for m in methods if not known(m)]
    build_dir = BUILD_ROOT / locomo_build_version()
    print(f"builds: {build_dir}", flush=True)
    errors, jobs = [], {}
    with ThreadPoolExecutor(max_workers=workers) as pool, ThreadPoolExecutor(max_workers=max(1, len(convs))) as prep:
        pf = {prep.submit(contextvars.copy_context().run, prepare, c, methods, budgets, results, run_id, build_dir):
              c["cid"] for c in convs}
        for f in as_completed(pf):
            try:
                p = f.result()
            except Exception:
                errors.append(pf[f])
                print(f"{pf[f]} build FAILED\n{traceback.format_exc()}", flush=True)
                continue
            if p is None:
                continue
            print(f"{pf[f]}: built, {len(p.todo)} jobs", flush=True)
            for q, m, B in p.todo:
                jobs[pool.submit(contextvars.copy_context().run, run_job, p, q, m, B, results, run_id, build_dir)] = (
                    q["qid"], m, B)
        for i, f in enumerate(as_completed(jobs), 1):
            try:
                f.result()
            except Exception:
                errors.append(jobs[f])
                print(f"{jobs[f]} FAILED\n{traceback.format_exc()}", flush=True)
            if i % 200 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] jobs done", flush=True)
    return errors


def results_path(run_id: str) -> Path:
    return ROOT / "artifacts/runs/locomo" / run_id / "results.jsonl"
