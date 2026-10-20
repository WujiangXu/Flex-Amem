"""LongMemEval runner. Memory is built online over the haystack without seeing the question (every
build call runs under a guard that rejects any prompt containing the question text); each build is
cached per question (one Flow build serves Flow and three ablations, one A-MEM build serves both A-MEM
arms). The question then drives rendering under each budget and one reader call.

Builds live in a directory named by a hash of the code and prompts they depend on and of the memory
model, so a changed prompt or builder never reuses a stale build. Failures of a method (as opposed to
API or disk failures, which fail the question and are retried on resume) become result rows with an
`error` field and no answer; they are scored as wrong."""
import contextlib
import contextvars
import copy
import fcntl
import gzip
import hashlib
import json
import os
import sqlite3
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai

from flexamem.bench.longmemeval import reader
from flexamem.interface import Query
from flexamem.llm.client import ROOT, LLMTransientError, forbid_text, get_llm
from flexamem.llm.embed import embed_one
from flexamem.llm.ledger import get_ledger, ledger_context
from flexamem.memory.segmenter import lme_units
from flexamem.memory.store import NoteStore
from flexamem.methods.afm import CALIBRATED, AFMMethod
from flexamem.methods.amem import AMemRetrieval, SummaryAMem, amem_build, summarize
from flexamem.methods.flow import FLOW_VARIANTS, RAW_SHARE, ZOOM, FlowMethod
from flexamem.methods.memgas import MemGASMethod
from flexamem.methods.hybrid import TimelinePieces
from flexamem.methods.rag import PieceRetrieval, RawRetrieval
from flexamem.methods.readagent import ReadAgentMethod
from flexamem.methods.sculptor import SculptorMethod
from flexamem.render import format as F
from flexamem.render.renderer import candidates, render_recent_raw
from flexamem.types import Note

BUILD_SOURCES = ("memory", "methods", "llm/embed.py", "llm/tokens.py", "types.py")


def build_version() -> str:
    h = hashlib.sha256(json.dumps(get_llm().role("memory"), sort_keys=True).encode())
    src = ROOT / "src/flexamem"
    for rel in BUILD_SOURCES:
        for f in sorted((src / rel).rglob("*.py")) if (src / rel).is_dir() else [src / rel]:
            h.update(f.relative_to(src).as_posix().encode() + f.read_bytes())
    return h.hexdigest()[:12]


BUILD_ROOT = ROOT / "artifacts/lme_builds"
MAIN_BUILDS = "12ee4448965f"  # the P0 build version of the main arm (p0_lme_noevo), which A-MEM Zoom renders
RESERVE = 0.95  # B bounds the prompt; 5% is kept as slack for tokenizer mismatch with the API


# method -> (builds it needs, factory(builds) -> MemoryMethod)
METHODS = {
    "flow": (("flow",), lambda b: FlowMethod(b["flow"])),
    "flow_noexp": (("flow",), lambda b: FlowMethod(b["flow"], expand=False)),
    "flow_nolinks": (("flow",), lambda b: FlowMethod(b["flow"], links=False)),
    "flow_stacked": (("flow",), lambda b: FlowMethod(b["flow"], stacked=True)),
    "flow_noevo": (("flow_noevo",), lambda b: FlowMethod(b["flow_noevo"])),
    "flow_links": (("flow_links",), lambda b: FlowMethod(b["flow_links"])),
    "flow_fullemb": (("flow_fullemb",), lambda b: FlowMethod(b["flow_fullemb"])),
    "flow_noevo_fullemb": (("flow_noevo_fullemb",), lambda b: FlowMethod(b["flow_noevo_fullemb"])),
    "amem": (("amem",), lambda b: AMemRetrieval(b["amem"])),
    "readagent": (("readagent",), lambda b: ReadAgentMethod(b["readagent"], domain="lme")),
    "memgas": (("memgas",), lambda b: MemGASMethod(b["memgas"], domain="lme")),
    "sculptor": (("sculptor",), lambda b: SculptorMethod(b["sculptor"], domain="lme")),
}
for name, kw in FLOW_VARIANTS.items():
    METHODS[name] = (("flow",), lambda b, kw=kw: FlowMethod(b["flow"], **kw))
    # a variant's ablations keep its render settings, so they can be run against the variant chosen for P0
    for abl, sw in (("noexp", dict(expand=False)), ("nolinks", dict(links=False)), ("stacked", dict(stacked=True))):
        METHODS[f"{name}_{abl}"] = (("flow",), lambda b, kw={**kw, **sw}: FlowMethod(b["flow"], **kw))
    METHODS[f"{name}_noevo"] = (("flow_noevo",), lambda b, kw=kw: FlowMethod(b["flow_noevo"], **kw))
    METHODS[f"{name}_links"] = (("flow_links",), lambda b, kw=kw: FlowMethod(b["flow_links"], **kw))
    METHODS[f"{name}_views"] = (("flow",), lambda b, kw=kw: FlowMethod(b["flow"], **kw, raw_share=RAW_SHARE))
    METHODS[f"{name}_noevo_zoom"] = (("flow_noevo",), lambda b, kw=kw: FlowMethod(b["flow_noevo"], **kw, **ZOOM))
    # A-MEM Zoom's ablations and variants after the mock review (2026-10-06; design deviation 33): no lines; Flow's
    # default history share; raw first (E0's C3); E 150 and 600; the units' first pieces; lines only over the same builds;
    # fit-or-degrade over Flow's registered builds (with evolution)
    for sfx, sw in (("_zoom_nofloor", dict(**ZOOM, floor=False)), ("_zoom_s50", dict(**ZOOM, split=0.5)),
                    ("_zoom_rawfirst", dict(zoom=150, raw_share=1.0)), ("_zoom_e150", dict(zoom=150)),
                    ("_zoom_e600", dict(zoom=600)), ("_zoom_lead", dict(**ZOOM, zoom_rank="lead")),
                    ("_zoom_e150_nofloor", dict(zoom=150, floor=False)),
                    ("_zoom_nolinks", dict(**ZOOM, links=False)),  # Zoom's component ablations (deviation 37)
                    # the render controls of deviations 38-39 on LongMemEval (2026-10-11)
                    ("_zoom_randrec", dict(**ZOOM, rand_recovery=True)), ("_zoom_shufcand", dict(**ZOOM, rand_order=True)),
                    ("_zoom_after", dict(**ZOOM, after=True)), ("_zoom_fixed", dict(**ZOOM, fixed=True)),
                    ("_noexp", dict(expand=False))):
        METHODS[f"{name}_noevo{sfx}"] = (("flow_noevo",), lambda b, kw={**kw, **sw}: FlowMethod(b["flow_noevo"], **kw))
    METHODS[f"{name}_zoom"] = (("flow",), lambda b, kw=kw: FlowMethod(b["flow"], **kw, **ZOOM))
    for kind in ("flow_fullemb", "flow_noevo_fullemb"):
        METHODS[name + kind[4:]] = ((kind,), lambda b, kw=kw, kind=kind: FlowMethod(b[kind], **kw))
METHODS["amem_zoom"] = (("amem",), lambda b: AMemRetrieval(b["amem"], zoom=ZOOM["zoom"]))
# the reviews' hybrid (2026-10-09, deviation 38): FlexA-MEM's timeline (the main arm's records in its 0.3 history share)
# with zero-build BM25 pieces as the recovery, over P0's builds
METHODS["flow_s30_ld50_noevo_bm25"] = (("flow_noevo",), lambda b: TimelinePieces(b["flow_noevo"], split=0.3))
# post hoc, after the reviewers' batch (2026-10-07): the two changes that batch favoured, E 150 and no floor of lines,
# on Zoom and (E 150) on A-MEM; chosen on held-out evidence, so they are reported as exploratory, not as a clean test
METHODS["amem_zoom_e150"] = (("amem",), lambda b: AMemRetrieval(b["amem"], zoom=150))
# "one memory or two" with the same writer, notes and candidate scores (an audit, 2026-10-06): Summary+A-MEM's summary
# in front of raw notes taken in the main arm's candidate order (Flow w/o evolution's builds, link discount 0.5) instead
# of A-MEM's retrieval
METHODS["summary_flow_s50"] = (("flow_noevo", "summary"), lambda b: SummaryAMem(
    b["flow_noevo"], b["summary"], split=0.5, persist_compression=False,
    order=lambda st, q: [u for u, _ in candidates(st, q, True, FLOW_VARIANTS["flow_s30_ld50"]["link_discount"])]))
# arms added after P0, which in the paper rendered P0's builds (--builds)
POST_P0 = ("summary_flow_s50", "amem_zoom")
for s in (30, 50, 70):
    METHODS[f"summary_amem_s{s}"] = (("amem", "summary"), lambda b, s=s: SummaryAMem(
        b["amem"], b["summary"], split=s / 100, persist_compression=False))
REFERENCES = ("truncation", "full", "oracle")
# methods outside the build kinds, built online for each question from its units (write(all units)) under the question
# guard: AFM (Cruz 2025), reproduced from the paper, with the paper's and the MiniLM-calibrated thresholds
ONLINE = {"afm": lambda: AFMMethod(), **{k: lambda t=t: AFMMethod(tau_high=t[0], tau_mid=t[1]) for k, t in CALIBRATED.items()}}
# zero-build retrieval baselines (methods/rag.py): built from the raw units, no LLM call, no build to read
RETRIEVAL = {"rag_raw": RawRetrieval, "rag_pieces": PieceRetrieval}
# groups are built in parallel, the kinds of one group in order: the writer calls of every Flow kind are flow's (only
# linking and embedding differ), so whichever is built after the first gets cache hits rather than a second, racing
# write; _build_lock keeps them apart across runs too
FLOW_KINDS = ("flow", "flow_noevo", "flow_links", "flow_fullemb", "flow_noevo_fullemb")
BUILD_GROUPS = (FLOW_KINDS, ("amem",), ("summary",), ("readagent",), ("memgas",), ("sculptor",))


def is_infra(e: BaseException) -> bool:
    # sqlite3.OperationalError: the LLM cache, shared by every process, was locked or unavailable; 401/403/404 (key,
    # permission or model id), 409 and the statuses the openai SDK does not map (bare APIStatusError) come from the
    # API, not the method
    return (isinstance(e, (LLMTransientError, OSError, sqlite3.OperationalError, openai.AuthenticationError,
                           openai.PermissionDeniedError, openai.NotFoundError, openai.ConflictError))
            or type(e) is openai.APIStatusError)


@contextlib.contextmanager
def _build_lock(build_dir: Path, kind: str, qid: str):
    """Exclusive lock on one question's builds of one group, across threads and processes (flock). BUILD_GROUPS alone
    orders builds within one process: p0_lme_flow and p0_lme_noevo, run in parallel, built both kinds of 93 questions
    at the same time, both missed the writer cache and wrote different notes (journal)."""
    group = next(g for g in BUILD_GROUPS if kind in g)[0]
    path = build_dir / ".locks" / f"{group}.{qid}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _build(kind: str, qid: str, units, build_dir: Path, run_id: str) -> tuple[object, str]:
    """Build (or load) one question's memory of the given kind, under its group's lock, so flow and flow_noevo of one
    question are never built at the same time, even by different runs."""
    with _build_lock(build_dir, kind, qid):
        return _build_unlocked(kind, qid, units, build_dir, run_id)


def _build_unlocked(kind: str, qid: str, units, build_dir: Path, run_id: str) -> tuple[object, str]:
    """Returns (object, build_id)."""
    path = build_dir / kind / f"{qid}.json.gz"
    if path.exists():
        if kind == "summary":
            with gzip.open(path, "rt") as f:
                d = json.load(f)
            return d["summary"], d["build_id"]
        st, extra = NoteStore.load(path)
        return st, extra["build_id"]
    build_id = uuid.uuid4().hex[:12]
    ledger = str(get_ledger().path)  # where this build's cost rows are, for runs that reuse it
    with ledger_context(stage="build", build=kind, build_id=build_id, qid=qid, run_id=run_id):
        if kind == "summary":
            s = summarize("", [f"[{u.time} · {u.span}]\n{u.text}" for u in units])
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            with gzip.open(tmp, "wt") as f:
                json.dump({"summary": s, "build_id": build_id, "ledger": ledger}, f)
            os.replace(tmp, path)
            return s, build_id
        if kind in FLOW_KINDS:
            m = FlowMethod(evolve="noevo" not in kind, rewrite=kind != "flow_links", full_embed=kind.endswith("_fullemb"))
            m.write(units)
            st, extra = m.store, {"link": m.stats()}
        elif kind == "amem":
            st = NoteStore()
            evo = amem_build(st, units)
            extra = {"evolved": sum(e["evolved"] for e in evo)}
        elif kind == "readagent":
            m = ReadAgentMethod()
            m.write(units)
            st, extra = m.store, {}
        elif kind == "memgas":
            m = MemGASMethod(domain="lme")
            m.write(units)
            st, extra = m.store, {"assoc": m.stats()}
        elif kind == "sculptor":
            m = SculptorMethod(domain="lme")
            m.write(units)
            m.summarize_all()
            st, extra = m.store, {}
        else:
            raise ValueError(kind)
    st.save(path, {"build_id": build_id, "ledger": ledger, **extra})
    return st, build_id


def run_item(item: dict, methods: list[str], budgets: list[int], results, run_id: str, build_dir: Path,
             seed: int = 0) -> None:
    qid = item["qid"]
    units = lme_units(item["sessions"])
    needed = sorted({b for m in methods if m in METHODS for b in METHODS[m][0]})
    todo = [(m, B) for m in methods for B in (budgets if m not in ("full", "oracle") else [0])
            if not results.has(qid=qid, method=m, budget=B)]
    if not todo:
        return
    attempt = uuid.uuid4().hex[:10]  # tags this attempt's ledger rows, so a rerun is not double-counted
    base = {"qid": qid, "qtype": item["qtype"], "abstention": item["gold"]["abstention"], "seed": seed,
            "n_units": len(units), "run_id": run_id, "attempt": attempt, "reader_model": get_llm().role("reader")["model"]}
    # the question may legitimately occur inside the haystack; the guard applies only when it does not
    haystack = "\n".join(u.text for u in units)
    guard = item["question"] if item["question"].strip() not in haystack else ""
    builds, build_ids, failed = {}, {}, {}
    kinds = [k for k in needed if any(k in METHODS[m][0] for m, _ in todo if m in METHODS)]
    groups = [[k for k in g if k in kinds] for g in BUILD_GROUPS if set(g) & set(kinds)]
    assert sum(map(len, groups)) == len(kinds), kinds

    def build(ks):
        out = {}
        for k in ks:
            try:
                with forbid_text(guard):
                    out[k] = _build(k, qid, units, build_dir, run_id)
            except Exception as e:
                out[k] = e
        return out
    with ThreadPoolExecutor(max_workers=max(1, len(groups))) as ex:
        futs = [ex.submit(contextvars.copy_context().run, build, g) for g in groups]
    for k, r in (kr for f in futs for kr in f.result().items()):
        if not isinstance(r, Exception):
            builds[k], build_ids[k] = r
        elif is_infra(r):
            raise r
        else:
            failed[k] = f"build {k}: {type(r).__name__}: {r}"
    for m in sorted({m for m, _ in todo if m in ONLINE}):
        bid = uuid.uuid4().hex[:12]
        try:
            with forbid_text(guard), ledger_context(stage="build", build=m, qid=qid, run_id=run_id, build_id=bid):
                inst = ONLINE[m]()
                inst.write(units)
            builds[m], build_ids[m] = inst, bid
        except Exception as e:
            if is_infra(e):
                raise
            failed[m] = f"build {m}: {type(e).__name__}: {e}"
    with ledger_context(stage="query_embed", qid=qid, run_id=run_id, attempt=attempt):
        q = Query(text=item["question"], emb=embed_one(item["question"], phase="embed"), date=item["question_date"])
    raw = NoteStore()
    for u in units:
        raw.add(Note.from_unit(u))
    for m, B in todo:
        cap = int(B * RESERVE) if B else None
        deps = METHODS[m][0] if m in METHODS else (m,) if m in ONLINE else ()
        bad = [failed[k] for k in deps if k in failed]
        if bad:
            results.add({**base, "method": m, "budget": B, "cap": cap, "answer": None, "error": bad[0]})
            continue
        try:
            row = _render_and_read(item, m, B, cap, q, raw, builds, run_id, seed, attempt)
        except Exception as e:
            if is_infra(e):
                raise
            row = {"answer": None, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()}
        results.add({**base, "method": m, "budget": B, "cap": cap,
                     "build_ids": {k: build_ids[k] for k in deps if k in build_ids}, **row})


def _render_and_read(item, m, B, cap, q, raw, builds, run_id, seed, attempt) -> dict:
    qid = item["qid"]
    salt = f"seed{seed}" if seed else None
    if m == "sculptor":  # the reader renders the fragments itself at each turn, after the model's tool calls
        with ledger_context(stage="read", method=m, budget=B, qid=qid, run_id=run_id, attempt=attempt):
            answer, n_prompt, stats = reader.read_sculptor(item, METHODS[m][1](builds), cap, salt=salt)
        return {"prompt_tokens": n_prompt, "answer": answer, "error": None, "render": stats}
    with ledger_context(stage="render", method=m, budget=B, qid=qid, run_id=run_id, attempt=attempt):
        stats = {}
        if m == "full":
            history = F.join([F.raw_block(n) for n in raw.notes.values()])
        elif m == "oracle":
            ev = set(item["gold"]["evidence_sids"])
            history = F.join([F.raw_block(raw.notes[u]) for u in raw.order if raw.notes[u].source in ev])
        else:
            allowance = reader.history_allowance(item, cap)
            if m == "truncation":
                r = render_recent_raw(raw, allowance)
            elif m in RETRIEVAL:
                r = RETRIEVAL[m](raw).render(q, allowance)
            elif m in ONLINE:
                r = builds[m].render(q, allowance)
            else:
                blds = {k: copy.copy(builds[k]) if k == "summary" else builds[k] for k in METHODS[m][0]}
                r = METHODS[m][1](blds).render(q, allowance)
            history, stats = r.text, r.stats
            stats["allowance"] = allowance
    with ledger_context(stage="read", method=m, budget=B, qid=qid, run_id=run_id, attempt=attempt):
        answer, n_prompt = reader.read(item, history, cap, salt=salt)
    return {"prompt_tokens": n_prompt, "answer": answer, "error": None, "render": stats}


def run(items, methods, budgets, results, run_id, workers=6, seed=0, builds=None):
    """builds: an existing build version to read instead of this code's; every build the methods need must exist
    there, so nothing is built. Without it, each question's memories are built with this code (and reused by later
    runs of the same code version). In the paper, the arms added after the first runs rendered those runs' builds."""
    build_dir = BUILD_ROOT / (builds or build_version())
    print(f"builds: {build_dir}", flush=True)
    if builds:
        kinds = {k for m in methods if m in METHODS for k in METHODS[m][0]}
        missing = [(k, it["qid"]) for k in kinds for it in items if not (build_dir / k / f"{it['qid']}.json.gz").exists()]
        assert not missing, f"{len(missing)} builds missing from {build_dir}, e.g. {missing[:3]}"
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(contextvars.copy_context().run, run_item, it, methods, budgets, results, run_id,
                          build_dir, seed): it["qid"] for it in items}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                f.result()
                print(f"[{i}/{len(items)}] {futs[f]} done", flush=True)
            except Exception:
                errors.append(futs[f])
                print(f"[{i}/{len(items)}] {futs[f]} FAILED\n{traceback.format_exc()}", flush=True)
    return errors


def results_path(run_id: str) -> Path:
    return ROOT / "artifacts/runs/lme" / run_id / "results.jsonl"
