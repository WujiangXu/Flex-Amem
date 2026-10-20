"""SWE-bench Verified runner: one mini-swe-agent episode per (instance, method, budget) in a fresh
container, graded by the official harness. The task is the problem statement alone (as upstream
mini-swe-agent; no hints), memory starts empty in every episode, and nothing from the hidden tests
reaches the agent or its memory. Agent settings are mini-swe-agent's swebench.yaml (step limit 250).

Failures are split by cause. Infrastructure failures (API outages after retries, container or daemon
errors) leave no result row and the episode is rerun on resume. Failures of the method itself (a memory
call that keeps returning invalid output, a context that cannot be made to fit) are recorded as
unresolved episodes with an `error` field, so no arm is re-sampled until it succeeds. A finished
episode whose grading has no verdict keeps its saved patch and a row with graded=false; --regrade
scores it later.

Agent containers are started with `docker run` on the host network, so an agent may reach the internet (in the
paper's runs, through the host's proxy). `egress=False` starts them with --network=none; it is used to rerun
episodes whose agent fetched external content (scripts/scan_egress.py).

Images are never pulled during a run (--pull=never; grading checks the same), so every episode of an instance runs
on the same image: pull them first (scripts/pull_swe_images.sh). A missing image is an infrastructure failure, and
every row records the ID of the image it ran on (`image_id`)."""
import hashlib
import json
import math
import re
import sqlite3
import subprocess
import time
import traceback
import uuid
from pathlib import Path

import litellm
import openai
import yaml
from minisweagent.config import builtin_config_dir
from minisweagent.environments.docker import DockerEnvironment

from flexamem.adapters.mswea import (ContextAgent, MaskingPolicy, MemoryPolicy, Policy, SculptorPolicy,
                                     TruncationPolicy)
from flexamem.bench.swe.checkpoints import BUDGET, THRESHOLDS
from flexamem.bench.swe.grade import THREAD_ENV, GradingInfraError, grade_one
from flexamem.llm.agent_model import make_agent_model
from flexamem.llm.client import ROOT, LLMTransientError, get_llm
from flexamem.llm.ledger import ledger_context
from flexamem.methods.afm import CALIBRATED, AFMMethod
from flexamem.methods.amem import AMemRetrieval, SummaryAMem
from flexamem.methods.flow import FLOW_VARIANTS, ZOOM_SWE, FlowMethod
from flexamem.methods.memgas import MemGASMethod
from flexamem.methods.hybrid import TimelinePieces
from flexamem.methods.rag import PieceRetrieval, RawRetrieval
from flexamem.methods.readagent import ReadAgentMethod
from flexamem.methods.sculptor import SculptorMethod
from flexamem.methods.trace import TraceMemory

DATA = ROOT / "artifacts/data/swebench/verified_test.json"
CFG = yaml.safe_load((builtin_config_dir / "benchmarks" / "swebench.yaml").read_text())
# 2.3.0 runs actions with non-login `bash -c`, which never sources the image's ~/.bashrc (`conda activate testbed`),
# so the agent's python lacked the repo's dependencies; mini-swe-agent 2.3.1 adds this same line (journal)
CFG["environment"]["env"]["BASH_ENV"] = "/root/.bashrc"
# uncapped reference: commit the container when raw history first reaches each diagnostic checkpoint (10000, 16000)
SNAPSHOT_AT = tuple(math.ceil(t * BUDGET) for t in THRESHOLDS)

MEMORY = {
    "flow": lambda: FlowMethod(),
    "flow_noexp": lambda: FlowMethod(expand=False),
    "flow_nolinks": lambda: FlowMethod(links=False),
    "flow_stacked": lambda: FlowMethod(stacked=True),
    "flow_noevo": lambda: FlowMethod(evolve=False),
    "flow_links": lambda: FlowMethod(rewrite=False),
    "flow_fullemb": lambda: FlowMethod(full_embed=True),
    "flow_noevo_fullemb": lambda: FlowMethod(evolve=False, full_embed=True),
    "flow_noevo_zoom": lambda: FlowMethod(evolve=False, **ZOOM_SWE),  # A-MEM Zoom (exploratory)
    # A-MEM Zoom's own component ablations (2026-10-08, design deviation 37): records only, no links, no lines
    "flow_noevo_zoom_noexp": lambda: FlowMethod(evolve=False, expand=False, **ZOOM_SWE),
    "flow_noevo_zoom_nolinks": lambda: FlowMethod(evolve=False, links=False, **ZOOM_SWE),
    "flow_noevo_zoom_nofloor": lambda: FlowMethod(evolve=False, floor=False, **ZOOM_SWE),
    # the reviews' two controls (2026-10-09, deviation 38): FlexA-MEM's timeline with BM25 pieces as the recovery, and
    # FlexA-MEM with its expansions chosen at random
    "flow_noevo_bm25": lambda: TimelinePieces(split=0.5),
    "flow_noevo_zoom_randrec": lambda: FlowMethod(evolve=False, rand_recovery=True, **ZOOM_SWE),
    "flow_noevo_zoom_shufcand": lambda: FlowMethod(evolve=False, rand_order=True, **ZOOM_SWE),  # render controls (39)
    "flow_noevo_zoom_after": lambda: FlowMethod(evolve=False, after=True, **ZOOM_SWE),
    "flow_noevo_zoom_fixed": lambda: FlowMethod(evolve=False, fixed=True, **ZOOM_SWE),
    "flow_noevo_noexp": lambda: FlowMethod(evolve=False, expand=False),  # main arm's ablations (AppWorld, 2026-10-08)
    "flow_noevo_nolinks": lambda: FlowMethod(evolve=False, links=False),
    "trace": lambda: TraceMemory(),  # rule-based action trace, no LLM (insight experiment B, 2026-10-08)
    # record_content's new directions, one change each on the main arm (AppWorld, 2026-10-08; design deviation 36)
    "flow_noevo_index": lambda: FlowMethod(evolve=False, index=True),
    "flow_noevo_digest": lambda: FlowMethod(evolve=False, digest=True),
    "flow_noevo_docsig": lambda: FlowMethod(evolve=False, docsig=True),
    "rag_raw": lambda: RawRetrieval(), "rag_pieces": lambda: PieceRetrieval(),  # zero-build retrieval (deviation 33)
    "amem": lambda: AMemRetrieval(),
    "amem_zoom": lambda: AMemRetrieval(**ZOOM_SWE),  # same writer as amem, fit-or-degrade render (AppWorld, deviation 39)
    "afm": lambda: AFMMethod(),  # AFM (Cruz 2025), reproduced from the paper (decision 2026-10-11_locomo_afm)
    **{k: lambda t=t: AFMMethod(tau_high=t[0], tau_mid=t[1]) for k, t in CALIBRATED.items()},  # MiniLM-scale thresholds
    "readagent": lambda: ReadAgentMethod(domain="swe"),
    "memgas": lambda: MemGASMethod(domain="swe"),
    "sculptor": lambda: SculptorMethod(domain="swe"),  # behind SculptorPolicy (make_policy)
}
for _n, _kw in FLOW_VARIANTS.items():
    MEMORY[_n] = lambda kw=_kw: FlowMethod(**kw)
    # a variant's ablations keep its render settings, so they can be run against the variant chosen for P0
    for _a, _sw in (("noexp", dict(expand=False)), ("nolinks", dict(links=False)), ("stacked", dict(stacked=True)),
                    ("noevo", dict(evolve=False)), ("links", dict(rewrite=False)), ("fullemb", dict(full_embed=True)),
                    ("noevo_fullemb", dict(evolve=False, full_embed=True))):
        MEMORY[f"{_n}_{_a}"] = lambda kw={**_kw, **_sw}: FlowMethod(**kw)
for _s in (30, 50, 70):
    MEMORY[f"summary_amem_s{_s}"] = lambda s=_s: SummaryAMem(split=s / 100)
REFERENCES = ("passthrough", "truncation", "masking")
CONTAINER_TIMEOUT = "24h"  # mini-swe-agent's default of 2h can end a long episode mid-run
_TRANSIENT = tuple(getattr(litellm.exceptions, n) for n in (
    "RateLimitError", "APIConnectionError", "Timeout", "ServiceUnavailableError", "InternalServerError",
    "BadGatewayError") if hasattr(litellm.exceptions, n))


def is_infra(e: BaseException) -> bool:
    # sqlite3.OperationalError: the LLM cache, shared by every process, was locked or unavailable.
    # 401/403/404 (key, permission or model id) and other statuses outside 400/422/429/5xx come from the API, not the
    # method: on the agent side litellm raises the unmapped ones as its APIError, on the memory side (client.py, the
    # openai SDK) they are ConflictError (409) or the bare APIStatusError. ContentPolicyViolationError is a
    # BadRequestError, not one of these, so a final flag stays a method outcome.
    return (isinstance(e, (LLMTransientError, GradingInfraError, subprocess.SubprocessError, OSError,
                           sqlite3.OperationalError, openai.AuthenticationError, openai.PermissionDeniedError,
                           openai.NotFoundError, openai.ConflictError, litellm.exceptions.APIError) + _TRANSIENT)
            or type(e) is openai.APIStatusError)


def make_policy(method: str, budget: int, task: str) -> Policy:
    if method == "passthrough":
        return Policy(None)
    if method == "truncation":
        return TruncationPolicy(budget)
    if method == "masking":
        return MaskingPolicy(budget)
    if method == "sculptor":
        return SculptorPolicy(MEMORY[method](), budget, task)
    return MemoryPolicy(MEMORY[method](), budget, task, name=method)


EXEC = "docker"  # recorded in every row; actions run through `docker exec` (mini-swe-agent's DockerEnvironment)
ENVIRONMENT = DockerEnvironment


def image_name(iid: str) -> str:
    return f"docker.io/swebench/sweb.eval.x86_64.{iid.replace('__', '_1776_')}:latest".lower()


def local_image_id(iid: str) -> str:
    """ID of the local image an episode of `iid` runs on; CalledProcessError (infra) if it is missing."""
    return subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", image_name(iid)], check=True,
                          capture_output=True, text=True, timeout=300).stdout.strip().removeprefix("sha256:")


def load_instances(ids=None) -> list[dict]:
    rows = {r["instance_id"]: r for r in json.loads(DATA.read_text())}
    return [rows[i] for i in ids] if ids else list(rows.values())


def snapshot_tag(iid: str, run_id: str, thr: int) -> str:
    """Carries the run id, so a later run cannot overwrite the images an older row points to."""
    rid = re.sub(r"[^A-Za-z0-9_.-]", "_", run_id)
    room = 128 - len(f"{iid}--{thr}")  # a tag holds at most 128 characters
    if len(rid) > room:
        rid = f"{rid[:room - 9]}-{hashlib.sha1(run_id.encode()).hexdigest()[:8]}"
    return f"localhost/flexamem-ckpt:{iid.lower()}-{rid}-{thr}"


def _snapshotter(iid: str, run_id: str, out: list):
    todo = list(SNAPSHOT_AT)

    def on_step(agent, info):
        while todo and info["raw_tokens"] >= todo[0]:
            thr = todo.pop(0)
            tag = snapshot_tag(iid, run_id, thr)
            subprocess.run([agent.env.config.executable, "commit", agent.env.container_id, tag],
                           check=True, capture_output=True, timeout=600)
            # the checkpoint is the container state plus the messages before this step's reply
            out.append({"threshold": thr, "step": info["step"], "n_messages": len(agent.messages), "image": tag})
    return on_step


def arm_name(method: str, budget: int) -> str:
    return f"{method}_{budget}" if budget else method


def grade(instance: dict, patch: str, arm: str, run_dir: Path, run_id: str, attempt: str) -> dict:
    """Grading outcome fields for a result row; never raises."""
    if not patch.strip():
        return {"graded": True, "graded_as": "empty_patch", "resolved": False, "patch_applied": False}
    try:
        rep = grade_one(instance, patch, run_id=f"{run_id}_{arm}_{attempt}", log_root=run_dir / "grading",
                        model_name=arm)
    except Exception as e:  # no verdict: keep the patch, re-grade later with --regrade
        return {"graded": False, "graded_as": None, "resolved": None, "patch_applied": None,
                "grade_error": f"{type(e).__name__}: {e}"}
    return {"graded": True, "graded_as": rep["graded_as"], "resolved": bool(rep.get("resolved")),
            "patch_applied": rep.get("patch_successfully_applied")}


def run_episode(instance: dict, method: str, budget: int, run_dir: Path, run_id: str,
                snapshots: bool = False, egress: bool = True) -> dict:
    iid = instance["instance_id"]
    arm = arm_name(method, budget)
    attempt = uuid.uuid4().hex[:10]
    traj = run_dir / "trajs" / arm / f"{iid}.traj.json"
    task = instance["problem_statement"]
    snaps, env, agent, error = [], None, None, None
    t0 = time.time()
    with ledger_context(stage="episode", method=method, budget=budget, iid=iid, run_id=run_id, attempt=attempt):
        policy = make_policy(method, budget, task)
        image_id = local_image_id(iid)  # raises (infra) if the image is missing
        try:
            env = ENVIRONMENT(image=image_name(iid), container_timeout=CONTAINER_TIMEOUT,
                              **{**{k: v for k, v in CFG["environment"].items() if k != "environment_class"},
                                 "env": {**CFG["environment"]["env"], **THREAD_ENV}},
                              run_args=["--rm", "--pull=never", "--network=host" if egress else "--network=none"])
            agent = ContextAgent(make_agent_model(CFG["model"]["observation_template"],
                                                  CFG["model"]["format_error_template"]),
                                 env, policy=policy, on_step=_snapshotter(iid, run_id, snaps) if snapshots else None,
                                 output_path=traj, **CFG["agent"])
            exit_info = agent.run(task)
        except Exception as e:
            if is_infra(e) or agent is None:
                raise
            error = f"{type(e).__name__}: {e}"
            exit_info = {"exit_status": "error", "submission": ""}
        finally:
            if env is not None:
                env.cleanup()
    patch = exit_info.get("submission", "") or ""
    pdir = run_dir / "patches" / arm
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / f"{iid}.diff").write_text(patch)  # the latest attempt's patch
    (pdir / f"{iid}.{attempt}.diff").write_text(patch)  # this attempt's, for --regrade of this row
    g = grade(instance, patch, arm, run_dir, run_id, attempt)
    log = agent.context_log
    return {"iid": iid, "repo": instance["repo"], "method": method, "budget": budget,
            "exit_status": exit_info.get("exit_status"), **g, "error": error,
            "steps": agent.n_calls, "compacted_steps": sum(x["mode"] != "pass" for x in log),
            "format_errors": sum(bool(x.get("format_error")) for x in log),
            "max_raw_tokens": max((x["raw_tokens"] for x in log), default=0),
            "max_prompt_tokens": max((x["tokens"] for x in log), default=0),
            "api_prompt_tokens": sum(x.get("api_prompt_tokens") or 0 for x in log),
            "memory": policy.stats(), "snapshots": snaps, "egress": egress, "exec": EXEC, "image_id": image_id,
            "agent_model": get_llm().role("agent")["model"], "cap": policy.cap,
            "max_concurrency": get_llm().cfg.get("max_concurrency"), "wall_s": round(time.time() - t0, 1),
            "run_id": run_id, "attempt": attempt, "traj": str(traj.relative_to(ROOT))}


def regrade(results, instances: dict, run_dir: Path, run_id: str, workers: int = 8) -> int:
    """Re-grade rows whose grading reached no verdict, from their saved patches (in parallel; grade() serialises
    the arms of one instance)."""
    from concurrent.futures import ThreadPoolExecutor

    def one(r):
        pdir = run_dir / "patches" / arm_name(r["method"], r["budget"])
        p = pdir / f"{r['iid']}.{r['attempt']}.diff"
        patch = (p if p.exists() else pdir / f"{r['iid']}.diff").read_text()  # older runs kept only the latest patch
        g = grade(instances[r["iid"]], patch, arm_name(r["method"], r["budget"]), run_dir, run_id,
                  f"{r['attempt']}r{int(time.time())}")
        results.add({**r, **g, "regraded": True})
        print(f"regrade {r['iid']} {r['method']} {r['budget']}: {g}", flush=True)
        return g["graded"]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return sum(ex.map(one, [r for r in results.rows() if r.get("graded") is False]))


def run(jobs, results, run_dir: Path, run_id: str, workers: int = 8, snapshots: bool = False,
        egress: bool = True) -> list:
    """jobs: [(instance, method, budget)]; resumable through `results`. Episodes that fail for
    infrastructure reasons are logged to errors.jsonl and rerun on the next resume (see module doc)."""
    import contextvars
    from concurrent.futures import ThreadPoolExecutor, as_completed
    todo = [j for j in jobs if not results.has(iid=j[0]["instance_id"], method=j[1], budget=j[2])]
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(contextvars.copy_context().run, run_episode, inst, m, b, run_dir, run_id,
                          snapshots and m == "passthrough", egress): (inst["instance_id"], m, b)
                for inst, m, b in todo}
        for i, f in enumerate(as_completed(futs), 1):
            key = futs[f]
            try:
                row = f.result()
                results.add(row)
                print(f"[{i}/{len(todo)}] {key} resolved={row['resolved']} graded_as={row['graded_as']} "
                      f"steps={row['steps']} compacted={row['compacted_steps']} error={row['error']}", flush=True)
            except Exception as e:
                errors.append(key)
                with (run_dir / "errors.jsonl").open("a") as fh:
                    fh.write(json.dumps({"key": key, "ts": time.time(), "infra": is_infra(e),
                                         "tb": traceback.format_exc()}) + "\n")
                print(f"[{i}/{len(todo)}] {key} FAILED\n{traceback.format_exc()}", flush=True)
    return errors


def run_dir_for(run_id: str) -> Path:
    return ROOT / "artifacts/runs/swe" / run_id
