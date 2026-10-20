"""AppWorld runner (Trivedi et al., ACL 2024, arXiv 2407.18901): one mini-swe-agent episode per (task, method, budget)
on Test-Normal, scored by AppWorld's unit tests on the final state of the apps' databases. The second agentic
benchmark (decision 2026-10-07_appworld; docs/benchmarks/survey_2026-10.md).

The agent loop, context policies, memory methods, budget cap and ledger are SWE's (adapters/mswea.py, bench/swe/runner.py);
only the environment, the action tool and the prompt change. The agent writes Python through a `python` tool; the code
runs in the task's world through the bridge (env.py), where variables persist across calls. The prompt is AppWorld's
ReAct instructions, condensed and without the few-shot demonstration so the fixed prefix stays small under B = 4k.
An episode ends when the agent calls apis.supervisor.complete_task() or hits the step limit. Memory starts empty in every
episode and sees only the agent's own trajectory; the tests and the ground truth stay inside AppWorld's evaluator.

Failures split as on SWE: a dead bridge (or one silent on a non-execute request) and API outages are infrastructure (no
row, rerun on resume); an execution the agent's code hangs is a method error, scored from the saved state; a memory
call that keeps failing or a context that cannot fit is a method error, recorded as a failed episode."""
import json
import time
import traceback
import uuid
from pathlib import Path

from jinja2 import StrictUndefined, Template

from flexamem.adapters.mswea import ContextAgent, MaskedTracePolicy, MaskingPolicy, MemoryPolicy, Policy, SculptorPolicy, TruncationPolicy
from flexamem.bench.appworld.env import APPWORLD_ROOT, AppWorldEnvironment
from flexamem.bench.swe.runner import MEMORY, is_infra
from flexamem.llm.agent_model import make_agent_model
from flexamem.llm.client import ROOT, get_llm
from flexamem.llm.ledger import ledger_context

ARMS = ("flow_noevo_zoom", "flow_noevo", "amem", "rag_pieces", "rag_raw")  # memory arms; references below
REFERENCES = ("passthrough", "truncation", "masking", "masking_trace")
SPLIT_DIR = APPWORLD_ROOT / "data/datasets"
AGENT = {"step_limit": 60, "cost_limit": 0.0}  # no wall-clock limit, as on SWE: a slow API must not fail an episode

PY_TOOL = {"type": "function", "function": {
    "name": "python",
    "description": "Run Python code in the task's REPL session (variables persist across calls) and return what it printed.",
    "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "The Python code to run"}},
                   "required": ["code"]}}}

SYSTEM = """You are a super intelligent AI assistant who completes my day-to-day tasks fully autonomously by operating \
my apps (e.g. spotify, venmo, amazon, gmail, phone) through their APIs on my behalf. You work in a Python REPL: each \
python tool call runs your code in the same session, so variables persist, and you see what it prints.

Find out what you need with these APIs:
print(apis.api_docs.show_app_descriptions())  # the apps available to you
print(apis.api_docs.show_api_descriptions(app_name='spotify'))  # the APIs of an app
print(apis.api_docs.show_api_doc(app_name='spotify', api_name='login'))  # the specification of one API

Key instructions:
1. Always look at an API's specification before calling it.
2. Write small pieces of code, one step at a time, and check their output before making any irreversible change.
3. The supervisor app holds my account credentials, addresses and payment cards. The phone app holds my contacts \
(friends, family, roommates, coworkers and so on).
4. Many APIs return items in pages. Loop over page_index until a page comes back empty.
5. Use only real data from API calls; never make up values.
6. When the task is done, call apis.supervisor.complete_task(). If the task asks for information, pass it as the \
answer, e.g. apis.supervisor.complete_task(answer=42); otherwise call it without arguments. This ends the task.

Every response must contain at least one python tool call."""

INSTANCE = """My name is {{first_name}} {{last_name}}. My personal email is {{email}} and my phone number is \
{{phone_number}}.
Task: {{instruction}}"""

OBSERVATION = """{% if output.exception_info -%}
<exception>{{output.exception_info}}</exception>
{% endif -%}
{% if output.output | length < 10000 -%}
<output>
{{ output.output -}}
</output>
{%- else -%}
<warning>
The output was too long. Print less: select the fields you need or slice long lists.
</warning>
{%- set elided_chars = output.output | length - 10000 -%}
<output_head>
{{ output.output[:5000] }}
</output_head>
<elided_chars>
{{ elided_chars }} characters elided
</elided_chars>
<output_tail>
{{ output.output[-5000:] }}
</output_tail>
{%- endif -%}"""

FORMAT_ERROR = """Tool call error:

<error>
{{error}}
</error>

Every response must call the python tool at least once, with your code as its `code` argument."""


def load_tasks(split: str = "test_normal", ids=None) -> list[str]:
    tasks = (SPLIT_DIR / f"{split}.txt").read_text().split()
    return [t for t in tasks if t in set(ids)] if ids else tasks


def make_policy(method: str, budget: int, task: str) -> Policy:
    if method == "passthrough":
        return Policy(None, [PY_TOOL])
    if method == "truncation":
        return TruncationPolicy(budget, [PY_TOOL])
    if method == "masking_trace":  # masking with a rule-written action log (the paper's Limitations, 2026-10-12)
        return MaskedTracePolicy(budget, [PY_TOOL])
    if method == "masking":
        return MaskingPolicy(budget, [PY_TOOL])
    if method == "sculptor":  # its context tools join python once compaction starts (domain "swe": the step format)
        return SculptorPolicy(MEMORY[method](), budget, task, tools=[PY_TOOL])
    return MemoryPolicy(MEMORY[method](), budget, task, name=method, tools=[PY_TOOL])


def arm_name(method: str, budget: int) -> str:
    return f"{method}_{budget}" if budget else method


def run_episode(task_id: str, method: str, budget: int, run_dir: Path, run_id: str, step_limit: int = 60) -> dict:
    arm, attempt = arm_name(method, budget), uuid.uuid4().hex[:10]
    traj = run_dir / "trajs" / arm / f"{task_id}.traj.json"
    env, agent, error, ev = None, None, None, None
    t0 = time.time()
    with ledger_context(stage="episode", method=method, budget=budget, iid=task_id, run_id=run_id, attempt=attempt):
        try:
            env = AppWorldEnvironment(task_id, f"flexamem/{run_id}/{arm}/{attempt}",
                                      run_dir / "bridge_logs" / arm / f"{task_id}.{attempt}.log")
            task = Template(INSTANCE, undefined=StrictUndefined).render(**env.get_template_vars())  # as the agent renders it
            policy = make_policy(method, budget, task)
            agent = ContextAgent(make_agent_model(OBSERVATION, FORMAT_ERROR, action_tool="python"), env,
                                 policy=policy, output_path=traj, system_template=SYSTEM, instance_template=INSTANCE,
                                 **{**AGENT, "step_limit": step_limit})
            try:
                exit_info = agent.run(task)
            except Exception as e:
                if is_infra(e):
                    raise
                error, exit_info = f"{type(e).__name__}: {e}", {"exit_status": "error"}
            ev = env.evaluate()  # the final database state, whatever ended the episode
        finally:
            if env is not None:
                env.cleanup()
    log = agent.context_log
    passes = len(ev["passes"])  # TestTracker.to_dict(): lists of passed and failed tests
    assert passes + len(ev["failures"]) == ev["num_tests"], ev
    return {"task_id": task_id, "scenario": task_id.split("_")[0], "method": method, "budget": budget,
            "success": bool(ev["success"]), "passes": passes, "num_tests": ev["num_tests"],
            "score": passes / ev["num_tests"] if ev["num_tests"] else None, "difficulty": ev.get("difficulty"),
            "exit_status": exit_info.get("exit_status"), "error": error,
            "steps": agent.n_calls, "step_limit": step_limit, "compacted_steps": sum(x["mode"] != "pass" for x in log),
            "format_errors": sum(bool(x.get("format_error")) for x in log),
            "max_raw_tokens": max((x["raw_tokens"] for x in log), default=0),
            "max_prompt_tokens": max((x["tokens"] for x in log), default=0),
            "api_prompt_tokens": sum(x.get("api_prompt_tokens") or 0 for x in log),
            "memory": agent.policy.stats(), "agent_model": get_llm().role("agent")["model"],
            "memory_model": get_llm().role("memory")["model"], "cap": agent.policy.cap,
            "max_concurrency": get_llm().cfg.get("max_concurrency"), "wall_s": round(time.time() - t0, 1),
            "run_id": run_id, "attempt": attempt, "traj": str(traj.relative_to(ROOT))}


def run(jobs, results, run_dir: Path, run_id: str, workers: int = 8, step_limit: int = 60) -> list:
    """jobs: [(task_id, method, budget)]; resumable through `results`. Infrastructure failures are logged to
    errors.jsonl and rerun on the next resume."""
    import contextvars
    from concurrent.futures import ThreadPoolExecutor, as_completed
    todo = [j for j in jobs if not results.has(task_id=j[0], method=j[1], budget=j[2])]
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(contextvars.copy_context().run, run_episode, t, m, b, run_dir, run_id, step_limit): (t, m, b)
                for t, m, b in todo}
        for i, f in enumerate(as_completed(futs), 1):
            key = futs[f]
            try:
                row = f.result()
                results.add(row)
                print(f"[{i}/{len(todo)}] {key} success={row['success']} tests={row['passes']}/{row['num_tests']} "
                      f"steps={row['steps']} compacted={row['compacted_steps']} error={row['error']}", flush=True)
            except Exception as e:
                errors.append(key)
                with (run_dir / "errors.jsonl").open("a") as fh:
                    fh.write(json.dumps({"key": key, "ts": time.time(), "infra": is_infra(e),
                                         "tb": traceback.format_exc()}) + "\n")
                print(f"[{i}/{len(todo)}] {key} FAILED\n{traceback.format_exc()}", flush=True)
    return errors


def run_dir_for(run_id: str) -> Path:
    return ROOT / "artifacts/runs/appworld" / run_id
