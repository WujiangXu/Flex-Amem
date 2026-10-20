"""AppWorld bridge: one task world served over JSON lines on stdin/stdout. It runs under the AppWorld venv
(artifacts/appworld/venv: AppWorld pins pydantic<2, ours is 2), so it imports nothing from flexamem. AppWorld freezes the
clock inside this process (freezegun), so nothing here measures time; the runner times episodes in its own process.

Requests and replies (one JSON object per line):
    {"op": "init", "task_id", "experiment_name", "timeout"} -> {"instruction", "supervisor"}
    {"op": "execute", "code"}                                -> {"output", "completed"}
    {"op": "evaluate"}                                       -> the task's TestTracker.to_dict()
    {"op": "evaluate_saved", "task_id", "experiment_name"}   -> the same for the state saved after the last finished
                                                                execution (a fresh bridge, after the first one hung)
    {"op": "close"}                                          -> {}
A failed request replies {"error": "<type>: <message>"}. Whatever the agent's code raises during an execution comes back
as its output, so only a dead or silent bridge is an infrastructure failure (review of 2026-10-07)."""
import json
import sys


def main():
    out, sys.stdout = sys.stdout, sys.stderr  # AppWorld prints; only replies go to the protocol channel
    from appworld import AppWorld

    world = None
    for line in sys.stdin:
        req = json.loads(line)
        op = req["op"]
        try:
            if op == "init":
                world = AppWorld(task_id=req["task_id"], experiment_name=req["experiment_name"],
                                 timeout_seconds=req["timeout"])
                # AppWorld rewrites its API and I/O logs after every execution, outside the cell, and agent data can make
                # that raise (an emoji escape it prints, an iterator passed to an API); no task's evaluation reads them
                world.save_logs = lambda: None
                s = world.task.supervisor
                resp = {"instruction": world.task.instruction,
                        "supervisor": {k: s[k] for k in ("first_name", "last_name", "email", "phone_number")}}
            elif op == "execute":
                try:
                    output = world.execute(req["code"])
                except Exception as e:  # raised by the agent's code outside AppWorld's own cell handling
                    output = f"Execution failed. Traceback:\n{type(e).__name__}: {e}"
                # lone surrogates (escaped emoji the agent printed) cannot be sent on to the model API
                resp = {"output": output.encode("utf-8", "replace").decode("utf-8"),
                        "completed": bool(world.task_completed())}
            elif op == "evaluate":
                resp = world.evaluate().to_dict()
            elif op == "evaluate_saved":
                from appworld.evaluator import evaluate_task
                resp = evaluate_task(task_id=req["task_id"], experiment_name=req["experiment_name"]).to_dict()
            elif op == "close":
                if world is not None:
                    world.close()
                resp = {}
            else:
                resp = {"error": f"unknown op {op!r}"}
        except Exception as e:
            resp = {"error": f"{type(e).__name__}: {e}"}
        out.write(json.dumps(resp, default=str) + "\n")
        out.flush()
        if op == "close":
            break


if __name__ == "__main__":
    main()
