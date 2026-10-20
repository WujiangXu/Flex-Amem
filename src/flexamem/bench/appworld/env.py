"""mini-swe-agent environment over one AppWorld task: the agent's Python runs in the task's world, in a bridge process
under the AppWorld venv (bridge.py). The episode ends when the agent has called apis.supervisor.complete_task(); the
world is then evaluated by AppWorld's own unit tests on the final database state."""
import json
import os
import selectors
import subprocess
import tempfile
from pathlib import Path

from minisweagent.exceptions import Submitted

from flexamem.llm.client import ROOT

APPWORLD = ROOT / "artifacts/appworld"
VENV_PY = APPWORLD / "venv/bin/python"
APPWORLD_ROOT = APPWORLD / "root"   # AppWorld's data and experiment outputs
BRIDGE = Path(__file__).with_name("bridge.py")
CODE_TIMEOUT = 60      # AppWorld's own limit on one execution (timeout_seconds)
REPLY_TIMEOUT = 600    # a bridge that does not answer within this is broken: infrastructure, rerun on resume
EXEC_TIMEOUT = 300     # an execution that outlives AppWorld's own limit this long is hung by the agent's code


class BridgeError(OSError):
    """The bridge died, hung or failed a request: an infrastructure failure (is_infra counts OSError)."""


class ExecutionHang(RuntimeError):
    """The agent's code never returned: it caught AppWorld's one-shot timeout (a plain Exception) inside a loop. A method
    outcome, not infrastructure: the episode ends with an error and the state saved after the last finished execution is
    scored (evaluate falls back to a fresh bridge)."""


class AppWorldEnvironment:
    def __init__(self, task_id: str, experiment_name: str, log_path: Path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.task_id, self.experiment_name = task_id, experiment_name
        self._log = log_path.open("a")
        self._cwd = tempfile.TemporaryDirectory(prefix="aw_")  # empty: no relative path from the REPL reaches the data
        self.proc = self._start()
        self.task = self._call({"op": "init", "task_id": task_id, "experiment_name": experiment_name,
                                "timeout": CODE_TIMEOUT})

    def _start(self) -> subprocess.Popen:
        # a minimal environment: the agent's code can read os.environ and list the host's files (AppWorld's guard allows
        # os.scandir and pathlib), so no API key, proxy or real HOME reaches it (journal, 2026-10-08)
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C.UTF-8", "HOME": self._cwd.name,
               "APPWORLD_ROOT": str(APPWORLD_ROOT)}
        return subprocess.Popen([str(VENV_PY), "-I", str(BRIDGE)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=self._log, text=True, cwd=self._cwd.name, env=env)

    def _call(self, req: dict, timeout: int = REPLY_TIMEOUT) -> dict:
        if self.proc.poll() is not None:
            raise BridgeError(f"{self.task_id}: bridge exited with {self.proc.returncode}")
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        sel = selectors.DefaultSelector()
        sel.register(self.proc.stdout, selectors.EVENT_READ)
        ready = sel.select(timeout)
        sel.close()
        line = self.proc.stdout.readline() if ready else ""
        if not line:
            alive = self.proc.poll() is None
            self.proc.kill()
            self.proc.wait(timeout=60)  # reaped, so evaluate() sees it gone and starts a fresh bridge
            if alive and req["op"] == "execute":
                raise ExecutionHang(f"{self.task_id}: the agent's code ran past {timeout} s (AppWorld's {CODE_TIMEOUT} s "
                                    f"timeout was caught)")
            raise BridgeError(f"{self.task_id}: no reply to {req['op']} within {timeout} s")
        resp = json.loads(line)
        if "error" in resp:
            raise BridgeError(f"{self.task_id}: {req['op']} failed: {resp['error']}")
        return resp

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict:
        resp = self._call({"op": "execute", "code": action.get("command", "")}, EXEC_TIMEOUT)
        if resp["completed"]:
            raise Submitted({"role": "exit", "content": "",
                             "extra": {"exit_status": "Submitted", "submission": ""}})
        return {"output": resp["output"], "returncode": 0, "exception_info": ""}

    def evaluate(self) -> dict:
        if self.proc.poll() is None:
            return self._call({"op": "evaluate"})
        self.proc.wait(timeout=60)
        self.proc = self._start()  # the first bridge was killed after a hang: score the saved state
        return self._call({"op": "evaluate_saved", "task_id": self.task_id, "experiment_name": self.experiment_name})

    def get_template_vars(self) -> dict:
        return {"task_id": self.task_id, **self.task["supervisor"], "instruction": self.task["instruction"]}

    def serialize(self) -> dict:
        return {"info": {"config": {"environment": {"task_id": self.task_id, "experiment_name": self.experiment_name,
                                                    "code_timeout": CODE_TIMEOUT},
                                    "environment_type": f"{__name__}.{type(self).__name__}"}}}

    def cleanup(self):
        try:
            if self.proc.poll() is None:
                self._call({"op": "close"})
        except BridgeError:
            pass
        finally:
            if self.proc.poll() is None:
                self.proc.kill()
            self.proc.wait(timeout=60)
            self._log.close()
            self._cwd.cleanup()
