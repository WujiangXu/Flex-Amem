"""Grade SWE-bench Verified patches with the official swebench 4.1.0 harness on prebuilt images.

make_test_spec() downloads requirements / environment.yml from GitHub to write the *env-image*
build script. With prebuilt instance images that script is never executed, so both lookups are stubbed
to "" (no network access needed). The log parser and report logic are the unmodified harness; the eval
script only gains THREAD_ENV exports (on many-core hosts OpenMP tests that start one thread per core time
out) and, for sphinx, PYTEST_ADDOPTS=-rA: its `-rA` lives in the images' tox.ini edit on top of the base
commit, which a path-less `git checkout <base>` reverts, leaving the parser no per-test lines.
Missing images are never pulled during a run (pull them first: scripts/pull_swe_images.sh); grading raises
GradingInfraError. Containers are created in the host network namespace (`network_mode="host"`), as in the
paper's runs.
The Docker daemon is the one `docker.from_env()` finds (DOCKER_HOST, else the default socket).
CLI: python -m flexamem.bench.swe.grade <run_evaluation args>   (keep the default --namespace swebench: prebuilt images)
"""
import json
import runpy
import shutil
import threading
from pathlib import Path

import swebench.harness.test_spec.python as _py

_py.get_requirements = lambda instance: ""
_py.get_environment_yml = lambda instance, env_name: ""

import docker  # noqa: E402
import swebench.harness.run_evaluation as _rev  # noqa: E402
from swebench.harness.constants import APPLY_PATCH_FAIL, LOG_INSTANCE, LOG_REPORT  # noqa: E402
from swebench.harness.test_spec.test_spec import make_test_spec  # noqa: E402


class GradingInfraError(RuntimeError):
    """The harness reached no verdict (daemon, image or container failure): retry, never score."""


def _no_pull(self, repository, *a, **kw):
    raise GradingInfraError(f"image {repository} is missing; not pulling it")


# the harness would pull a missing instance image in the middle of a run. Patched on the class, since client.images
# is a new collection on every access; also covers the CLI below.
docker.models.images.ImageCollection.pull = _no_pull

_create = docker.models.containers.ContainerCollection.create


def _create_host_net(self, image, command=None, **kw):
    # grading containers share the host's network namespace, as in the paper's runs (the images used there set proxy
    # variables pointing at localhost:8080, so tests got no internet)
    kw.setdefault("network_mode", "host")
    return _create(self, image, command, **kw)


docker.models.containers.ContainerCollection.create = _create_host_net

THREAD_ENV = {"OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8"}  # also the agent's env
TIMEOUT_MARK = "Test timed out after"
_LOCKS: dict[str, threading.Lock] = {}
_GUARD = threading.Lock()


def with_eval_env(spec):
    spec.eval_script_list[:0] = [f"export {k}={v}" for k, v in THREAD_ENV.items()]
    if spec.repo == "sphinx-doc/sphinx":
        spec.eval_script_list.insert(0, "export PYTEST_ADDOPTS=-rA")  # tox.ini forwards it
    return spec


def docker_client():
    # under many parallel episodes (and checkpoint commits) the daemon can answer in tens of seconds: docker's 60 s
    # default timed out gradings, so the timeout is raised and the API version pinned (skips the /version round trip).
    # It exceeds grade_one's test timeout, since a silent test run must end as the harness's `timeout` verdict.
    client = docker.from_env(version="1.44", timeout=2100)
    client.ping()
    return client


def _lock(iid: str) -> threading.Lock:
    # the harness names its logger after the instance id, so two arms of one instance must not overlap
    with _GUARD:
        return _LOCKS.setdefault(iid, threading.Lock())


def grade_one(instance: dict, patch: str, run_id: str, log_root: Path, model_name: str = "flexamem",
              timeout: int = 1800) -> dict:
    """Run the hidden tests for one prediction. Returns the harness report for this instance with
    `graded_as`: "report" (tests ran), "apply_failed" or "timeout" (both unresolved). Raises
    GradingInfraError when there is no verdict. run_id must be unique per attempt: the harness reuses
    any report.json already in its log directory."""
    iid = instance["instance_id"]
    log_dir = Path(log_root) / "run_evaluation" / run_id / model_name / iid
    with _lock(iid):
        shutil.rmtree(log_dir, ignore_errors=True)
        _rev.RUN_EVALUATION_LOG_DIR = Path(log_root) / "run_evaluation"
        spec = with_eval_env(make_test_spec(instance, namespace="swebench"))
        pred = {"instance_id": iid, "model_name_or_path": model_name, "model_patch": patch}
        client = docker_client()
        try:  # fail with a clear message before the harness reaches its (disabled) pull
            client.images.get(spec.instance_image_key)
        except docker.errors.ImageNotFound as e:
            raise GradingInfraError(f"{iid}: image {spec.instance_image_key} is missing; not pulling it") from e
        _rev.run_instance(spec, pred, rm_image=False, force_rebuild=False, client=client,
                          run_id=run_id, timeout=timeout)
    if (log_dir / LOG_REPORT).exists():
        return {**json.loads((log_dir / LOG_REPORT).read_text())[iid], "graded_as": "report"}
    log = (log_dir / LOG_INSTANCE).read_text(errors="replace") if (log_dir / LOG_INSTANCE).exists() else ""
    if APPLY_PATCH_FAIL in log:
        return {"resolved": False, "patch_successfully_applied": False, "graded_as": "apply_failed"}
    if TIMEOUT_MARK in log:
        return {"resolved": False, "patch_successfully_applied": True, "graded_as": "timeout"}
    raise GradingInfraError(f"{iid}: the harness reached no verdict; see {log_dir / LOG_INSTANCE}")


if __name__ == "__main__":
    import swebench.harness.test_spec.test_spec as _ts  # run_module re-imports make_test_spec from here

    _ts.make_test_spec = lambda *a, **kw: with_eval_env(make_test_spec(*a, **kw))
    runpy.run_module("swebench.harness.run_evaluation", run_name="__main__", alter_sys=True)
