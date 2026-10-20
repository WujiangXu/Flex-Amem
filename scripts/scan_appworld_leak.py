"""Leakage scan of AppWorld runs (review of 2026-10-07): the agent's code runs in AppWorld's REPL, whose syntax guard
allows `appworld` and `pathlib`, so code could read a task's ground truth or another arm's evaluation report. Lists every
code action of the runs' trajectories that touches the data, the experiment outputs or AppWorld's internals. Hits are
rerun or excluded before tables are made; the result goes in the run card.
    scan_appworld_leak.py <run_id> [<run_id> ...]"""
import json
import re
import sys

from flexamem.llm.client import ROOT

PATTERN = re.compile(r"ground_truth|private_data|data/tasks|experiments/outputs|APPWORLD_ROOT|\bappworld\b|Task\.load|"
                     r"pathlib|read_text|read_bytes|\bopen\s*\(|\bos\.|scandir|listdir|\bglob\b|rglob|__import__|importlib|"
                     r"evaluat", re.I)

hits, n_actions, n_trajs = [], 0, 0
for rid in sys.argv[1:]:
    for f in sorted((ROOT / "artifacts/runs/appworld" / rid / "trajs").glob("*/*.traj.json")):
        n_trajs += 1
        for m in json.loads(f.read_text())["messages"]:
            for tc in m.get("tool_calls") or []:
                n_actions += 1
                try:
                    code = json.loads(tc["function"]["arguments"]).get("code", "")
                except (json.JSONDecodeError, TypeError):
                    code = tc["function"]["arguments"]
                if isinstance(code, str) and (hit := PATTERN.search(code)):
                    hits.append({"run": rid, "traj": str(f.relative_to(ROOT)), "match": hit.group(0), "code": code[:300]})
print(json.dumps({"trajectories": n_trajs, "code_actions": n_actions, "hits": hits}, indent=1))
