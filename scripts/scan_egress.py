"""List SWE episodes whose agent fetched external content that arrived (journal: "Agent containers reach the internet
through the host proxy"). A fetch is an agent command that itself makes an HTTP request, installs a package or clones
a repository (curl, wget, pip, git clone/fetch, urllib, requests, ...), without a mock in the same command, whose
observation has return code 0 and no network error. Project code that reaches the network on its own (e.g. sphinx
intersphinx during a docs build) is not visible here. Only the final row of each episode is scanned, and rows run
with --no-egress are skipped. Such episodes are rerun with --no-egress.
    scan_egress.py p1_swe_rest385 p1_swe_16k [--json out.json]"""
import argparse
import collections
import json
import re

from flexamem.llm.client import ROOT

FETCH = re.compile(r"\bcurl\s|\bwget\s|\bgit\s+(clone|fetch|pull|ls-remote)\b|\bpip3?\s+(install|download)\b"
                   r"|-m\s+pip\s+(install|download)\b|\b(conda|mamba|apt-get|apt)\s+install\b|urlopen\("
                   r"|\brequests\.(get|post|head)\(|\bhttp\.client\b|socket\.create_connection", re.I)
MOCK = re.compile(r"mock|monkeypatch|fake_urlopen|urlopen\s*=", re.I)
FAILED = re.compile(r"Network is unreachable|URLError|ConnectionError|ProxyError|Connection reset|Could not resolve"
                    r"|Failed to connect|Temporary failure in name resolution|Name or service not known|timed out"
                    r"|No matching distribution|Could not fetch URL|ERROR: Could not|curl: \(\d+\)", re.I)

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--json", default="")
a = ap.parse_args()
hits, n_cmds = [], 0
for run in a.runs:
    rd = ROOT / "artifacts/runs/swe" / run
    last = {}
    for line in (rd / "results.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            last[(r["iid"], r["method"], r["budget"])] = r
    for (iid, method, budget), r in sorted(last.items()):
        traj = ROOT / r["traj"]
        if r.get("egress") is False or not traj.exists():
            continue
        msgs = json.loads(traj.read_text())["messages"]
        obs = {m.get("tool_call_id"): m.get("content") or "" for m in msgs if m.get("role") == "tool"}
        for m in msgs:
            for tc in m.get("tool_calls") or []:
                n_cmds += 1
                args = tc.get("function", {}).get("arguments", "")
                try:
                    cmd = json.loads(args).get("command", args)
                except (ValueError, AttributeError):
                    cmd = args
                if not FETCH.search(cmd) or MOCK.search(cmd):
                    continue
                out = obs.get(tc.get("id"), "")
                out = out if isinstance(out, str) else json.dumps(out)
                if "<returncode>0</returncode>" in out and not FAILED.search(out):
                    hits.append({"run": run, "iid": iid, "method": method, "budget": budget,
                                 "resolved": r.get("resolved"), "hosts": re.findall(r"https?://([^/\s'\"]+)", cmd),
                                 "command": cmd[:400], "output": out[:400]})
print(f"{n_cmds} agent commands scanned in {', '.join(a.runs)}")
eps = collections.defaultdict(list)
for h in hits:
    eps[(h["run"], h["iid"], h["method"], h["budget"])].append(h)
for k, hs in sorted(eps.items()):
    print(*k, f"fetches={len(hs)} resolved={hs[0]['resolved']}", ",".join(sorted({x for h in hs for x in h["hosts"]})))
if a.json:
    with open(a.json, "w") as f:
        json.dump(hits, f, indent=1)
