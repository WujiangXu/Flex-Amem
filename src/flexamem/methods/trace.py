"""Action trace: a memory that writes nothing with an LLM (insight experiment B, 2026-10-08). Each step becomes one
deterministic line:
- what it did: the app API calls of its code (AppWorld: apis.<app>.<api>(...), literal arguments shortened, others by
  their source) or the command (SWE);
- the outcome: ok with the output on one line, or the error's last line;
- the REPL names the code bound (`vars:`), an explicit index of the handles that persist across steps.

API docs and lists are digested from their JSON: a doc keeps its method, path and parameters (* required, =default),
a list keeps every name, so the trace records what the exploration found, as an LLM record would.

The render follows render_flow's policy for the history (deviation 9), so only the writer differs from flow_noevo_noexp
(LLM-written records in the same layout). Merges are permanent and sized by max(allowance, ref_allowance): groups of up
to 4 adjacent entries at the lowest level, oldest first, become one line ("steps a-b: called <what>; vars ...; errors
k", at most MAX_WORDS words). A render whose allowance dips below that hides the oldest lines for that render only, and
a line that does not fit is dropped, never cut (review of 2026-10-08)."""
import ast
import json
import re

from flexamem.interface import MemoryMethod, Query
from flexamem.llm.tokens import count
from flexamem.memory.segmenter import ACTION_TOOLS, parse_steps
from flexamem.render.renderer import Rendered

NOTE = re.compile(r"<(warning|elided_chars)>.*?</\1>", re.S)  # what the harness adds to long outputs, not output
TAG = re.compile(r"</?(?:output|output_head|output_tail|returncode|exception)>")
RC = re.compile(r"<returncode>(-?\d+)</returncode>")
CALL = re.compile(r"apis\.(\w+)\.(\w+)\s*\(")
FILE = re.compile(r"[\w./-]+\.(?:py|pyi|txt|cfg|toml|ini|json|yaml|yml|md|rst|c|h|cpp|js|ts)\b")
ARG, ACT, OUT, DOC, VARS = 24, 200, 80, 300, 8  # characters per argument, action, output, doc digest; names in a vars list
GROUP, MAX_WORDS = 4, 150             # entries per merge, words per merged line (as memory/merge.py)


def _cut(s: str, n: int) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n - 3] + "..."


def _chain(f) -> list[str]:
    out = []
    while isinstance(f, ast.Attribute):
        out.append(f.attr)
        f = f.value
    if isinstance(f, ast.Name):
        out.append(f.id)
    return out[::-1]


def _arg(node) -> str:
    try:
        return _cut(repr(ast.literal_eval(node)), ARG)
    except (ValueError, SyntaxError, TypeError, RecursionError):
        return _cut(ast.unparse(node), ARG)


def _name(app: str, api: str, n) -> str:
    """What a call did, for a merged line: the doc or API list it read (a literal target), else app.api."""
    if app == "api_docs":
        lits = [a.value for a in [*n.args, *(k.value for k in n.keywords)] if isinstance(a, ast.Constant)
                and isinstance(a.value, str)]
        if api == "show_api_doc" and len(lits) == 2:
            return f"doc:{lits[0]}.{lits[1]}"
        if api == "show_api_descriptions" and len(lits) == 1:
            return f"apis:{lits[0]}"
        if api == "show_app_descriptions":
            return "apps"
    return f"{app}.{api}"


class _Bound(ast.NodeVisitor):
    """Names a cell binds at module level, so they persist in the REPL (function bodies and comprehensions left out)."""

    def __init__(self):
        self.names = []

    def visit_FunctionDef(self, n):
        self.names.append(n.name)

    visit_AsyncFunctionDef = visit_ClassDef = visit_FunctionDef

    def visit_Lambda(self, n):
        pass

    visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = visit_Lambda

    def visit_Name(self, n):
        if isinstance(n.ctx, ast.Store):
            self.names.append(n.id)

    def visit_Import(self, n):
        self.names += [(a.asname or a.name).split(".")[0] for a in n.names]

    visit_ImportFrom = visit_Import


def parse_code(code: str) -> tuple[list[tuple[str, str]], list[str]]:
    """(what each apis.<app>.<api>(...) call did, the call shortened) in source order, and the names the code binds."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return [(f"{a}.{b}", f"{a}.{b}(...)") for a, b in CALL.findall(code)], []
    calls = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and len(c := _chain(n.func)) == 3 and c[0] == "apis":
            args = [_arg(a) for a in n.args] + [f"{k.arg}={_arg(k.value)}" for k in n.keywords if k.arg]
            calls.append((n.lineno, n.col_offset, _name(c[1], c[2], n), f"{c[1]}.{c[2]}({', '.join(args)})"))
    b = _Bound()
    b.visit(tree)
    return [(name, call) for *_, name, call in sorted(calls)], list(dict.fromkeys(b.names))


def _first(text: str) -> str:
    return next((x.strip() for x in text.splitlines() if x.strip() and not x.strip().startswith("#")), "")


def outcome(tool: str, out: str | None) -> str:
    if out is None:
        return "no output"
    if tool == "bash":
        rc = RC.search(out)
        body = _first(TAG.sub("", NOTE.sub("", RC.sub("", out))))
        return f"rc={rc.group(1) if rc else '?'}" + (f": {_cut(body, OUT)}" if body else "")
    body = TAG.sub("", NOTE.sub("", out)).strip()
    if body.startswith("Execution failed"):
        last = [x.strip() for x in body.splitlines() if x.strip()]
        return f"error: {_cut(last[-1], OUT)}" if last else "error"
    if not body or body == "Execution successful.":
        return "ok"
    return f"ok: {digest(body)}"


def _doc(d: dict) -> str:
    ps = [p.get("name", "?") + ("*" if p.get("required") else "") +
          (f"={p['default']!r}" if p.get("default") not in (None, "", []) else "") for p in d.get("parameters") or []]
    return f"{d.get('method', '?')} {d.get('path', '?')}({', '.join(ps)})"


def digest(body: str) -> str:
    """The output on one line. Printed API docs and API or app lists (AppWorld's api_docs JSON) are parsed: a doc keeps
    its method, path and parameters, a list every name; anything else its first OUT characters."""
    dec, i, parts = json.JSONDecoder(), 0, []
    while i < len(body):
        while i < len(body) and body[i].isspace():
            i += 1
        if i >= len(body):
            break
        try:
            v, i = dec.raw_decode(body, i)
        except json.JSONDecodeError:
            return _cut(body, OUT) if not parts else _cut("; ".join(parts) + " ...", DOC)
        if isinstance(v, dict) and "api_name" in v and "parameters" in v:
            parts.append(_doc(v))
        elif isinstance(v, list) and v and all(isinstance(x, dict) and "name" in x for x in v):
            parts.append("[" + ", ".join(str(x["name"]) for x in v) + "]")
        else:
            return _cut(body, OUT)
    return _cut("; ".join(parts), DOC) if parts else _cut(body, OUT)


def _bash_names(cmd: str) -> list[str]:
    last = re.split(r"&&|;|\|\|", cmd)[-1].split()
    return ([last[0]] if last else []) + list(dict.fromkeys(FILE.findall(cmd)))[:3]


def step_entry(step: list[dict], n: int) -> dict:
    a = step[0]
    outs = {m.get("tool_call_id"): m.get("content") or "" for m in step if m.get("role") == "tool"}
    parts, names, bound, errors = [], [], [], 0
    for tc in a.get("tool_calls") or []:
        tool = tc["function"]["name"]
        try:
            args = json.loads(tc["function"]["arguments"])
        except (json.JSONDecodeError, TypeError):
            args = {}
        code = args.get(ACTION_TOOLS.get(tool, "command"), "") if isinstance(args, dict) else ""
        code = code if isinstance(code, str) else str(code)
        res = outcome(tool, outs.get(tc.get("id")))
        if tool == "python":
            calls, vs = parse_code(code)
            names += [nm for nm, _ in calls]
            bound += vs if not res.startswith("error") else []  # a failed cell may not have bound its names
            act = "; ".join(c for _, c in calls) or _first(code)
        else:
            names += _bash_names(code)
            act = _first(code)
        errors += res.startswith("error") or (res.startswith("rc=") and not res.startswith("rc=0"))
        parts.append(f"{_cut(act, ACT)} -> {res}")
    if not parts:  # a step without a tool call (a plain or rejected reply)
        parts.append(_cut(_first(a.get("content") or "") or "(no action)", ACT))
    bound = list(dict.fromkeys(bound))
    text = f"[step {n}] " + " | ".join(parts) + (f" | vars: {', '.join(bound[:VARS])}" if bound else "")
    return {"a": n, "b": n, "level": 0, "text": text, "names": list(dict.fromkeys(names)), "vars": bound,
            "errors": errors}


def _merge(group: list[dict], level: int) -> dict:
    names = list(dict.fromkeys(x for e in group for x in e["names"]))
    bound = list(dict.fromkeys(x for e in group for x in e["vars"]))
    errors = sum(e["errors"] for e in group)
    a, b = group[0]["a"], group[-1]["b"]
    text = (f"[steps {a}-{b}] called: {', '.join(names) or 'nothing'}" + (f"; vars: {', '.join(bound)}" if bound else "")
            + (f"; errors: {errors}" if errors else ""))
    words = text.split()
    if len(words) > MAX_WORDS:
        text = " ".join(words[:MAX_WORDS]) + " ..."
    return {"a": a, "b": b, "level": level, "text": text, "names": names, "vars": bound, "errors": errors}


class TraceMemory(MemoryMethod):
    name = "trace"

    def __init__(self):
        self.frontier: list[dict] = []  # the history: step lines and the permanent merges of older ones
        self.steps = 0

    def write(self, units) -> None:
        for u in units:
            first = int(re.search(r"\d+", u.span).group())  # "step a" or "steps a-b" (segmenter.swe_units)
            for i, s in enumerate(parse_steps(u.messages)):
                self.frontier.append(step_entry(s, first + i))
                self.steps += 1

    def _join(self, es) -> str:
        return "\n".join(e["text"] for e in es)

    def render(self, query: Query, allowance: int, ref_allowance: int | None = None) -> Rendered:
        allowance = max(allowance, 0)
        perm = max(allowance, ref_allowance) if ref_allowance is not None else allowance
        es, merges = self.frontier, 0
        while len(es) > 1 and count(self._join(es)) > perm:  # permanent: sized by perm, never undone
            low = min(e["level"] for e in es)
            i = next(k for k, e in enumerate(es) if e["level"] == low)
            j = i
            while j < len(es) and es[j]["level"] == low and j - i < GROUP:
                j += 1
            if j - i == 1:  # a lone entry at the lowest level waits for the next one up
                es[i] = {**es[i], "level": low + 1}
                continue
            es[i:j] = [_merge(es[i:j], low + 1)]
            merges += 1
        drop = 0  # this render only: hide the oldest lines until the text fits (a dip, or a line too long for perm)
        while drop < len(es) and count(self._join(es[drop:])) > allowance:
            drop += 1
        text = self._join(es[drop:])
        tok = count(text) if text else 0
        assert tok <= allowance, f"trace render {tok} > allowance {allowance}"
        return Rendered(text, tok, {"entries": self.steps, "shown": len(es) - drop, "merges": merges,
                                    "overflow_dropped": drop, "dip": allowance < perm})

    def stats(self) -> dict:
        return {"steps": self.steps, "lines": len(self.frontier)}
