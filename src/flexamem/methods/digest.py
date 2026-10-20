"""Deterministic digests of an agent's units for three exploratory variants of the main arm (2026-10-08, design
deviation 36), each built from the actions with no model call, reusing the action trace's parsing (methods/trace.py).
They follow record_content's findings: at AppWorld's decisive steps the memory mostly supplies exact names (APIs,
docs) rather than REPL handles, 8.9% of compacted steps show no memory at all, and docs are re-read even when a
record names them.
- names(messages): what each step of a unit called (app APIs, the docs and API lists it read; SWE: the command and the
  files it named) and the REPL names bound by cells that ran, in order of first use. A doc read with non-literal
  arguments is named by the docs its output printed; names AppWorld reports as missing ("No APIs with name ...") and
  calls after the line a failed cell stopped at are left out.
- index_line: the names of every unit in memory, grouped by app; the explored-so-far floor (flow_noevo_index).
- format.names_text: one unit's names after its record, the union of its leaves' names after a merge
  (flow_noevo_digest).
- sign_docs(raw): every complete AppWorld API doc printed in a unit's outputs (head and tail of a long output too)
  replaced in place by one signature line (method, path, parameters with type, required mark, default, constraints and
  the sort or date format their descriptions state, response fields); everything else verbatim (flow_noevo_docsig)."""
import ast
import json
import re

from flexamem.memory.segmenter import parse_steps
from flexamem.methods.trace import _chain, _name, outcome, step_entry
from flexamem.render import format as F

INDEX_HEAD = "[explored so far]"
INDEX_WORDS = 200  # cap on the index line, so the floor stays a floor
NOAPI = re.compile(r"No APIs with name (\w+) found for the (\w+) app|No API named '(\w+)' found in the (\w+) app")
NOAPP = re.compile(r"No APIs found for the app name (\w+)")
LINE = re.compile(r'File "<python-input>", line (\d+), in <module>')
PRINTED_DOC = re.compile(r"""(["'])app_name\1: (["'])(\w+)\2,\s*(["'])api_name\4: (["'])(\w+)\5,\s*(["'])path\7""")
BARE = "api_docs.show_api_doc"  # trace._name of a doc read whose arguments are not both literals
BLOCK = re.compile(r"(<(output|output_head|output_tail)>)(.*?)(</\2>)", re.S)
OBJ = re.compile(r"(?m)^[ \t]*\{")
VALID = re.compile(r"prefixed with \+/-.*?Valid attributes: (.+?)\.(?:\s|$)", re.S)
DATE = re.compile(r"YYYY-MM-DD(?:\|HH:MM:SS)?")
JSON_TYPE = {"str": "string", "int": "integer", "float": "number", "bool": "boolean"}


def _late(code: str, out: str) -> set[str]:
    """Names of a failed python cell's apis calls that come only after the line it stopped at (AppWorld runs and numbers
    the stripped code)."""
    lines = LINE.findall(out)
    if not lines or not outcome("python", out).startswith("error"):
        return set()
    try:
        tree = ast.parse(code.strip())
    except (SyntaxError, ValueError, RecursionError):
        return set()
    at, ran, late = int(lines[-1]), set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and len(c := _chain(n.func)) == 3 and c[0] == "apis":
            (ran if n.lineno <= at else late).add(_name(c[1], c[2], n))
    return late - ran


def names(messages: list[dict]) -> tuple[list[str], list[str]]:
    """(names called, REPL names bound by cells that ran) over a unit's steps, each in order of first use."""
    called, bound = [], []
    for i, s in enumerate(parse_steps(messages)):
        e = step_entry(s, i)
        outs = {m.get("tool_call_id"): m.get("content") or "" for m in s if m.get("role") == "tool"}
        missing, apps, late = set(), set(), set()
        for tc in s[0].get("tool_calls") or []:
            out = outs.get(tc.get("id")) or ""
            missing |= {f"{a or c}.{b or d}" for b, a, d, c in NOAPI.findall(out)}
            apps |= set(NOAPP.findall(out))
            if tc["function"]["name"] == "python":
                try:
                    code = json.loads(tc["function"]["arguments"]).get("code", "")
                except (json.JSONDecodeError, TypeError, AttributeError):
                    code = ""
                late |= _late(code, out) if isinstance(code, str) else set()
        nm = e["names"]
        if BARE in nm:  # the docs this step printed stand for its non-literal doc reads
            text = "\n".join(outs.values())
            docs = list(dict.fromkeys(f"doc:{m.group(3)}.{m.group(6)}" for m in PRINTED_DOC.finditer(text)))
            if docs:
                k = nm.index(BARE)
                nm = nm[:k] + docs + nm[k + 1:]

        def gone(n):
            app = n.removeprefix("doc:").removeprefix("apis:").split(".")[0]
            return (n.removeprefix("doc:") in missing or n in late
                    or (n.startswith(("doc:", "apis:")) and app in apps))
        called += [n for n in nm if not gone(n)]
        bound += [v for v in e["vars"] if len(v) >= 3]  # loop counters and the like left out
    return list(dict.fromkeys(called)), list(dict.fromkeys(bound))


def index_line(called: list[str], bound: list[str]) -> str:
    """'[explored so far] app list read; spotify: API list, doc login, login, ...; variables: ...' over every unit in
    memory (format.names_text), at most INDEX_WORDS words."""
    if not called and not bound:
        return ""
    words = f"{INDEX_HEAD} {F.names_text(called, bound)}".split()
    return " ".join(words[:INDEX_WORDS]) + (" …" if len(words) > INDEX_WORDS else "")


def _shape(x) -> str:
    """A response schema's field names: {a, b{c, d}} for an object, [..] for a list of objects, [type] for a list of
    scalars."""
    if isinstance(x, dict):
        return "{" + ", ".join(k + (_shape(v) if isinstance(v, (dict, list)) else "") for k, v in x.items()) + "}"
    if isinstance(x, list):
        if not x:
            return "[]"
        inner = _shape(x[0]) if isinstance(x[0], (dict, list)) else JSON_TYPE.get(type(x[0]).__name__, "value")
        return f"[{inner}]"
    return ""


def _hint(p: dict) -> str:
    """What only a parameter's description states and a call needs: sort_by's valid attributes, the format of a date
    parameter without a default (a default already shows it)."""
    de = p.get("description") or ""
    if p.get("name") == "sort_by" and (m := VALID.search(de)):
        return "+/- then " + m.group(1).replace(" and ", ", ")
    if p.get("default") in (None, "", []) and (m := DATE.search(de)):
        return m.group(0)
    return ""


def signature(d: dict) -> str:
    ps = []
    for p in d["parameters"]:
        s = f"{p.get('name', '?')}{'*' if p.get('required') else ''}: {p.get('type', '?')}"
        if p.get("default") not in (None, "", []):
            s += f" = {json.dumps(p['default'])}"
        cons = [str(c) for c in p.get("constraints") or []] + ([h] if (h := _hint(p)) else [])
        if cons:
            s += f" ({'; '.join(cons)})"
        ps.append(s)
    ok = (d.get("response_schemas") or {}).get("success")
    return (f"[API doc {d['app_name']}.{d['api_name']}] {d.get('method', '?')} {d.get('path', '?')}: "
            f"{(d.get('description') or '').strip()} Parameters (* required): {', '.join(ps) or 'none'}."
            + (f" Returns: {_shape(ok)}." if isinstance(ok, (dict, list)) else ""))


def _is_doc(v) -> bool:
    return (isinstance(v, dict) and isinstance(v.get("api_name"), str) and isinstance(v.get("app_name"), str)
            and isinstance(v.get("parameters"), list)
            and all(isinstance(p, dict) and isinstance(p.get("constraints") or [], list) for p in v["parameters"])
            and isinstance(v.get("response_schemas") or {}, dict) and isinstance(v.get("description") or "", str))


def _sign(body: str) -> str:
    """Every complete JSON doc that starts a line, replaced by its signature; the rest verbatim."""
    dec, out, i = json.JSONDecoder(), [], 0
    for m in OBJ.finditer(body):
        s = m.end() - 1
        if s < i:  # inside a value already decoded
            continue
        try:
            v, e = dec.raw_decode(body, s)
        except json.JSONDecodeError:
            continue
        try:
            rep = signature(v) if _is_doc(v) else body[s:e]
        except (AttributeError, TypeError, KeyError):  # doc-like JSON the agent built itself: keep it verbatim
            rep = body[s:e]
        out.append(body[i:s] + rep)
        i = e
    return "".join(out) + body[i:]


def sign_docs(raw: str) -> str:
    """A unit's raw text with every complete API doc in its outputs (and in the head and tail of a long one) replaced by
    its signature line. Docs printed as Python reprs stay as they are."""
    return BLOCK.sub(lambda m: m.group(1) + _sign(m.group(3)) + m.group(4), raw)
