"""How memory entries appear in context. Every method uses the same headers, so differences between
methods come from what is shown, not from how it is formatted."""
import re

from flexamem.llm.tokens import count
from flexamem.types import MergeNode, Note

SEP = "\n\n"
SEP_TOKENS = 1


def when(n: Note) -> str:
    return f"{n.time} · {n.span}" if n.time else n.span


FILE = re.compile(r"[\w./-]+\.(?:py|pyi|txt|cfg|toml|ini|json|yaml|yml|md|rst|c|h|cpp|js|ts)\b")  # as methods/trace.py
DIGEST_NAMES = 30  # names after a record (flow_noevo_digest); a merge's union keeps MERGE_DIGEST_NAMES
MERGE_DIGEST_NAMES = 40


def names_text(called: list[str], bound: list[str], cap: int | None = None) -> str:
    """What units did, grouped by app: 'app list read; spotify: API list, doc login, login; variables: x, y', from the
    action trace's names (apps, apis:<app>, doc:<app>.<api>, <app>.<api>; SWE: commands and files), each once, in order
    of first use, at most cap names (flow_noevo_index and flow_noevo_digest)."""
    called, bound = list(dict.fromkeys(called)), list(dict.fromkeys(bound))
    more = max(0, len(called) + len(bound) - cap) if cap is not None else 0
    if more:
        called = called[:cap]
        bound = bound[:cap - len(called)]
    by_app, other = {}, []
    for c in called:
        if c == "apps":
            other.append("app list read")
        elif c.startswith("apis:"):
            by_app.setdefault(c[5:], []).append("API list")
        elif c.startswith("doc:") and "." in c:
            app, api = c[4:].split(".", 1)
            by_app.setdefault(app, []).append(f"doc {api}")
        elif re.fullmatch(r"\w+\.\w+", c) and not FILE.fullmatch(c):  # an app API, not a SWE file name
            app, api = c.split(".", 1)
            by_app.setdefault(app, []).append(api)
        else:
            other.append(c)
    parts = other + [f"{a}: {', '.join(xs)}" for a, xs in by_app.items()]
    if bound:
        parts.append("variables: " + ", ".join(bound))
    return "; ".join(parts) + (f" (+{more} more)" if more else "")


def compressed(n: Note, digest: bool = True) -> str:
    rec = f"[{n.uid} · {when(n)}] {n.short}"
    return f"{rec} [{names_text(*n.digest, cap=DIGEST_NAMES)}]" if digest and n.digest and any(n.digest) else rec


def expanded(n: Note) -> str:
    return f"[{n.uid} · {when(n)} · full record]\n{n.raw}"


def excerpt(n: Note, lines: list[str]) -> str:
    """Graded views: the short record, then the kept raw lines with markers for what was left out."""
    return f"[{n.uid} · {when(n)} · excerpt] {n.short}\n" + "\n".join(lines)


def raw_excerpt(n: Note, lines: list[str]) -> str:
    """An excerpt headed like a raw block, for notes shown without a record (A-MEM, zero-build retrieval)."""
    return f"[{when(n)} · excerpt]\n" + "\n".join(lines)


def raw_block(n: Note) -> str:
    return f"[{when(n)}]\n{n.raw}"


def gist_block(n: Note, page: int) -> str:
    return f"<Page {page}> [{when(n)}]\n{n.short}"


def lookup_page(n: Note, page: int) -> str:
    """A page in the SWE lookup prompt: without the step span, whose numbers the lookup mistook for page numbers."""
    return f"<Page {page}>\n{n.short}"


def page_block(n: Note, page: int) -> str:
    return f"<Page {page}> [{when(n)} · full text]\n{n.raw}"


def memgas_block(n: Note) -> str:
    """A MemGAS candidate: the raw unit with its summary and keywords (the filter sees all granularities)."""
    return f"[{when(n)}]\n{n.raw}\n[summary] {n.short}\n[keywords] {'; '.join(n.keywords)}"


def fragment_raw(n: Note, fid: str) -> str:
    """Sculptor fragments: the ID the model's tools take, then the raw text, a summary or a fold marker."""
    return f"[fragment {fid} · {when(n)}]\n{n.raw}"


def fragment_summary(n: Note, fid: str, focus: str, summary: str) -> str:
    return f"[fragment {fid} · {when(n)} · summarized ({focus})] {summary}"


def fragment_folded(n: Note, fid: str) -> str:
    return f"[fragment {fid} · {when(n)} · folded]"


def _steps(span: str) -> tuple[str, str]:
    nums = re.findall(r"\d+", span)
    return (nums[0], nums[-1]) if nums else (span, span)


def merged(m: MergeNode, first: Note, last: Note) -> str:
    if first.source == "traj":
        rng = f"steps {_steps(first.span)[0]}-{_steps(last.span)[1]}"
    else:
        rng = f"{first.time} … {last.time}" if first.time else f"{first.span} … {last.span}"
    return f"[{first.uid}–{last.uid} · {rng} · condensed] {m.text}"


def block_tokens(b: str) -> int:
    return count(b) + SEP_TOKENS


def join(blocks: list[str]) -> str:
    return SEP.join(blocks)
