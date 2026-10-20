"""Per-unit note writing. Every call here depends on the unit's content only, so results are
cached (cost is still reported as if uncached)."""
import re

from flexamem.llm.client import get_llm
from flexamem.memory import prompts as P
from flexamem.types import Note


def _clean_list(xs, lo: int, hi: int) -> list:
    out = []
    for x in xs:
        x = str(x).strip()
        if x and x not in out:
            out.append(x)
    return out[:hi] if len(out) >= lo else out


def flow_write(note: Note) -> None:
    """Fill short / keywords / tags / retrieval_desc. Raw stays untouched."""
    r = get_llm().call("memory", P.FLOW_WRITER.format(time=note.time or "n/a", span=note.span, raw=note.raw),
                       phase="write", schema=P.FLOW_WRITER_SCHEMA, cache=True, unit=note.uid)
    note.short = r["short"].strip()
    note.keywords = _clean_list(r["keywords"], 1, 8)
    note.tags = _clean_list(r["tags"], 1, 5)
    note.desc = r["retrieval_desc"].strip()
    if not note.short:
        raise ValueError(f"empty short record for {note.uid}")


def flow_embed_text(n: Note) -> str:
    return f"{n.short}\n{n.desc}\nkeywords: {', '.join(n.keywords)}\ntags: {', '.join(n.tags)}"


def amem_analyze(note: Note) -> None:
    r = get_llm().call("memory", P.AMEM_ANALYZE + note.raw, phase="write", schema=P.AMEM_ANALYZE_SCHEMA,
                       cache=True, unit=note.uid)
    note.keywords = [str(k) for k in r["keywords"]]
    note.desc = r["context"] or "General"
    note.tags = [str(t) for t in r["tags"]]


def amem_embed_text(n: Note) -> str:
    # Repair: A-MEM embeds content+context+keywords+tags; with a 256-word-piece embedder and 1-2k-token
    # units the metadata would be truncated away, so it goes first (docs/logs/debugging_journal.md).
    return f"{n.desc}\nkeywords: {', '.join(n.keywords)}\ntags: {', '.join(n.tags)}\n{n.raw}"


def readagent_gist(note: Note) -> None:
    g = get_llm().call("memory", P.READAGENT_GIST.format(text=note.raw), phase="gist", cache=True, unit=note.uid)
    note.short = g.strip()


def parse_pages(text: str) -> list[int]:
    """ReadAgent lookup reply: 'I want to look up Page [7, 12] to ...' -> [7, 12] (LLM order)."""
    m = re.search(r"\[([\d,\s]+)\]", text)
    if not m:
        return []
    out = []
    for x in m.group(1).split(","):
        x = x.strip()
        if x.isdigit() and int(x) not in out:
            out.append(int(x))
    return out
