"""A-MEM baselines sharing one (repaired) A-MEM build.

AMemRetrieval: pinned + recent + retrieved raw notes (each hit followed by its links) until the
allowance is full - how papers citing A-MEM deploy it outside conversation.
SummaryAMem: the key control. A rolling summary of evicted history (OpenHands' summarizing
condenser prompt) compresses the past, and the same A-MEM retriever independently recovers raw details, which
are appended after the summary. The summary gets `split` of the allowance and lends what it leaves.
"""
from flexamem.interface import MemoryMethod, Query
from flexamem.llm.client import get_llm
from flexamem.llm.ledger import pmap
from flexamem.llm.tokens import count, truncate
from flexamem.memory.linker import amem_evolve
from flexamem.memory.prompts import summarization_call
from flexamem.memory.store import NoteStore
from flexamem.memory.writer import amem_analyze
from flexamem.render.renderer import Rendered, render_retrieval
from flexamem.render.views import PieceIndex
from flexamem.types import Note, Unit

SUMMARY_HEADER = "[Summary of the earlier interaction]\n"
EVENT_BATCH_TOKENS = 12000  # max event tokens per summarizer call


def amem_build(store: NoteStore, units: list[Unit], workers: int = 8) -> list[dict]:
    notes = [Note.from_unit(u) for u in units]
    pmap(amem_analyze, notes, workers)
    return [amem_evolve(store, n) for n in notes]


class AMemRetrieval(MemoryMethod):
    name = "amem"

    def __init__(self, store: NoteStore | None = None, workers=8, zoom=None, recent=False):
        """zoom=E: fit-or-degrade plugged into A-MEM's retrieval (an ablation): a retrieved note whose raw text does not
        fit gets an excerpt of at most E tokens of its best-matching pieces. recent: the excerpts rank pieces against
        the query's newest unit, as FlexA-MEM's do on SWE and AppWorld (same writer, FlexA-MEM's renderer setting;
        AppWorld 2026-10-11, deviation 39)."""
        self.store = store if store is not None else NoteStore()
        self.workers = workers
        self.evo = []
        self.zoom, self.recent = zoom, recent

    def write(self, units):
        self.evo += amem_build(self.store, units, self.workers)

    def render(self, query: Query, allowance: int, ref_allowance: int | None = None):
        excerpt = None
        if self.zoom is not None:
            pieces = PieceIndex.of(self.store)
            pieces.update(self.store)
            text = query.recent if self.recent and query.recent else query.text
            excerpt = pieces.excerpter(self.store, text, query.emb, self.zoom, record=False)
        return render_retrieval(self.store, query.emb, allowance, excerpt=excerpt)

    def stats(self):
        return {"notes": len(self.store), "evolved": sum(e.get("evolved", False) for e in self.evo),
                "links": sum(len(n.links) for n in self.store.notes.values())}


def summarize(prev: str, events: list[str], unit_ids=()) -> str:
    """Fold events into the rolling summary, batching so each call sees at most EVENT_BATCH_TOKENS."""
    batch, size = [], 0
    for e in events:
        if batch and size + count(e) > EVENT_BATCH_TOKENS:
            prev = get_llm().call("memory", summarization_call(prev, batch), phase="summarize", cache=True).strip()
            batch, size = [], 0
        batch.append(e)
        size += count(e)
    if batch:
        prev = get_llm().call("memory", summarization_call(prev, batch), phase="summarize", cache=True).strip()
    return prev


def fit_summary(summary: str, max_tokens: int) -> tuple[str, dict]:
    """Re-compress the summary if it exceeds its share (one cached call); truncate as a last resort."""
    st = {"summary_compressed": False, "summary_truncated": False}
    if count(SUMMARY_HEADER + summary) <= max_tokens:
        return summary, st
    words = max(20, int((max_tokens - count(SUMMARY_HEADER)) * 0.7))
    prompt = summarization_call("", [summary]) + f"\nThe summary must be at most {words} words."
    summary = get_llm().call("memory", prompt, phase="summarize", cache=True).strip()
    st["summary_compressed"] = True
    if count(SUMMARY_HEADER + summary) > max_tokens:
        summary = truncate(summary, max(0, max_tokens - count(SUMMARY_HEADER) - 2))
        st["summary_truncated"] = True
    return summary, st


class SummaryAMem(MemoryMethod):
    name = "summary_amem"

    def __init__(self, store: NoteStore | None = None, summary: str = "", split=0.5, workers=8,
                 persist_compression=True, order=None):
        """order(store, q) -> notes to try, in order (the Summary + Flow notes control passes the main arm's candidates);
        default A-MEM's search_agentic order."""
        self.order = order
        self.store = store if store is not None else NoteStore()
        self.summary = summary
        self.split, self.workers = split, workers
        self.persist_compression = persist_compression  # SWE: the compressed summary replaces the rolling one
        self.evo = []

    def write(self, units):
        self.evo += amem_build(self.store, units, self.workers)
        self.summary = summarize(self.summary, [u.text for u in units])

    def render(self, query: Query, allowance: int, ref_allowance: int | None = None):
        # the persisted compression is sized by the step-independent allowance; a dip below it only
        # shortens this render's copy
        perm = max(allowance, ref_allowance) if ref_allowance is not None else allowance
        if self.persist_compression and self.summary and perm > 0:
            self.summary, st_perm = fit_summary(self.summary, int(perm * self.split))
        else:
            st_perm = {}
        if allowance <= 0:
            return Rendered("", 0, {"persisted": st_perm})
        summary, st = fit_summary(self.summary, int(allowance * self.split)) if self.summary else ("", {})
        st["persisted"] = st_perm
        prefix = [SUMMARY_HEADER + summary] if summary else []
        r = render_retrieval(self.store, query.emb, allowance, prefix_blocks=prefix,
                             order=None if self.order is None else self.order(self.store, query.emb))
        r.stats.update(st, summary_tokens=count(prefix[0]) if prefix else 0)
        return r

    def stats(self):
        return {"notes": len(self.store), "summary_tokens": count(self.summary)}
