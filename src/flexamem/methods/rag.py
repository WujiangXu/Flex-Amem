"""Zero-build retrieval baselines (exploratory, after the mock review of 2026-10-06): memory with no LLM call at all,
built from the raw units the moment they arrive. They answer "why not just retrieve raw text?".

  rag_raw     raw units ranked by their best chunk's MiniLM cosine to the query (A-MEM's raw-chunk index without
              A-MEM's notes, links or evolution: 150-token chunks, memory/linker.CHUNK_TOKENS), taken in rank order
              while they fit and shown chronologically, as A-MEM's retrieval shows its raw notes
  rag_pieces  A-MEM Zoom's pieces (render/views.py) of every unit, ranked as Zoom ranks an excerpt's pieces (BM25 over the
              query's content words with IDF over every piece, ties to MiniLM closeness, then the earlier piece), the
              best pieces that fit, shown per unit chronologically under the unit's time and span, verbatim, with
              "[...]" for what was left out

Neither shows anything for units it does not retrieve: no records, no lines, no history. On SWE (write()) a unit is
stored as raw text the moment it leaves the recent window, with no LLM call; the query is the main arm's (task + newest
unit)."""
import weakref

from flexamem.interface import MemoryMethod, Query
from flexamem.llm.embed import embed
from flexamem.llm.tokens import chunks, count
from flexamem.memory.linker import CHUNK_TOKENS
from flexamem.memory.store import NoteStore
from flexamem.render import format as F
from flexamem.render.renderer import Rendered, _finalize
from flexamem.render.views import PieceIndex, excerpt_lines
from flexamem.types import Note


_VECS = weakref.WeakKeyDictionary()  # store -> {uid: chunk embeddings of its raw text}, shared by the store's renders


class _RawStore(MemoryMethod):
    """A store of raw units, filled by write() with no LLM call (SWE) or handed over whole (LongMemEval)."""

    def __init__(self, store: NoteStore | None = None):
        self.store = store if store is not None else NoteStore()

    def write(self, units) -> None:
        for u in units:
            self.store.add(Note.from_unit(u))

    def stats(self) -> dict:
        return {"notes": len(self.store)}


class RawRetrieval(_RawStore):
    name = "rag_raw"

    def __init__(self, store: NoteStore | None = None):
        super().__init__(store)
        self.vecs = _VECS.setdefault(self.store, {})

    def render(self, query: Query, allowance: int, ref_allowance: int | None = None) -> Rendered:
        st, allowance = self.store, max(allowance, 0)
        todo = [u for u in st.order if u not in self.vecs]
        if todo:
            cs = [chunks(st.notes[u].raw, CHUNK_TOKENS) or [st.notes[u].raw] for u in todo]
            e, i = embed([c for x in cs for c in x]), 0
            for u, x in zip(todo, cs):
                self.vecs[u], i = e[i:i + len(x)], i + len(x)
        best = {u: float((self.vecs[u] @ query.emb).max()) for u in st.order}
        chosen, room = [], allowance
        for u in sorted(st.order, key=lambda x: (-best[x], st.notes[x].order)):
            d = F.block_tokens(F.raw_block(st.notes[u]))
            if d <= room:
                chosen.append(u)
                room -= d

        def build(k_exp, k_drop):
            return F.join([F.raw_block(st.notes[u]) for u in sorted(chosen[:k_exp], key=lambda x: st.notes[x].order)])

        r = _finalize(build, len(chosen), allowance, {}, 0)
        r.stats["retrieved"] = sorted(chosen[:r.stats["n_expanded"]], key=lambda x: st.notes[x].order)
        return r


class PieceRetrieval(_RawStore):
    name = "rag_pieces"

    def __init__(self, store: NoteStore | None = None):
        super().__init__(store)
        self.pieces = PieceIndex.of(self.store)

    def render(self, query: Query, allowance: int, ref_allowance: int | None = None) -> Rendered:
        st, ix, allowance = self.store, self.pieces, max(allowance, 0)
        ix.update(st)
        ex = ix.excerpter(st, query.text, query.emb, 0, record=False)
        ranked = sorted(((u, i) for u in st.order for i in range(len(ix.pieces[u]))),
                        key=lambda ui: (-ex.score(*ui), st.notes[ui[0]].order, ui[1]))

        def build(k, _=0):
            keep = {}
            for u, i in ranked[:k]:
                keep.setdefault(u, set()).add(i)
            return F.join([F.raw_excerpt(st.notes[u], excerpt_lines(ix, st, u, keep[u]))
                           for u in sorted(keep, key=lambda x: st.notes[x].order)])

        lo, hi = 0, len(ranked)  # the most pieces whose render fits (piece by piece, so a bisection on the count)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            lo, hi = (mid, hi) if count(build(mid)) <= allowance else (lo, mid - 1)
        r = _finalize(build, lo, allowance, {}, 0)
        r.stats.update(pieces=r.stats["n_expanded"], units=len({u for u, _ in ranked[:r.stats["n_expanded"]]}))
        return r
