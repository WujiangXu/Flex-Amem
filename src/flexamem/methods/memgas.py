"""MemGAS (Xu et al., ICLR 2026): multi-granularity memory association and selection, under the shared budget.

Each unit is kept at four granularities: its raw text (MemGAS's session level), its turns (LongMemEval: a user turn
with the replies that follow it; SWE: a step), an LLM summary and LLM keywords (the paper's two generation prompts).
When a unit is written, its similarity to every earlier unit is computed per granularity, a two-component Gaussian
mixture splits those similarities, and the earlier units in the higher-mean component (the accept set) are linked to
it. At render, an entropy router weights the granularities by the inverse entropy of the query's similarity
distribution over units, the ALPHA best units by the weighted score seed a Personalized PageRank over the association
graph, and units are taken in PageRank order, each with its summary and keywords, until they fill the allowance
(the paper's same-cost comparison, App. D.2, over-retrieves and truncates at the token budget). One LLM call filters
them to the parts relevant to the query, preserving original tokens; its reply is the memory text, cut to the
allowance as a last resort. Adaptations to the shared units, embedder and budget: docs/method/design.md, deviation 26."""
import re
import warnings

import networkx as nx
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

from flexamem.interface import MemoryMethod, Query
from flexamem.llm.client import get_llm
from flexamem.llm.embed import embed
from flexamem.llm.ledger import pmap
from flexamem.llm.tokens import chunks, count, truncate
from flexamem.memory.linker import CHUNK_TOKENS
from flexamem.memory.prompts import MEMGAS_FILTER, MEMGAS_KEYWORDS, MEMGAS_NOUNS, MEMGAS_SUMMARY
from flexamem.memory.store import NoteStore
from flexamem.render import format as F
from flexamem.render.renderer import Rendered, normalize
from flexamem.types import Note, Unit

GRANULARITIES = ("session", "turn", "summary", "keywords")
LAMBDA = 0.2        # entropy temperature (paper, Appendix F)
ALPHA = 15          # PageRank seed units (paper, Appendix F)
DAMPING = 0.85      # PageRank damping; the paper gives none (networkx default)
FILTER_INPUT = 1.0  # candidates fill this multiple of the allowance before the filter call (App. D.2)
TURN_SPLIT = {"lme": re.compile(r"\n(?=user: )"), "swe": re.compile(r"\n(?=\[step \d+\] assistant:)")}


def _vecs(texts: list[str]) -> np.ndarray:
    """One vector per text: the normalised mean of its CHUNK_TOKENS chunks, so a text past the embedder's window is
    represented whole (one chunk for summaries, keywords and most turns)."""
    pieces, owner = [], []
    for i, t in enumerate(texts):
        cs = chunks(t, CHUNK_TOKENS) or [""]
        pieces += cs
        owner += [i] * len(cs)
    e, owner = embed(pieces), np.asarray(owner)
    return np.stack([normalize(e[owner == i].mean(axis=0)) for i in range(len(texts))])


def parse_keywords(text: str) -> list[str]:
    text = re.sub(r"^\s*keywords?\s*:\s*", "", text.strip(), flags=re.I)
    out = []
    for k in re.split(r"[;\n]", text):
        k = k.strip().strip("-*•").strip()
        if k and k not in out:
            out.append(k)
    return out


def accept_set(sims: np.ndarray) -> np.ndarray:
    """Indices of the earlier units in the higher-mean component of a two-component GMM over their similarities to
    the new unit. With a single earlier unit there is nothing to split and it is accepted."""
    if len(sims) < 2:
        return np.arange(len(sims))
    x = sims.reshape(-1, 1).astype(np.float64)
    if np.ptp(x) < 1e-9:  # identical similarities: no second component to find
        return np.arange(len(sims))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gm = GaussianMixture(n_components=2, random_state=0).fit(x)
    return np.flatnonzero(gm.predict(x) == int(np.argmax(gm.means_.ravel())))


def router_weights(sims: dict[str, np.ndarray], lam: float = LAMBDA) -> dict[str, float]:
    """w_g proportional to 1 / H_g, H_g the entropy of softmax(s_g / lam) over units (paper, Eqs. 3-4)."""
    inv = {}
    for g, s in sims.items():
        z = s / lam
        p = np.exp(z - z.max())
        p /= p.sum()
        h = float(-(p * np.log(np.clip(p, 1e-300, None))).sum())
        inv[g] = 1.0 / (h + 1e-9)
    tot = sum(inv.values())
    return {g: v / tot for g, v in inv.items()}


class MemGASMethod(MemoryMethod):
    name = "memgas"

    def __init__(self, store: NoteStore | None = None, domain="lme", workers=8, query_tokens=3000,
                 alpha=ALPHA, lam=LAMBDA, filter_input=FILTER_INPUT):
        self.store = store if store is not None else NoteStore()
        self.domain, self.workers, self.query_tokens = domain, workers, query_tokens
        self.alpha, self.lam, self.filter_input = alpha, lam, filter_input
        self.accepted = {g: 0 for g in GRANULARITIES}  # association edges proposed per granularity, this process

    # ---- write ----
    def _generate(self, note: Note) -> None:
        memory, noun = MEMGAS_NOUNS[self.domain]
        llm = get_llm()
        s = llm.call("memory", MEMGAS_SUMMARY.format(memory=memory, noun=noun, text=note.raw), phase="write",
                     cache=True, unit=note.uid).strip()
        k = llm.call("memory", MEMGAS_KEYWORDS.format(memory=memory, text=note.raw), phase="write", cache=True,
                     unit=note.uid)
        if not s:
            raise ValueError(f"empty MemGAS summary for {note.uid}")
        note.short, note.keywords = s, parse_keywords(k)

    def turns(self, note: Note) -> list[str]:
        return [t for t in TURN_SPLIT[self.domain].split(note.raw) if t.strip()] or [note.raw]

    def write(self, units: list[Unit]) -> None:
        notes = [Note.from_unit(u) for u in units]
        pmap(self._generate, notes, self.workers)
        for n in notes:
            turns = self.turns(n)
            v = _vecs([n.raw, n.short, "; ".join(n.keywords)] + turns)
            n.vecs = {"session": v[:1], "summary": v[1:2], "keywords": v[2:3], "turn": v[3:]}
            n.emb = v[0]
            old = list(self.store.order)
            self.store.add(n)
            for g in GRANULARITIES:
                sims = np.array([float((n.vecs[g] @ self.store.notes[u].vecs[g].T).max()) for u in old])
                for i in accept_set(sims):
                    self.store.link(n.uid, old[i])
                    self.accepted[g] += 1

    # ---- render ----
    def unit_sims(self, q) -> dict[str, np.ndarray]:
        """Query similarity per granularity and unit; a unit's turn similarity is that of its best turn."""
        q = np.asarray(q, dtype=np.float32)
        return {g: np.array([float((self.store.notes[u].vecs[g] @ q).max()) for u in self.store.order])
                for g in GRANULARITIES}

    def rank(self, q) -> tuple[list[str], dict]:
        order = self.store.order
        sims = self.unit_sims(q)
        w = router_weights(sims, self.lam)
        score = sum(w[g] * sims[g] for g in GRANULARITIES)
        seeds = [order[i] for i in np.argsort(-score, kind="stable")[:self.alpha]]
        idx = {u: i for i, u in enumerate(order)}
        pers = {u: max(float(score[idx[u]]), 0.0) + 1e-6 for u in seeds}
        g = nx.Graph()
        g.add_nodes_from(order)
        g.add_edges_from((u, v) for u in order for v in self.store.notes[u].links)
        ppr = nx.pagerank(g, alpha=DAMPING, personalization=pers, max_iter=1000, tol=1e-10)
        ranked = sorted(order, key=lambda u: (-ppr[u], -score[idx[u]], idx[u]))
        return ranked, {"weights": {k: round(v, 4) for k, v in w.items()}, "seeds": seeds}

    def render(self, query: Query, allowance: int, ref_allowance: int | None = None):
        allowance = max(allowance, 0)
        stats = {"n_units": len(self.store.order)}
        if not self.store.order or allowance <= 0:
            return Rendered("", 0, stats)
        ranked, st = self.rank(query.emb)
        room, chosen = int(self.filter_input * allowance), []
        for u in ranked:
            t = F.block_tokens(F.memgas_block(self.store.notes[u]))
            if t <= room:
                chosen.append(u)
                room -= t
        stats.update(st, candidates=chosen)
        if not chosen:
            return Rendered("", 0, {**stats, "tokens": 0})
        chron = sorted(chosen, key=lambda u: self.store.notes[u].order)
        retrieved = F.join([F.memgas_block(self.store.notes[u]) for u in chron])
        prompt = MEMGAS_FILTER[self.domain].format(retrieved_texts="\n" + retrieved, question_date=query.date or "n/a",
                                                   question=truncate(query.text, self.query_tokens))
        # the prompt asks to remove whatever is irrelevant, so an empty reply means no candidate was relevant
        reply = text = get_llm().call("memory", prompt, phase="filter", cache=True, allow_empty=True).strip()
        n = full = count(text)
        cut = allowance
        while n > allowance:  # last resort; a decoded cut can re-tokenize slightly longer, hence the loop
            text = truncate(reply, cut)
            n = count(text)
            cut -= max(1, n - allowance)
        stats.update(candidate_tokens=count(retrieved), filter_tokens=full, filter_truncated=full > allowance,
                     tokens=n)
        return Rendered(text, n, stats)

    def stats(self) -> dict:
        return {"units": len(self.store), "links": sum(len(n.links) for n in self.store.notes.values()) // 2,
                "accepted": dict(self.accepted)}
