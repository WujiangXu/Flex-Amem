"""AFM, Adaptive Focus Memory (Cruz 2025, arXiv:2511.12712), reproduced from the paper as a baseline (decision
2026-10-11_locomo_afm): the closest prior work, per-message fidelity tiers under a fixed token budget.

Before every call each stored unit (AFM's "message") gets a score from three signals (paper §3.2):
- sim: cosine of the unit and the query. AFM embeds each message once with text-embedding-3-small; here every arm's
  MiniLM embedder, whose window is short, so a unit scores its best 150-token chunk (as rag_raw does).
- recency: 0.5^(k/h), h = 12, k = t - i in units: from the newest unit (an agent's, in the recent window, Query.now) or,
  for a question after the history (LongMemEval, LoCoMo), from one past the last stored unit.
- importance: one LLM call per unit at write time, CRITICAL, RELEVANT or TRIVIAL (memory role).
score = 1.0 if CRITICAL; max(0, sim) * (0.4 + 0.4 recency) if RELEVANT; max(0, sim) * 0.25 recency if TRIVIAL.
Tiers (§3.3): score >= 0.45 full, >= 0.25 compressed, else stub. Packing runs oldest to newest under the allowance: full
falls back to compressed then stub, compressed to stub, and a unit whose stub does not fit is dropped. The compressed form
is an LLM rewrite in at most COMPRESS_TOKENS tokens (§3.4, LLMCompressor), made inside packing only when a unit needs it
(its full text does not fit, or it is intended compressed), then reused. The thresholds were set for text-embedding-3-small
cosines; MiniLM's best-chunk cosines run on another scale, and each render logs the intended tiers.

Choices the paper leaves open (its reference code was not consulted): the two prompts (memory/prompts.py), the
compression target (100 tokens) and the stub ("[<time · span> · omitted]"). Like every arm, AFM packs the units outside
the recent window into the memory allowance; the pinned task and recent window are the harness's."""
import weakref

import numpy as np

from flexamem.interface import MemoryMethod, Query
from flexamem.llm.client import get_llm
from flexamem.llm.embed import embed
from flexamem.llm.ledger import pmap
from flexamem.llm.tokens import chunks, count
from flexamem.memory import prompts as P
from flexamem.memory.linker import CHUNK_TOKENS
from flexamem.memory.store import NoteStore
from flexamem.render import format as F
from flexamem.render.renderer import Rendered
from flexamem.types import Note

HALF_LIFE = 12
TAU_HIGH, TAU_MID = 0.45, 0.25
# thresholds on MiniLM's scale (decision 2026-10-11_afm_calibration): the paper's tau_high lies above the 99th percentile of
# AFM's scores on LoCoMo conv-26 with MiniLM (median 0.21, 90th percentile 0.32), so almost nothing is shown in full
CALIBRATED = {"afm_t30_15": (0.30, 0.15), "afm_t20_10": (0.20, 0.10), "afm_t15_07": (0.15, 0.07),
              # LongMemEval's ~80-unit haystacks score lower still (unit median 0.05, per-question maximum median 0.28)
              "afm_t08_05": (0.08, 0.05)}
COMPRESS_TOKENS = 100
_VECS = weakref.WeakKeyDictionary()  # store -> {uid: chunk embeddings of its raw text}


def classify(note: Note) -> str:
    r = get_llm().call("memory", P.AFM_IMPORTANCE.format(text=note.raw), phase="write",
                       schema=P.AFM_IMPORTANCE_SCHEMA, cache=True, unit=note.uid)
    return r["label"]


def compress(note: Note) -> str:
    out = get_llm().call("memory", P.AFM_COMPRESS.format(tokens=COMPRESS_TOKENS, text=note.raw), phase="summarize",
                         cache=True, unit=note.uid)
    return str(out).strip()


def score(label: str, sim: float, recency: float) -> float:
    if label == "CRITICAL":
        return 1.0
    return max(0.0, sim) * ((0.4 + 0.4 * recency) if label == "RELEVANT" else 0.25 * recency)


class AFMMethod(MemoryMethod):
    name = "afm"

    def __init__(self, store: NoteStore | None = None, *, workers: int = 8, tau_high: float = TAU_HIGH,
                 tau_mid: float = TAU_MID):
        """tau_high, tau_mid: the paper's thresholds by default; CALIBRATED: candidates for MiniLM's score scale, chosen
        on one LoCoMo conversation (decision 2026-10-11_afm_calibration)."""
        self.store = store if store is not None else NoteStore()
        self.tau_high, self.tau_mid = tau_high, tau_mid
        self.workers = workers
        self.labels: dict[str, str] = {}
        self.summaries: dict[str, str] = {}
        self.vecs = _VECS.setdefault(self.store, {})

    def write(self, units) -> None:
        notes = [Note.from_unit(u) for u in units]
        for n, lab in zip(notes, pmap(classify, notes, self.workers)):
            self.labels[n.uid] = lab
            self.store.add(n)

    def _sims(self, q) -> dict[str, float]:
        st = self.store
        todo = [u for u in st.order if u not in self.vecs]
        if todo:
            cs = [chunks(st.notes[u].raw, CHUNK_TOKENS) or [st.notes[u].raw] for u in todo]
            e, i = embed([c for x in cs for c in x]), 0
            for u, x in zip(todo, cs):
                self.vecs[u], i = e[i:i + len(x)], i + len(x)
        return {u: float(np.max(self.vecs[u] @ q)) for u in st.order}

    def render(self, query: Query, allowance: int, ref_allowance: int | None = None) -> Rendered:
        st, allowance = self.store, max(allowance, 0)
        if allowance <= 0 or not st.order:
            return Rendered("", 0, {})
        sims, n = self._sims(query.emb), len(st.order)
        now = query.now if query.now >= 0 else st.notes[st.order[-1]].order + 1
        tier = {}
        for u in st.order:
            s = score(self.labels[u], sims[u], 0.5 ** ((now - st.notes[u].order) / HALF_LIFE))
            tier[u] = "full" if s >= self.tau_high else "compressed" if s >= self.tau_mid else "stub"

        def form(u, level):
            nt, w = st.notes[u], F.when(st.notes[u])
            if level == "full":
                return F.raw_block(nt)
            if level == "compressed":
                if u not in self.summaries:  # made when first needed, then reused (§3.3)
                    self.summaries[u] = compress(nt)
                return f"[{w} · compressed] {self.summaries[u]}"
            return f"[{w} · omitted]"

        order = {"full": ("full", "compressed", "stub"), "compressed": ("compressed", "stub"), "stub": ("stub",)}
        blocks, left, shown = [], allowance, []
        for u in st.order:  # chronological, oldest first (§3.3)
            for level in order[tier[u]]:
                text = form(u, level)
                c = F.block_tokens(text)
                if c <= left:
                    blocks.append(text)
                    shown.append((u, level))
                    left -= c
                    break
        text = F.join(blocks)
        while blocks and count(text) > allowance:  # a merge across a join: drop the newest block
            blocks.pop()
            shown.pop()
            text = F.join(blocks)
        tok = count(text) if blocks else 0
        assert tok <= allowance, (tok, allowance)
        lv = [lev for _, lev in shown]
        return Rendered(text, tok, {"tokens": tok, "n_notes": n, "full": lv.count("full"),
                                    "compressed": lv.count("compressed"), "stub": lv.count("stub"),
                                    "dropped": n - len(shown),
                                    "intended": {t: sum(v == t for v in tier.values()) for t in ("full", "compressed", "stub")}})

    def stats(self) -> dict:
        lab = list(self.labels.values())
        return {"notes": len(self.store), **{k.lower(): lab.count(k) for k in ("CRITICAL", "RELEVANT", "TRIVIAL")}}
