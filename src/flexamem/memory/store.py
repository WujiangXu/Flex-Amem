"""Note store: chronological notes, a dense index over their embeddings, and link bookkeeping.
A note may carry extra chunk embeddings (A-MEM's raw text, or the windows of a Flow note's retrieval text past the
first, both longer than the embedder reads); its similarity is then the maximum over its main and chunk vectors.
A query may likewise have several vectors (rows); a note then scores its best pair."""
import gzip
import json
import os
from pathlib import Path

import numpy as np

from flexamem.types import Note

_KEEP = object()


class NoteStore:
    def __init__(self):
        self.notes: dict[str, Note] = {}
        self.order: list[str] = []          # uids, chronological
        self._mat = None                    # cached embedding matrix: main and chunk vectors of all notes
        self._owner = None                  # row -> index into self.order

    def __len__(self):
        return len(self.order)

    def add(self, note: Note):
        assert note.uid not in self.notes, note.uid
        assert not self.order or self.notes[self.order[-1]].order < note.order, "notes must arrive in order"
        self.notes[note.uid] = note
        self.order.append(note.uid)
        self._mat = None

    def set_emb(self, uid: str, emb, chunk_embs=_KEEP):
        self.notes[uid].emb = emb
        if chunk_embs is not _KEEP:
            self.notes[uid].chunk_embs = chunk_embs
        self._mat = None

    def link(self, a: str, b: str):
        if a == b:
            return
        if b not in self.notes[a].links:
            self.notes[a].links.append(b)
        if a in self.notes[b].links:
            return
        self.notes[b].links.append(a)

    def sims(self, q) -> dict[str, float]:
        if not self.order:
            return {}
        if self._mat is None:
            vecs, owner = [], []
            for i, u in enumerate(self.order):
                n = self.notes[u]
                for v in [n.emb] + list(n.chunk_embs if n.chunk_embs is not None else []):
                    vecs.append(v)
                    owner.append(i)
            self._mat, self._owner = np.stack(vecs), np.asarray(owner)
        q = np.asarray(q, dtype=np.float32)
        s = self._mat @ q if q.ndim == 1 else (self._mat @ q.T).max(axis=1)
        best = np.full(len(self.order), -np.inf, dtype=np.float32)
        np.maximum.at(best, self._owner, s)
        return dict(zip(self.order, best.tolist()))

    def nearest(self, q, k: int, exclude=()) -> list[str]:
        s = self.sims(q)
        return [u for u, _ in sorted(s.items(), key=lambda x: -x[1]) if u not in exclude][:k]

    def neighbors_in_time(self, uid: str) -> list[str]:
        i = self.order.index(uid)
        return [self.order[j] for j in (i - 1, i + 1) if 0 <= j < len(self.order)]

    # ---- persistence (LongMemEval build cache) ----
    def save(self, path, extra: dict | None = None):
        rows = []
        for u in self.order:
            n = self.notes[u]
            d = dict(n.__dict__)
            d["emb"] = None if n.emb is None else np.asarray(n.emb).round(6).tolist()
            d["chunk_embs"] = None if n.chunk_embs is None else np.asarray(n.chunk_embs).round(6).tolist()
            d["vecs"] = None if n.vecs is None else {g: np.asarray(v).round(6).tolist() for g, v in n.vecs.items()}
            rows.append(d)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with gzip.open(tmp, "wt") as f:
            json.dump({"notes": rows, "extra": extra or {}}, f)
        os.replace(tmp, path)  # a crash mid-write never leaves a partial build behind

    @classmethod
    def load(cls, path) -> tuple["NoteStore", dict]:
        with gzip.open(path, "rt") as f:
            d = json.load(f)
        st = cls()
        for r in d["notes"]:
            r["emb"] = None if r["emb"] is None else np.asarray(r["emb"], dtype=np.float32)
            if r.get("chunk_embs") is not None:
                r["chunk_embs"] = np.asarray(r["chunk_embs"], dtype=np.float32)
            if r.get("vecs") is not None:
                r["vecs"] = {g: np.asarray(v, dtype=np.float32) for g, v in r["vecs"].items()}
            st.add(Note(**r))
        return st, d.get("extra", {})
