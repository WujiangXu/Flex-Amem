"""Local sentence embedder shared by every method (A-MEM's default all-MiniLM-L6-v2)."""
import os
import threading

import numpy as np

from flexamem.llm.ledger import get_ledger

_MODEL = None
_LOCK = threading.Lock()
NAME = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"  # the snapshot every run of the paper used
# torch defaults to one thread per core; with many runs on one many-core machine a 3-text call took 172 ms instead
# of 15 ms, and every worker of a run waits on _LOCK behind it
THREADS = int(os.getenv("FLEXAMEM_EMBED_THREADS", "8"))


WINDOW = 256  # word pieces the model reads, [CLS] and [SEP] included; the rest of an input is cut off


def _model():
    global _MODEL
    if _MODEL is None:
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(THREADS)
        _MODEL = SentenceTransformer(NAME, device="cpu", revision=REVISION)
    return _MODEL


def embed(texts: list[str], phase: str = "embed") -> np.ndarray:
    """L2-normalised embeddings, shape (len(texts), 384). Inputs past WINDOW word pieces are truncated."""
    with _LOCK:
        out = _model().encode(list(texts), batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    get_ledger().record(phase=phase, model=NAME, n_texts=len(texts))
    return np.asarray(out, dtype=np.float32)


def windows(text: str) -> list[str]:
    """Pieces of text the model reads whole. A text that fits is returned unchanged; otherwise its lines are packed
    greedily, and a line longer than the window is split between words, at any whitespace (only a single longer word
    is still cut). A piece without word pieces never starts a window, so no window is empty."""
    limit = WINDOW - 2
    with _LOCK:
        tok = _model().tokenizer
        n = lambda t: len(tok(t, add_special_tokens=False)["input_ids"])
        if n(text) <= limit:
            return [text]
        out, cur, used = [], "", 0
        for line in text.split("\n"):
            k = n(line)
            for j, (w, k) in enumerate([(line, k)] if k <= limit else [(w, n(w)) for w in line.split()]):
                if cur and k and used + k > limit:
                    out.append(cur)
                    cur, used = "", 0
                cur = cur + ("\n" if j == 0 else " ") + w if cur else w
                used += k
        out.append(cur)
    return out


def embed_windows(texts: list[str], phase: str = "embed") -> list[np.ndarray]:
    """Per text, the embeddings of its windows, shape (k, 384); row 0 of a text that fits is embed([text])[0]."""
    pieces = [windows(t) for t in texts]
    e = embed([p for ps in pieces for p in ps], phase)
    out, i = [], 0
    for ps in pieces:
        out.append(e[i:i + len(ps)])
        i += len(ps)
    return out


def embed_one(text: str, phase: str = "embed") -> np.ndarray:
    return embed([text], phase)[0]
