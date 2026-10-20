"""Core records. A Unit is a complete stretch of interaction (never splits a turn, sentence, or an
action from its observations); a Note is the memory written for one unit; a MergeNode compresses
consecutive history nodes while keeping references to its children."""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Unit:
    uid: str               # "u<order>"
    order: int
    source: str            # opaque session id (LongMemEval) or "traj" (SWE)
    time: str              # timestamp string, "" if none
    span: str              # human-readable position, e.g. "session s7, turns 1-6" / "steps 12-14"
    text: str              # raw transcript; this is what an expansion shows verbatim
    tokens: int            # tokens of `text`
    messages: list = field(default_factory=list)  # SWE only: the original chat messages
    msg_tokens: int = 0    # SWE only: tokens of `messages` as sent to the agent model


@dataclass
class Note:
    uid: str
    order: int
    source: str
    time: str
    span: str
    raw: str                                    # immutable after writing
    raw_tokens: int
    short: str = ""                             # Flow short record / ReadAgent gist
    keywords: list = field(default_factory=list)
    tags: list = field(default_factory=list)
    desc: str = ""                              # Flow retrieval description / A-MEM "context"
    links: list = field(default_factory=list)
    emb: Any = None
    chunk_embs: Any = None                      # A-MEM: embeddings of consecutive raw chunks; Flow (full_embed):
                                                # windows of the retrieval text after the first
    vecs: Any = None                            # MemGAS only: {granularity: embedding matrix}
    digest: Any = None                          # flow_noevo_digest: (names called, REPL names bound), shown after short

    @classmethod
    def from_unit(cls, u: Unit) -> "Note":
        return cls(uid=u.uid, order=u.order, source=u.source, time=u.time, span=u.span,
                   raw=u.text, raw_tokens=u.tokens)


@dataclass
class MergeNode:
    nid: str
    children: list          # node ids (note uids or merge nids), chronological
    level: int
    text: str
    first: int              # first / last note order covered
    last: int
