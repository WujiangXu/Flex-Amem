"""Common interface for memory methods. A method receives units as they leave the recent window
(write) and produces memory text for a query under a token allowance (render). The benchmark adapters
own everything else (pinned task, recent window, reader/agent prompt), identically for every method.
ref_allowance (SWE only) is a step-independent allowance that sizes permanent operations (merges, a
persisted summary), so a transient dip in the allowance never forces an irreversible compression."""
from dataclasses import dataclass

import numpy as np

from flexamem.render.renderer import Rendered
from flexamem.types import Unit


@dataclass
class Query:
    text: str
    emb: np.ndarray
    date: str = ""  # LongMemEval question date (MemGAS's filter prompt shows it)
    recent: str = ""  # SWE: the newest unit's text, what the agent is doing now (A-MEM Zoom's excerpts can rank by it)
    now: int = -1     # agents: the newest unit's order (it sits in the recent window); AFM's recency counts from it


class MemoryMethod:
    name = "base"

    def write(self, units: list[Unit]) -> None:
        raise NotImplementedError

    def render(self, query: Query, allowance: int, ref_allowance: int | None = None) -> Rendered:
        raise NotImplementedError

    def reserve(self, units: list[Unit]) -> int:
        """Tokens the memory needs whatever the recent window takes, given the units it may hold (all but the newest);
        the policy leaves them by shrinking the newest unit first. Only Flow's explored-so-far floor reserves."""
        return 0

    def stats(self) -> dict:
        return {}
