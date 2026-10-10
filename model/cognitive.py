"""
cognitive.py — two mechanisms from the memory literature, used by llm.py.

Both operate on the context `run.py` hands us. Neither touches retrieval, the
index, or k: the set of chunks in play is fixed and identical for every
submission, and these only decide how that fixed set is presented and how the
evidence it yields is converted into a report/withhold decision.

1. SPREADING ACTIVATION (Collins & Loftus 1975; Anderson's ACT-R activation
   equation, of which Anderson & Bower 1972 is the ancestor). Multi-hop
   questions are answerable only through a *bridge entity*: the question names
   an entity, one document links it to a second entity, and a second document
   states the property being asked about. A similarity ranking cannot see that
   second document, because it is not similar to the question — it is similar to
   something the first document said. Activation spreading from the question
   through the documents reaches it in two steps, which is exactly what the
   baseline's distance-based view of the context misses.

2. BOUNDED EVIDENCE ACCUMULATION (Ratcliff's diffusion model; recognition-memory
   decisions as a race to a threshold). Koriat & Goldsmith's control process
   compares confidence to a report criterion, but says nothing about how that
   confidence is formed. A sequential-sampling account supplies it: evidence
   increments accumulate until one of two boundaries is crossed. Two asymmetric
   boundaries are exactly the right shape here, because reporting and
   withholding are not symmetric acts under this scorer.

Kept deliberately small: Wilson & Collins (2019) warn that a model's complexity
should be forced by the phenomena it explains, and there are only two phenomena
here (failure to incorporate, failure to ignore).
"""
from __future__ import annotations

import math
import re
import string
from typing import Dict, List, Sequence, Tuple

_PUNCT = set(string.punctuation)

# Function words carry no associative weight; they would link every document to
# every other one and wash the activation out.
_STOP = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "but",
    "is", "was", "were", "are", "be", "been", "by", "with", "as", "that", "this",
    "it", "its", "from", "which", "who", "whom", "whose", "what", "when", "where",
    "he", "she", "they", "his", "her", "their", "him", "them", "had", "has",
    "have", "not", "no", "s", "also", "after", "before", "into", "than", "then",
    "there", "here", "about", "one", "two", "first", "new", "other", "such",
}


def _terms(text: str) -> set[str]:
    """Content terms, plus multi-word capitalised spans as entity proxies."""
    ents = {m.group(0).lower()
            for m in re.finditer(r"\b[A-Z][\w'’-]*(?:\s+[A-Z][\w'’-]*)+", text or "")}
    flat = "".join(ch if ch not in _PUNCT else " " for ch in (text or "").lower())
    toks = {t for t in flat.split() if len(t) > 2 and t not in _STOP}
    return toks | ents


def _assoc(a: set[str], b: set[str]) -> float:
    """Association strength. Normalised by the smaller set so that a short
    document is not penalised for having fewer terms to share."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def spread(question: str, chunks: Sequence[dict], *, decay: float = 0.6,
           steps: int = 2) -> Tuple[List[float], float]:
    """Spread activation from the question through the documents.

    Returns per-chunk activation and a *closure* score: how much of the final
    activation arrived indirectly, through another document, rather than
    straight from the question. High closure means a bridge was found; near-zero
    closure on a multi-hop question is evidence that no chain exists in this
    context, which is a signal about abstention that does not come from asking
    the model how confident it feels.
    """
    n = len(chunks)
    if n == 0:
        return [], 0.0

    q = _terms(question)
    docs = [_terms(f"{c.get('title') or ''} {c.get('text') or ''}") for c in chunks]

    # Direct activation: what the question itself reaches.
    direct = [_assoc(q, d) for d in docs]
    # Document-to-document association, excluding terms the question already
    # supplies — a shared term the question named is not a *new* link.
    w = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = _assoc(docs[i] - q, docs[j] - q)
            w[i][j] = w[j][i] = s

    act = list(direct)
    for _ in range(steps):
        nxt = []
        for i in range(n):
            inflow = sum(w[i][j] * act[j] for j in range(n) if j != i)
            nxt.append(direct[i] + decay * inflow)
        m = max(nxt) or 1.0
        act = [a / m for a in nxt]

    dmax = max(direct) or 1.0
    norm_direct = [d / dmax for d in direct]
    # Closure: activation gained beyond what the question reached directly.
    gained = sorted((act[i] - norm_direct[i] for i in range(n)), reverse=True)
    closure = max(0.0, sum(gained[:2]) / 2)
    return act, closure


def order_by_activation(chunks: Sequence[dict], act: Sequence[float]) -> List[int]:
    """Indices of `chunks` ordered most-activated first. Ties keep the original
    order so the arrangement stays deterministic."""
    return sorted(range(len(chunks)), key=lambda i: (-act[i], i))


# --- bounded accumulation ----------------------------------------------------

def _logit(p: float, lo: float = 0.02, hi: float = 0.98) -> float:
    p = min(max(p, lo), hi)
    return math.log(p / (1 - p))


def accumulate(increments: Sequence[Tuple[float, float]]) -> float:
    """Sum weighted evidence increments into a single accumulated position.

    Each increment is (evidence in [0,1], weight). Evidence is mapped to
    log-odds before summing, so independent sources combine additively the way
    a drift process does, rather than being averaged.
    """
    return sum(wt * _logit(e) for e, wt in increments)


def decide(position: float, report_bound: float,
           withhold_bound: float) -> Tuple[bool, float]:
    """Two-boundary decision. Returns (report?, position).

    The boundaries are asymmetric by design: under this scorer a withheld answer
    on an answerable question costs the full F1 of that item, while a volunteered
    answer on an unanswerable one costs the full 1.0. `report_bound` is the
    criterion proper; `withhold_bound` sets how much contrary evidence is needed
    to actively refuse rather than merely fail to reach criterion.
    """
    if position >= report_bound:
        return True, position
    if position <= -withhold_bound:
        return False, position
    # Between the boundaries the process has not resolved; the conservative
    # default under a scorer that punishes unsupported answers is to withhold.
    return False, position
