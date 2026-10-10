"""
rag.py — retrieval and context-building helpers for the RAG assignment.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

class STEmbedder:
    """Wrapper for sentence-transformers. Adds the right prefixes for
    e5-/bge-style models automatically."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5",
                 device: str | None = None, batch_size: int = 256):
        from sentence_transformers import SentenceTransformer
        self.name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        low = model_name.lower()
        if "e5" in low:
            self.q_prefix, self.d_prefix = "query: ", "passage: "
        elif "bge" in low:
            self.q_prefix, self.d_prefix = "Represent this sentence for searching relevant passages: ", ""
        else:
            self.q_prefix = self.d_prefix = ""

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        prefix = self.q_prefix if is_query else self.d_prefix
        vecs = self.model.encode([prefix + t for t in texts],
                                 batch_size=self.batch_size,
                                 convert_to_numpy=True,
                                 show_progress_bar=len(texts) > 5000,
                                 normalize_embeddings=True)
        return vecs.astype(np.float32)


def _get_embedder(model_name: str) -> STEmbedder:
    return STEmbedder(model_name)


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def load_jsonl(path: Path) -> List[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def doc_to_text(doc: dict) -> str:
    return f"{doc['title']}. {doc['text']}" if doc.get("title") else doc["text"]


def build_index(corpus_path: Path, index_path: Path,
                embedder_spec: str = "BAAI/bge-small-en-v1.5",
                device: str | None = None) -> None:
    """Embed the corpus and save a numpy matrix plus metadata."""
    corpus = load_jsonl(corpus_path)
    emb = _get_embedder(embedder_spec)
    mat = emb.encode([doc_to_text(d) for d in corpus], is_query=False)
    np.save(index_path, mat)
    meta = {
        "embedder": embedder_spec,
        "n_docs": len(corpus),
        "dim": int(mat.shape[1]),
        "doc_ids": [d["doc_id"] for d in corpus],
    }
    (index_path.with_suffix(".meta.json")).write_text(json.dumps(meta))
    print(f"indexed {len(corpus):,} paragraphs -> {index_path} (dim={mat.shape[1]})")


class DenseRetriever:
    """Cosine-similarity retriever over a pre‑built index."""

    def __init__(self, corpus_path: Path, index_path: Path, device: str | None = None):
        self.corpus = load_jsonl(corpus_path)
        self.by_id = {d["doc_id"]: d for d in self.corpus}
        self.mat = np.load(index_path)
        meta = json.loads(index_path.with_suffix(".meta.json").read_text())
        self.doc_ids = meta["doc_ids"]
        self.embedder = _get_embedder(meta["embedder"])
        assert len(self.doc_ids) == self.mat.shape[0], "index/corpus mismatch — rebuild"

    def search(self, queries: Sequence[str], k: int = 5) -> List[List[str]]:
        qv = self.embedder.encode(list(queries), is_query=True)
        scores = qv @ self.mat.T
        kk = min(k, scores.shape[1])
        idx = np.argpartition(-scores, kk - 1, axis=1)[:, :kk]
        rows = []
        for r in range(idx.shape[0]):
            order = idx[r][np.argsort(-scores[r, idx[r]])]
            rows.append([self.doc_ids[i] for i in order])
        return rows

def _distractors(question: dict, retrieved_ids: List[str], by_id: Dict[str, dict],
                 banned: set, k: int) -> List[dict]:
    """Pick distractors: injected hard ones first, then retrieved filler."""
    out, seen = [], set(banned)
    for did in list(question.get("hard_distractor_ids") or []) + list(retrieved_ids):
        if len(out) >= k:
            break
        if did in seen or did not in by_id:
            continue
        seen.add(did)
        out.append(by_id[did])
    return out


def build_context(condition: str, question: dict, retrieved_ids: List[str],
                  by_id: Dict[str, dict], k: int,
                  rng: random.Random) -> tuple[List[dict], Dict[str, Any]]:
    """Return (chunks, metadata) for a given context condition."""
    gold = [g for g in question["gold_doc_ids"] if g in by_id]
    gold_set = set(gold)
    meta: Dict[str, Any] = {
        "expected_answer": question["answer"],
        "expect_abstain": not question.get("answerable", True),
    }

    if condition == "closed_book":
        chunks = []

    elif condition == "oracle":
        # Unanswerable items have no gold; use local docs instead.
        src = gold or [d for d in question.get("local_doc_ids", []) if d in by_id][:k]
        chunks = [by_id[g] for g in src]
        rng.shuffle(chunks)

    elif condition == "retrieved":
        chunks = [by_id[d] for d in retrieved_ids[:k] if d in by_id]

    elif condition == "gold_plus_noise":
        # All gold paragraphs plus k distractors, shuffled.
        chunks = [by_id[g] for g in gold] + _distractors(
            question, retrieved_ids, by_id, gold_set, k)
        rng.shuffle(chunks)

    elif condition == "no_gold":
        # Keep only the gold paragraphs that set up the question; drop the
        # answer-bearing one and any leaky paragraphs.
        partial = [d for d in (question.get("partial_gold_ids") or []) if d in by_id]
        banned = gold_set | set(question.get("leaky_doc_ids") or [])
        chunks = [by_id[d] for d in partial] + _distractors(
            question, retrieved_ids, by_id, banned, max(0, k - len(partial)))
        rng.shuffle(chunks)
        meta["expect_abstain"] = True
        meta["expected_answer"] = None

    else:
        raise ValueError(f"unknown condition {condition!r}")

    return chunks, meta

def main() -> None:
    ap = argparse.ArgumentParser(description="Build the retrieval index.")
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus.jsonl"))
    ap.add_argument("--index", type=Path, default=Path("data/index.npy"))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    build_index(args.corpus, args.index, device=args.device)

if __name__ == "__main__":
    main()