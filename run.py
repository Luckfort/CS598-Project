"""
run.py — generate answers for the evaluation subsets and write one record file
per subset.

    python run.py                      # both graded subsets, baseline settings

The model is loaded once for both. 
Each record carries the model and k it was produced with.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List

from model.llm import build_messages, get_model, parse_output
from model.rag import DenseRetriever, build_context, build_index, load_jsonl

# Context conditions for testing, and conditions used for actually
# evaluating your model performance (i.e. the grading conditions)
conditions = ["closed_book", "oracle", "retrieved", "gold_plus_noise", "no_gold"]
graded_conditions = ["gold_plus_noise", "no_gold"]

def select_questions(all_questions: List[dict], condition: str,
                     limit: int | None, seed: int) -> List[dict]:
    """Filter questions for a condition."""
    qs = all_questions
    if condition in graded_conditions:
        qs = [q for q in qs if condition in q.get("subsets", graded_conditions)]
    if limit and limit < len(qs):
        keep = set(random.Random(seed).sample(range(len(qs)), limit))
        qs = [q for i, q in enumerate(qs) if i in keep]
    return qs

def run_condition(condition: str, questions: List[dict], by_id: Dict[str, dict],
                  retrieved: Dict[str, List[str]], model, args) -> List[dict]:
    pending: List[Dict[str, Any]] = []
    records: List[dict] = []
    t0 = time.time()

    def flush():
        if not pending:
            return
        raws = model.generate([p["messages"] for p in pending])
        for p, raw in zip(pending, raws):
            parsed = parse_output(raw, p["n_chunks"])
            cited_doc_ids = [p["chunk_ids"][i - 1] for i in parsed["cited"]
                             if 1 <= i <= len(p["chunk_ids"])]
            records.append({
                "qid": p["qid"],
                "condition": condition,
                "model": getattr(model, "name", args.model),
                "k": args.k,
                "question": p["question"],
                "context_doc_ids": p["chunk_ids"],
                "retrieved_doc_ids": p["retrieved"],
                "prediction": parsed["answer"],
                "cited_positions": parsed["cited"],
                "cited_doc_ids": cited_doc_ids,
                "parse_ok": parsed["parse_ok"],
                "expected_answer": p["meta"]["expected_answer"],
                "expect_abstain": p["meta"]["expect_abstain"],
                "raw": raw[:2000],
            })
        pending.clear()

    for n, q in enumerate(questions, 1):
        rng = random.Random(f"{args.seed}:{q['qid']}")
        hits = retrieved.get(q["qid"], [])
        chunks, meta = build_context(condition, q, hits, by_id, args.k, rng)
        pending.append({
            "qid": q["qid"],
            "question": q["question"],
            "messages": build_messages(q["question"], chunks),
            "chunk_ids": [c["doc_id"] for c in chunks],
            "retrieved": hits[: args.k],
            "n_chunks": len(chunks),
            "meta": meta,
        })
        if len(pending) >= args.batch_size:
            flush()
            print(f"  {n}/{len(questions)}  ({time.time()-t0:.0f}s)", end="\r")
    flush()
    return records


def main() -> None:

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", type=Path, default=Path("data/questions.jsonl"))
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus.jsonl"))
    ap.add_argument("--index", type=Path, default=Path("data/index.npy"))
    ap.add_argument("--condition", choices=conditions, nargs="+",
                    default=graded_conditions,
                    help="default: the two graded subsets")
    ap.add_argument("--k", type=int, default=5, help="chunks in context")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--dtype", default="float16",
                    help="auto | float16 | bfloat16 (pre-Ampere GPUs need float16)")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path("runs"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_questions = load_jsonl(args.questions)
    corpus = load_jsonl(args.corpus)
    by_id = {d["doc_id"]: d for d in corpus}

    # Retrieve once for all conditions that need it.
    needs_retrieval = any(c not in {"closed_book", "oracle"} for c in args.condition)
    retrieved: Dict[str, List[str]] = {}
    if needs_retrieval:
        if not args.index.exists():
            print(f"Building {args.index}")
            build_index(args.corpus, args.index)
        t0 = time.time()
        retriever = DenseRetriever(args.corpus, args.index)
        todo = {q["qid"]: q["question"] for c in args.condition
                for q in select_questions(all_questions, c, args.limit, args.seed)}
        qids = list(todo)
        hits = retriever.search([todo[i] for i in qids], k=6 * args.k + 60)
        retrieved = dict(zip(qids, hits))
        print(f"retrieval: {len(qids)} queries in {time.time()-t0:.1f}s")

    model = get_model(args.model,
                      max_new_tokens=args.max_new_tokens,
                      load_4bit=args.load_4bit,
                      dtype=args.dtype)

    for condition in args.condition:
        questions = select_questions(all_questions, condition, args.limit, args.seed)
        print(f"\n{condition}: {len(questions)} questions")
        t0 = time.time()
        records = run_condition(condition, questions, by_id, retrieved, model, args)
        out_path = args.out_dir / f"{condition}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\nwrote {len(records)} records -> {out_path} "
              f"({time.time()-t0:.0f}s, model={getattr(model, 'name', args.model)})")


if __name__ == "__main__":
    main()