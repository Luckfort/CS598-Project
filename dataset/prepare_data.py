"""
prepare_data.py — build the corpus and question files for the RAG assignment.
Run as-is; you do NOT need to modify it.
"""
import argparse
import hashlib
import json
import random
import re
import string
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List

_punct = set(string.punctuation)
def _norm(s: str | None) -> str:
    return unicodedata.normalize("NFKC", s or "").strip()

def _norm_text(s: str) -> str:
    """Lowercase, punctuation-free form used for answer-leak detection."""
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = "".join(ch if ch not in _punct else " " for ch in s)
    return " ".join(s.split())

def _doc_id(dataset: str, title: str, text: str) -> str:
    h = hashlib.sha1((_norm(title) + "|" + _norm(text)).encode("utf-8")).hexdigest()[:12]
    return f"{dataset}:{h}"

def load_2wiki(path: Path) -> Iterator[Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    for ex in raw:
        sup: Dict[str, set] = {}
        for title, sent_idx in ex.get("supporting_facts", []):
            sup.setdefault(_norm(title), set()).add(sent_idx)

        paragraphs = []
        for title, sentences in ex["context"]:
            key = _norm(title)
            idxs = sorted(i for i in sup.get(key, set()) if 0 <= i < len(sentences))
            paragraphs.append({
                "title": title,
                "text": " ".join(s.strip() for s in sentences),
                "is_supporting": key in sup,
                "supporting_sentences": [sentences[i].strip() for i in idxs],
            })

        yield {
            "qid": ex["_id"],
            "question": ex["question"],
            "answer": ex["answer"],
            "answer_aliases": [],
            "task_type": ex.get("type", "unknown"),
            "answerable": True,
            "evidence_triples": ex.get("evidences", []),
            "paragraphs": paragraphs,
        }

def load_musique(path: Path) -> Iterator[Dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ex = json.loads(line)
        paragraphs = [{
            "title": p.get("title", ""),
            "text": p.get("paragraph_text", ""),
            "is_supporting": bool(p.get("is_supporting", False)),
            "supporting_sentences": [],
        } for p in ex.get("paragraphs", [])]

        m = re.match(r"(\d+hop)", ex["id"])
        hops = m.group(1) if m else "unknown"

        answerable = bool(ex.get("answerable", True))
        yield {
            "qid": ex["id"] if answerable else f"{ex['id']}:unanswerable",
            "question": ex["question"],
            "answer": ex.get("answer") or "",
            "answer_aliases": ex.get("answer_aliases") or [],
            "task_type": hops,
            "answerable": answerable,
            "paragraphs": paragraphs,
        }


loaders = {
    "2wiki": load_2wiki,
    "musique": load_musique,
}

def _stratified_sample(items: List[Dict[str, Any]], limit: int,
                       rng: random.Random) -> List[Dict[str, Any]]:
    """Sample evenly across (task_type, answerable) buckets."""
    buckets: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for ex in items:
        buckets[(ex["task_type"], ex["answerable"])].append(ex)
    for b in buckets.values():
        rng.shuffle(b)

    out: List[Dict[str, Any]] = []
    keys = sorted(buckets, key=lambda k: (str(k[0]), k[1]))
    while len(out) < limit and any(buckets[k] for k in keys):
        for k in keys:
            if buckets[k] and len(out) < limit:
                out.append(buckets[k].pop())
    rng.shuffle(out)
    return out

def pool(datasets: List[tuple[str, Path]], limit: int | None, rng: random.Random):
    corpus: Dict[str, Dict[str, Any]] = {}
    questions: List[Dict[str, Any]] = []
    seen_qids: set[str] = set()
    collisions = 0

    for name, path in datasets:
        items = list(loaders[name](path))
        if limit:
            items = _stratified_sample(items, limit, rng)

        for ex in items:
            qid = f"{name}:{ex['qid']}"
            if qid in seen_qids:
                collisions += 1
                qid = f"{qid}#{collisions}"
            seen_qids.add(qid)

            gold, local, gold_sents = [], [], []
            for p in ex["paragraphs"]:
                did = _doc_id(name, p["title"], p["text"])
                if did not in corpus:
                    corpus[did] = {
                        "doc_id": did,
                        "title": _norm(p["title"]),
                        "text": _norm(p["text"]),
                        "synthetic": False,
                        "derived_from": None,
                    }
                local.append(did)
                if p["is_supporting"]:
                    gold.append(did)
                    gold_sents.extend(p.get("supporting_sentences", []))

            questions.append({
                "qid": qid,
                "dataset": name,
                "question": _norm(ex["question"]),
                "answer": _norm(ex["answer"]),
                "answer_aliases": [_norm(a) for a in ex.get("answer_aliases", []) if _norm(a)],
                "gold_doc_ids": sorted(set(gold)),
                "gold_sentences": gold_sents,
                "evidence_triples": ex.get("evidence_triples", []),
                "task_type": ex["task_type"],
                "answerable": ex["answerable"],
                "local_doc_ids": local,
                "hard_distractor_ids": [],
                "leaky_doc_ids": [],
                "partial_gold_ids": [],
                "subsets": [],
            })

    if collisions:
        print(f"warning: {collisions} duplicate question id(s) were disambiguated")
    return corpus, questions


_year_re = re.compile(r"^\d{3,4}$")
def _title_variants(title: str) -> List[str]:
    t = _norm(title)
    out = [t]
    bare = re.sub(r"\s*\([^)]*\)\s*$", "", t).strip()
    if bare and bare != t:
        out.append(bare)
    toks = bare.split()
    if len(toks) > 1 and len(toks[-1]) > 3 and toks[-1][0].isupper():
        out.append(toks[-1])
    return sorted(set(out), key=len, reverse=True)

def _title_bucket(title: str) -> tuple:
    t = _norm(title)
    return (min(len(t.split()), 4),
            bool(re.search(r"\d", t)),
            t.endswith(")"))

def _answer_bucket(a: str) -> str:
    a = a.strip()
    low = a.lower()
    if low in {"yes", "no"}:
        return "yesno"
    if _year_re.match(a):
        return "year"
    if re.fullmatch(r"[\d\s,.]+", a):
        return "number"
    if re.search(r"\d", a):
        return "mixed"
    return f"entity{min(len(a.split()), 4)}"

def _sub_entity(text: str, old: str, new: str) -> tuple[str, int]:
    if not old:
        return text, 0
    pat = re.compile(r"(?<!\w)" + re.escape(old) + r"(?!\w)", re.IGNORECASE)
    return pat.subn(new, text)

def inject_hard_distractors(corpus: Dict[str, Dict[str, Any]],
                            questions: List[Dict[str, Any]],
                            n_per_question: int, rng: random.Random) -> int:
    """Add near-copies of gold paragraphs with subject and answer swapped."""
    if n_per_question <= 0:
        return 0

    titles: Dict[str, Dict[tuple, List[str]]] = defaultdict(lambda: defaultdict(list))
    answers: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    by_token: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for q in questions:
        ds = q["dataset"]
        for did in q["local_doc_ids"]:
            t = corpus[did]["title"]
            if t:
                titles[ds][_title_bucket(t)].append(t)
                for tok in set(_norm_text(t).split()):
                    if len(tok) > 2:
                        by_token[ds][tok].add(t)
        a = q["answer"]
        if a and a.lower() not in {"yes", "no"} and len(a.split()) <= 6:
            answers[ds][_answer_bucket(a)].append(a)

    def _confusable(ds: str, title: str, forbidden: set):
        toks = [t for t in _norm_text(title).split() if len(t) > 2]
        overlap: Counter = Counter()
        for tok in toks:
            for cand in by_token[ds].get(tok, ()):
                overlap[cand] += 1
        self_norm = _norm_text(title)
        ranked = [c for c, _ in overlap.most_common(40)
                  if c.lower() not in forbidden and _norm_text(c) != self_norm]
        if not ranked:
            return None
        return rng.choice(ranked[:8])

    def _donor(pool_: Dict[Any, List[str]], key, forbidden: set, tries: int = 24):
        cands = pool_.get(key) or []
        if len(cands) < 4:
            cands = [c for v in pool_.values() for c in v]
        for _ in range(min(tries, len(cands))):
            c = rng.choice(cands)
            if c.lower() not in forbidden:
                return c
        return None

    n_new = 0
    for q in questions:
        gold_ids = [d for d in q["gold_doc_ids"] if d in corpus]
        if not gold_ids:
            continue
        ds = q["dataset"]
        ans = q["answer"]
        swappable = bool(ans) and ans.lower() not in {"yes", "no"}
        ans_norm = _norm_text(ans) if swappable else ""
        qnorm = _norm_text(q["question"])
        forbidden = {corpus[d]["title"].lower() for d in q["local_doc_ids"]}
        forbidden |= {ans.lower()}

        made = 0
        for did in gold_ids:
            if made >= n_per_question:
                break
            doc = corpus[did]
            donor_title = (_confusable(ds, doc["title"], forbidden)
                           or _donor(titles[ds], _title_bucket(doc["title"]), forbidden))
            if not donor_title or _norm_text(donor_title) in qnorm:
                continue
            donor_answer = (_donor(answers[ds], _answer_bucket(ans),
                                   forbidden | {ans.lower()})
                            if swappable else None)

            src = _title_variants(doc["title"])
            dst = _title_variants(donor_title)
            text, hits = doc["text"], 0
            for i, s in enumerate(src):
                repl = dst[min(i, len(dst) - 1)]
                text, n = _sub_entity(text, s, repl)
                hits += n
            if not hits:
                continue
            if donor_answer:
                text, _ = _sub_entity(text, ans, donor_answer)

            # drop if the true answer still appears in the distractor
            padded = f" {_norm_text(text)} "
            if ans_norm and f" {ans_norm} " in padded:
                continue
            if swappable and any(_norm_text(al) and f" {_norm_text(al)} " in padded
                                 for al in q["answer_aliases"]):
                continue

            new_id = _doc_id(ds + ":syn", donor_title, text)
            if new_id not in corpus:
                corpus[new_id] = {
                    "doc_id": new_id,
                    "title": donor_title,
                    "text": text,
                    "synthetic": True,
                    "derived_from": q["qid"],
                }
                n_new += 1
            if new_id not in q["hard_distractor_ids"]:
                q["hard_distractor_ids"].append(new_id)
                made += 1

    return n_new

def mark_answer_leaks(corpus: Dict[str, Dict[str, Any]],
                      questions: List[Dict[str, Any]]) -> None:
    """Record every non-gold paragraph that states the answer verbatim."""
    doc_ids = list(corpus)
    norm_text = [f" {_norm_text(corpus[d]['text'])} " for d in doc_ids]
    inverted: Dict[str, List[int]] = defaultdict(list)
    for i, t in enumerate(norm_text):
        for tok in set(t.split()):
            inverted[tok].append(i)

    for q in questions:
        gold = set(q["gold_doc_ids"])
        hits: set[str] = set()
        for cand in [q["answer"], *q["answer_aliases"]]:
            a = _norm_text(cand)
            if not a or a in {"yes", "no"} or len(a) < 2:
                continue
            toks = a.split()
            rarest = min(toks, key=lambda t: len(inverted.get(t, ())))
            for i in inverted.get(rarest, ()):
                if f" {a} " in norm_text[i] and doc_ids[i] not in gold:
                    hits.add(doc_ids[i])
        q["leaky_doc_ids"] = sorted(hits)

def _answer_recall(answer_norm: str, doc_tokens: set) -> float:
    toks = [t for t in answer_norm.split() if len(t) > 2] or answer_norm.split()
    if not toks:
        return 0.0
    return sum(1 for t in toks if t in doc_tokens) / len(toks)

def mark_partial_gold(corpus: Dict[str, Dict[str, Any]],
                      questions: List[Dict[str, Any]]) -> None:
    """Keep only gold paragraphs that do not contain the answer."""
    for q in questions:
        q["partial_gold_ids"] = []
        if not (q["answerable"] and q["gold_doc_ids"]):
            continue
        cands = [_norm_text(a) for a in [q["answer"], *q["answer_aliases"]]]
        cands = [a for a in cands if a and a not in {"yes", "no"} and len(a) >= 2]
        if not cands:
            continue

        keep = []
        for did in q["gold_doc_ids"]:
            doc = corpus.get(did)
            if not doc:
                continue
            norm = _norm_text(doc["text"])
            padded = f" {norm} "
            toks = set(norm.split())
            if any(f" {a} " in padded or _answer_recall(a, toks) >= 0.5
                   for a in cands):
                continue
            keep.append(did)
        if keep and len(keep) < len(q["gold_doc_ids"]):
            q["partial_gold_ids"] = keep

def assign_subsets(questions: List[Dict[str, Any]]) -> None:
    for q in questions:
        subs = []
        if q["answerable"] and q["gold_doc_ids"]:
            subs.append("gold_plus_noise")
        subs.append("no_gold")
        q["subsets"] = subs

def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def build(datasets: List[tuple[str, Path]], out_dir: Path,
          limit: int | None = None, seed: int = 0,
          distractors: int = 2) -> None:
    rng = random.Random(seed)
    corpus, questions = pool(datasets, limit, rng)
    n_syn = inject_hard_distractors(corpus, questions, distractors, rng)
    mark_answer_leaks(corpus, questions)
    mark_partial_gold(corpus, questions)
    assign_subsets(questions)

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "corpus.jsonl", corpus.values())
    _write_jsonl(out_dir / "questions.jsonl", questions)

    n_gpn = sum(1 for q in questions if "gold_plus_noise" in q["subsets"])
    n_ng = sum(1 for q in questions if "no_gold" in q["subsets"])
    n_unans = sum(1 for q in questions if not q["answerable"])
    print(f"corpus paragraphs   : {len(corpus):,}  ({n_syn:,} injected distractors)")
    print(f"questions           : {len(questions):,} ({n_unans} natively unanswerable)")
    print(f"  gold_plus_noise   : {n_gpn:,}")
    print(f"  no_gold           : {n_ng:,}")
    print(f"avg gold / question : "
          f"{sum(len(q['gold_doc_ids']) for q in questions) / max(1, len(questions)):.2f}")
    print(f"avg hard distractors: "
          f"{sum(len(q['hard_distractor_ids']) for q in questions) / max(1, len(questions)):.2f}")
    print(f"avg leaky paragraphs: "
          f"{sum(len(q['leaky_doc_ids']) for q in questions) / max(1, len(questions)):.2f}"
          f"   (filtered out of no_gold)")
    n_partial = sum(1 for q in questions if q["partial_gold_ids"])
    print(f"partial-gold no_gold: {n_partial:,} questions keep their non-answer "
          f"gold paragraphs")
    for k, v in sorted(Counter(f"{q['dataset']}/{q['task_type']}"
                               for q in questions).items()):
        print(f"  {k:<28} {v}")

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", action="append", nargs=2,
                    metavar=("NAME", "PATH"),
                    help="e.g. --dataset 2wiki raw/2wiki_dev.json (repeatable)")
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--limit", type=int, default=None,
                    help="max questions per dataset (stratified by task type)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--distractors", type=int, default=2,
                    help="hard distractors injected per question (0 disables)")
    args = ap.parse_args()

    if not args.dataset:
        ap.error("at least one --dataset NAME PATH is required")

    specs = []
    for name, path in args.dataset:
        if name not in loaders:
            ap.error(f"unknown dataset {name!r}; choose from {sorted(loaders)}")
        specs.append((name, Path(path)))

    build(specs, args.out, limit=args.limit, seed=args.seed,
          distractors=args.distractors)


if __name__ == "__main__":
    main()