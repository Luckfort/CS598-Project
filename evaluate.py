"""
evaluate.py -- scoring/evaluation for the assignment. DO NOT CHANGE ANYTHING HERE!

    python evaluate.py --runs "runs/*.jsonl" --by-task

Three numbers, from two runs:

  Use F1       token-level answer F1 on `gold_plus_noise`, where the evidence
               needed to answer is in the context alongside distractors.
               Low Use F1 = failure to incorporate.

  Ignore F1    the same score on `no_gold`, where no paragraph in the context
               supports any answer. The reference there is "no answer", so a
               refusal scores 1 and any answer scores 0.
               Low Ignore F1 = failure to ignore.

  Combined F1  Combination of both individual metrics: 2 * Use * Ignore / (Use + Ignore).

Writes report.json (the numbers) and per_example.json (one row per example, for
error analysis). **Do not modify this file**: every submission is scored with it.
"""
import argparse
import glob
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

def normalize(s: str | None) -> str:
    if s is None:
        return ""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str | None, golds: List[str]) -> float:
    p = normalize(pred)
    return float(any(p == normalize(g) for g in golds if g))


def f1(pred: str | None, golds: List[str]) -> float:
    """Token overlap against the best-matching gold answer."""
    best = 0.0
    pt = normalize(pred).split()
    for g in golds:
        gt = normalize(g).split()
        if not pt or not gt:
            best = max(best, float(pt == gt))
            continue
        common = Counter(pt) & Counter(gt)
        n = sum(common.values())
        if n == 0:
            continue
        prec, rec = n / len(pt), n / len(gt)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best

_ABSTAIN_EXACT = {"", "null", "none", "nil", "unknown", "unanswerable",
                  "na", "n a", "no answer", "idk"}

_ABSTAIN_PHRASES = (
    "dont know", "do not know", "not know",
    "not enough information", "not enough evidence", "not enough context",
    "insufficient information", "insufficient evidence", "insufficient context",
    "no information", "no evidence", "no relevant", "no supporting",
    "cannot be determined", "can not be determined", "cannot determine",
    "cannot be answered", "cannot answer", "can not answer", "unable to answer",
    "not answerable", "unanswerable", "not possible to answer",
    "not provided", "not mentioned", "not specified", "not stated",
    "not contain", "not include", "not support", "not say", "not indicate",
    "no mention",
)


def is_abstention(pred: str | None) -> bool:
    if pred is None:
        return True
    p = normalize(pred)
    if p in _ABSTAIN_EXACT:
        return True
    return any(ph in p for ph in _ABSTAIN_PHRASES)

def load_jsonl(path: Path) -> List[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def gold_answers(q: dict) -> List[str]:
    return [q["answer"]] + list(q.get("answer_aliases") or [])


def score(rec: dict, q: dict) -> Dict[str, Any]:
    """One example -> one row. `expect_abstain` marks the examples where no
    answer is supported (all of no_gold); everything else is token F1."""
    abstained = is_abstention(rec["prediction"])
    expect_abstain = bool(rec.get("expect_abstain"))
    if expect_abstain:
        s = float(abstained)
    else:
        s = f1(rec["prediction"], gold_answers(q))

    return {
        "qid": rec["qid"],
        "condition": rec["condition"],
        "question": rec["question"],
        "prediction": rec["prediction"],
        "gold": q["answer"],
        "expect_abstain": expect_abstain,
        "abstained": abstained,
        "f1": round(s, 4),
        "em": None if expect_abstain else exact_match(rec["prediction"],
                                                      gold_answers(q)),
        "parse_ok": rec["parse_ok"],
        "dataset": q["dataset"],
        "task_type": q["task_type"],
    }


def _mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def summarize(rows: List[dict]) -> Dict[str, Any]:
    if not rows:
        return {}
    return {
        "n": len(rows),
        "f1": _mean([r["f1"] for r in rows]),
        "em": _mean([r["em"] for r in rows]),
        "abstention_rate": _mean([float(r["abstained"]) for r in rows]),
        "json_parse_rate": _mean([float(r["parse_ok"]) for r in rows]),
    }


def combined(use: float | None, ignore: float | None) -> float | None:
    if use is None or ignore is None:
        return None
    if use + ignore <= 0:
        return 0.0
    return round(2 * use * ignore / (use + ignore), 4)

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["runs/*.jsonl"])
    ap.add_argument("--questions", type=Path, default=Path("data/questions.jsonl"))
    ap.add_argument("--report", type=Path, default=Path("report.json"))
    ap.add_argument("--by-task", action="store_true")
    args = ap.parse_args()

    questions = {q["qid"]: q for q in load_jsonl(args.questions)}

    paths = sorted({p for pat in args.runs for p in glob.glob(pat)})
    by_condition: Dict[str, List[dict]] = defaultdict(list)
    for p in paths:
        for rec in load_jsonl(Path(p)):
            by_condition[rec["condition"]].append(rec)
    if not by_condition:
        raise SystemExit(f"no run files matched {args.runs}")

    per_example: Dict[str, List[dict]] = {}
    report: Dict[str, Any] = {"conditions": {}}
    for cond, recs in by_condition.items():
        rows = [score(r, questions[r["qid"]]) for r in recs if r["qid"] in questions]
        if not rows:
            print(f"warning: no record in '{cond}' matches {args.questions} "
                  f"-- skipping that run")
            continue
        # a run over the wrong question set would silently change the metric
        off = sum(1 for r in recs if r["qid"] in questions
                  and cond in ("gold_plus_noise", "no_gold")
                  and cond not in questions[r["qid"]].get("subsets", [cond]))
        if off:
            print(f"warning: {off} record(s) in '{cond}' are not in that subset "
                  f"(rerun run.py, which filters on questions.jsonl:subsets)")
        per_example[cond] = rows
        report["conditions"][cond] = summarize(rows)
        report["conditions"][cond]["run_config"] = sorted(
            {f"{r.get('model')} k={r.get('k')}" for r in recs})

    # the comparison is only meaningful if both runs used the same model and k
    cfgs = {c for cond in ("gold_plus_noise", "no_gold")
            for c in report["conditions"].get(cond, {}).get("run_config", [])}
    if len(cfgs) > 1:
        print(f"warning: the graded runs do not share one configuration: "
              f"{', '.join(sorted(cfgs))}")

    use = report["conditions"].get("gold_plus_noise", {}).get("f1")
    ignore = report["conditions"].get("no_gold", {}).get("f1")
    report["headline"] = {
        "use_f1": use,
        "ignore_f1": ignore,
        "combined_f1": combined(use, ignore),
        "n_use": report["conditions"].get("gold_plus_noise", {}).get("n"),
        "n_ignore": report["conditions"].get("no_gold", {}).get("n"),
    }

    if args.by_task:
        groups: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
        for cond in ("gold_plus_noise", "no_gold"):
            for r in per_example.get(cond, []):
                groups[f"{r['dataset']}/{r['task_type']}"][cond].append(r)
        report["by_task"] = {}
        for key in sorted(groups):
            u = _mean([r["f1"] for r in groups[key]["gold_plus_noise"]])
            i = _mean([r["f1"] for r in groups[key]["no_gold"]])
            report["by_task"][key] = {
                "use_f1": u, "ignore_f1": i, "combined_f1": combined(u, i),
                "n_use": len(groups[key]["gold_plus_noise"]),
                "n_ignore": len(groups[key]["no_gold"]),
            }

    args.report.write_text(json.dumps(report, indent=2))
    Path("per_example.json").write_text(json.dumps(per_example, indent=2))

    cols = ["n", "f1", "em", "abstention_rate", "json_parse_rate"]
    print(f"\n{'condition':<18}" + "".join(f"{c:>18}" for c in cols))
    print("-" * (18 + 18 * len(cols)))
    for cond in sorted(report["conditions"], key=lambda c: c not in ("gold_plus_noise", "no_gold")):
        s = report["conditions"][cond]
        cells = "".join(
            f"{('-' if s.get(c) is None else (f'{s[c]}' if c == 'n' else f'{s[c]:.3f}')):>18}"
            for c in cols)
        print(f"{cond:<18}{cells}")

    if args.by_task:
        print(f"\n{'task':<28}{'use_f1':>10}{'ignore_f1':>12}{'combined_f1':>14}")
        print("-" * 64)
        for key, v in report["by_task"].items():
            fmt = lambda x: "-" if x is None else f"{x:.3f}"
            print(f"{key:<28}{fmt(v['use_f1']):>10}{fmt(v['ignore_f1']):>12}"
                  f"{fmt(v['combined_f1']):>14}")

    h = report["headline"]
    if h["combined_f1"] is not None:
        print(f"\n{'=' * 56}")
        print(f"  Use F1       (gold_plus_noise, n={h['n_use']:<6}) {h['use_f1']:.3f}")
        print(f"  Ignore F1    (no_gold,         n={h['n_ignore']:<6}) {h['ignore_f1']:.3f}")
        print(f"  COMBINED F1                            {h['combined_f1']:.3f}")
        print(f"{'=' * 56}")
    else:
        print("\nboth runs/gold_plus_noise.jsonl and runs/no_gold.jsonl are "
              "needed for the headline numbers")

    print(f"\nwrote {args.report} and per_example.json")


if __name__ == "__main__":
    main()
