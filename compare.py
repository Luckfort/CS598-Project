"""compare.py — score a run directory against the saved baseline on the SAME
questions. Not part of the submission; a development aid.

    python compare.py runs_probe
"""
import json, re, string, sys
from collections import Counter, defaultdict
from pathlib import Path

def norm(s):
    s = (s or "").lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())

def f1(p, golds):
    best = 0.0
    for g in golds:
        pt, gt = norm(p).split(), norm(g).split()
        if not pt or not gt:
            best = max(best, float(pt == gt)); continue
        n = sum((Counter(pt) & Counter(gt)).values())
        if n:
            pr, rc = n / len(pt), n / len(gt)
            best = max(best, 2 * pr * rc / (pr + rc))
    return best

d = Path(sys.argv[1] if len(sys.argv) > 1 else "runs_probe")
qs = {q["qid"]: q for q in (json.loads(l) for l in open("data/questions.jsonl"))}
base = json.load(open("per_example_baseline.json"))
mine, theirs, tasks = {}, {}, defaultdict(lambda: [0.0, 0.0, 0])
for cond in ("gold_plus_noise", "no_gold"):
    recs = [json.loads(l) for l in (d / f"{cond}.jsonl").open()]
    ids = {r["qid"] for r in recs}
    b = {r["qid"]: r["f1"] for r in base[cond] if r["qid"] in ids}
    src = Counter(); tot = 0.0
    for r in recs:
        q = qs[r["qid"]]
        s = (float(r["prediction"] is None) if cond == "no_gold"
             else f1(r["prediction"], [q["answer"]] + list(q.get("answer_aliases") or [])))
        tot += s
        o = json.loads(r["raw"])
        src[o.get("_src") or (o.get("_why", "?").split("(")[0].strip())] += 1
        k = f"{q['dataset']}/{q['task_type']}"
        tasks[k][0 if cond == "gold_plus_noise" else 1] += s
        if cond == "gold_plus_noise":
            tasks[k][2] += 1
    mine[cond] = tot / len(recs)
    theirs[cond] = sum(b.values()) / len(b)
    print(f"[{cond:16s}] mine {mine[cond]:.3f}   baseline {theirs[cond]:.3f}   n={len(recs)}")
    print(f"                   {dict(src.most_common())}")

h = lambda u, i: 2 * u * i / (u + i) if u + i else 0.0
mu, mi = mine["gold_plus_noise"], mine["no_gold"]
bu, bi = theirs["gold_plus_noise"], theirs["no_gold"]
print(f"\n{'':18}{'Use':>8}{'Ignore':>9}{'Combined':>10}")
print(f"{'mine':<18}{mu:>8.3f}{mi:>9.3f}{h(mu,mi):>10.3f}")
print(f"{'baseline':<18}{bu:>8.3f}{bi:>9.3f}{h(bu,bi):>10.3f}")
print(f"{'delta':<18}{mu-bu:>+8.3f}{mi-bi:>+9.3f}{h(mu,mi)-h(bu,bi):>+10.3f}")
