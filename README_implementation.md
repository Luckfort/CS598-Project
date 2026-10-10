# Setup and Run Instructions

Everything you need to get the baseline running and to measure your own method.
The assignment itself (i.e., what the two failure modes are, what Use / Ignore /
Combined F1 mean, and what Phase 1 and Phase 2 ask for) is in the project
description.

---

## 0. What you need

- A GPU with ~16 GB of VRAM, which is what a default run peaks at. The
  model weights are ~8 GB; the rest is the batch. You should be able to 
  run this on ILab. With less VRAM see
  [Low VRAM](#low-vram) below — it fits in ~5 GB with the flags there. It will
  run on CPU, but a full run would take many hours instead of ~25 minutes.
- Python 3.10 or newer.
- About 2 GB of disk for the datasets, plus ~8 GB for the model, which
  downloads itself the first time you run it.

## 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Download the two datasets

```bash
python dataset/download_data.py
```

About 115 MB, a few minutes. It writes:

```
raw/2wiki_dev.json                 12,576 questions
raw/musique_full_v1.0_dev.jsonl     4,834 questions
```

Safe to re-run — files already downloaded are skipped.

## 3. Build the task

```bash
python dataset/prepare_data.py \
  --dataset 2wiki    raw/2wiki_dev.json \
  --dataset musique  raw/musique_full_v1.0_dev.jsonl \
  --limit 500 --out data
```

Takes about a minute and prints a summary ending in a task-type table. You
should see ~11,000 corpus paragraphs and 1,000 questions.

**Run this exactly once and do not change the flags.** `--limit 500` fixes the
size of the corpus everyone is scored against. Rebuilding with different flags
gives you a different task, and your numbers stop being comparable to the
baseline or to anyone else's.

## 4. Run the baseline and score it

```bash
python run.py
python evaluate.py --runs "runs/*.jsonl" --by-task
```

The first command downloads the model (once, ~8 GB), builds the retrieval index
(once), and answers both evaluation subsets. This could take a while, up to 
an hour on a decent GPU. It writes
`runs/gold_plus_noise.jsonl` and `runs/no_gold.jsonl`.

The evaluate.py command scores them and prints the baseline:

```
========================================================
  Use F1       (gold_plus_noise, n=750   ) 0.357
  Ignore F1    (no_gold,         n=1000  ) 0.808
  COMBINED F1                              0.495
========================================================
```

**If your first run does not print roughly those three numbers, stop and fix
your setup before writing any code** You have a different task than everyone
else, and nothing you measure afterwards will be comparable. Small differences
are normal (a different GPU, a different transformers version); differences in
the first decimal place are not.

It also writes `report.json` and
`per_example.json`, which has one row per question with the prediction, the gold
answer, and whether it was scored as a refusal.

You should probably save a copy of the baseline before you change anything, so that you can compare them:

```bash
cp report.json report_baseline.json
```

## 5. Do the assignment

**Edit `llm.py`.** That is the file the assignment is about. You may change
anything in it: the prompt, how the documents are formatted, how the output is
parsed, how many model calls you make per question. You may also import new
modules of your own from it if you choose to.

Then repeat step 4 and compare `report.json` to `report_baseline.json`.

**Do not touch these:**

| file / setting | why |
|---|---|
| `evaluate.py` | it is the scorer; changing it changes your evaluation, not your method |
| `prepare_data.py` | it defines the data evaluated and the two subsets everyone shares |
| the retriever and `--k` | fixed so that every submission is compared on identical retrieved chunks |

`rag.py` and `run.py` you can read — it is worth reading `build_context` in
`rag.py` to see exactly what your model is handed — but do not change them.

## 6. Hand in

1. Your `llm.py` (plus any module you added).
2. `report.json` from your final run, and the baseline one for comparison.
3. Your write-up, with Use F1, Ignore F1 and Combined F1 for both phases.

---

## Useful during development

You can test your method more quickly by testing it on small amounts of data. `--limit` takes the same
seeded sample every time, so two `--limit 200` runs are the same:

```bash
python run.py --limit 200
python evaluate.py --runs "runs/*.jsonl" --by-task
```

**Report your final numbers from a full run with no `--limit`.**

The optional diagnostic conditions, if you want to know *why* a number moved:

```bash
python run.py --condition closed_book oracle retrieved
```

`closed_book` gives the model no documents at all, so anything it still gets
right, it already knew. `oracle` gives it only the gold paragraphs, which is
your ceiling. Neither is graded.

<a name="low-vram"></a>
## Low VRAM

| situation | flag |
|---|---|
| less than ~10 GB | `python run.py --load-4bit` |
| GTX 10xx/16xx, RTX 20xx, Quadro RTX | `python run.py --dtype float16` (no bf16 on these) |
| out of memory anyway | add `--batch-size 1` |

## When something goes wrong

**`CUDA out of memory`** — see the table above. `--load-4bit --batch-size 1`
fits in about 5 GB.

**`index/corpus mismatch — rebuild`** — you rebuilt `data/` after building the
index. Delete `data/index.npy` and rerun `python run.py`, which rebuilds it.

**`no run files matched`** — `run.py` did not finish, so `runs/` is empty. Rerun
step 4 and read the error it prints.

**`warning: the graded runs do not share one configuration`** — your two run
files came from different models or different `--k`. Delete `runs/` and rerun
step 4 in one go.

**It is very slow** — check it is actually using the GPU (`nvidia-smi` during a
run). Raise `--batch-size` if you have extra space.