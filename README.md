# CS598 Project: Cognitive Insight for Retrieval-Augmented Generation

**Generate–Recognize–Report (GRR)**: a training-free generation-side method for
RAG that separates *recall* (proposing an answer) from *recognition* (checking
that the documents actually state it) and *report control* (deciding whether to
volunteer it), after Anderson & Bower (1972) and Koriat & Goldsmith (1996).

Model: `Qwen/Qwen3-4B-Instruct-2507`, greedy decoding. Retriever:
`BAAI/bge-small-en-v1.5`, `k=5`. Both are fixed by the course harness.

For environment setup, hardware options and troubleshooting of the underlying
course harness, see [`README_implementation.md`](README_implementation.md).

## Results (full run, no `--limit`)

| | Use F1 | Ignore F1 | Combined F1 |
|---|---|---|---|
| Baseline (`report_baseline.json`) | 0.3695 | 0.7910 | 0.5037 |
| GRR (`report.json`) | 0.3716 | **0.8490** | 0.5170 |
| Δ, 95% paired-bootstrap CI | +0.002 [−0.029, +0.033] | **+0.058 [+0.034, +0.082]** | +0.013 [−0.017, +0.043] |

* **Phase 1 (single-metric improvement): met.** The Ignore F1 gain is significant.
* **Phase 2 (Combined F1): not yet met.** The gain is positive but not significant (p ≈ 0.20).

Ablations, as exact counterfactuals from the same run. Both gates are applied in
code after all model passes, and the `raw` field of every record in
`runs/*.jsonl` logs the recall candidate, the monitor's verdict and the closure
score. Each row below therefore re-applies a different gate to the same outputs,
with no new model calls:

| Configuration | Use | Ignore | Combined |
|---|---|---|---|
| Recall + re-search, no gates | **0.529** | 0.458 | 0.491 |
| … without re-search | 0.517 | 0.485 | 0.501 |
| Closure gate only | 0.509 | 0.512 | 0.511 |
| Support gate only | 0.385 | 0.831 | **0.526** |
| Support + closure (GRR, submitted) | 0.372 | 0.849 | 0.517 |

All prompts and thresholds were developed on the same 1,000 questions reported
here; there is no separate held-out split. The "support gate only" row was
identified only after we saw these results, so treat it as an ablation, not as
the headline result.

## Method in one paragraph

`model/llm.py` wraps the model in `GenerateRecognizeReport`, which makes up to
three passes per question. (1) **Recall** classifies the question as extraction
or comparison, quotes evidence, and proposes an answer or `null`. (2) **Re-search**
runs only when recall returned `null`, and looks again for a bridge entity whose
answer is split across two documents. (3) **Monitor** runs a separate pass on
every candidate. It resolves the target entity and the property asked about,
quotes the sentence that states that property, and returns `supported`.
**Control** is plain code: withhold the answer if `supported` is false, or if a
lexical spreading-activation "closure" score (`model/cognitive.py`) is below
0.05. `run.py`, `evaluate.py`, the retriever and `k` are unchanged.

## Repository layout

| Path | What it is |
|---|---|
| `model/llm.py` | **The method (GRR).** The only harness file we changed. |
| `model/cognitive.py` | Spreading activation (closure score) and an unused evidence-accumulation rule. |
| `model/llm_baseline.py.bak` | The original baseline `llm.py`, kept for reference. |
| `model/rag.py`, `run.py`, `evaluate.py`, `dataset/*.py` | Course harness, **unmodified**. |
| `compare.py` | Development aid: scores a run directory against the baseline on the same questions. |
| `patch_shapes.py` | A four-way question-shape prompt, prepared for Phase 2. **Not applied** to the submitted `llm.py`, and not part of the reported results. |
| `README_implementation.md` | The course's original setup guide: installation, low-VRAM options, troubleshooting. |
| `report.json`, `per_example.json` | Scores of the final GRR run (output of `evaluate.py`). |
| `report_baseline.json`, `per_example_baseline.json` | Scores of the unmodified baseline. |
| `runs/{gold_plus_noise,no_gold}.jsonl` | Raw model outputs of the final GRR run. The `raw` field logs the monitor features used by the ablations. |
| `runs_dev/`, `runs_probe/`, `runs_open/`, `report_dev.json` | Development runs on 100–120-question subsets, made with **earlier versions** of `llm.py`. Kept as a record only; they are not the final results. |

## Files that are NOT in this repository

These files are large and are produced deterministically by the course scripts,
so we did not commit them. Regenerate them with the commands below.

| File | Size | Produced by | Contents |
|---|---|---|---|
| `raw/2wiki_dev.json` | ~56 MB | `python dataset/download_data.py` | 2WikiMultihopQA dev set (12,576 questions), from the Hugging Face dataset `xanhho/2WikiMultihopQA` (`dev.parquet`, converted to JSON). |
| `raw/musique_full_v1.0_dev.jsonl` | ~59 MB | `python dataset/download_data.py` | MuSiQue-Full dev set (4,834 questions, including unanswerable ones), from the Hugging Face dataset `bdsaglam/musique`. |
| `data/questions.jsonl` | ~3.6 MB | `dataset/prepare_data.py` (command below) | The 1,000 evaluation questions with gold paragraph ids, answers, task types and subset labels. |
| `data/corpus.jsonl` | ~6.9 MB | `dataset/prepare_data.py` | The shared retrieval pool: 11,342 paragraphs, 1,615 of them injected synthetic distractors. |
| `data/index.npy`, `data/index.meta.json` | ~17 MB | built automatically by the first `python run.py` | Dense embeddings of the corpus (`BAAI/bge-small-en-v1.5`). |
| Qwen3-4B-Instruct-2507 weights | ~8 GB | downloaded automatically by `run.py` | Hugging Face model cache. |
| `class_project_3_RAG.pdf` | — | course handout | Assignment description (not redistributed here). |

## The two evaluation subsets: `gold_plus_noise` and `no_gold`

`dataset/prepare_data.py --limit 500` draws a stratified sample of 500
questions from each dataset: 500 from 2Wiki (125 each of comparison,
bridge_comparison, compositional and inference) and 500 from MuSiQue (250
answerable and 250 natively unanswerable, spread over 2/3/4 hops). The
paragraphs of all 1,000 questions are pooled into **one shared corpus**, so
each question's paragraphs become noise for the others. For each question, up
to 2 **hard synthetic distractors** are also injected: near-copies of a gold
paragraph with the subject entity, and where possible the answer, swapped for
another entity. Each question is then placed in one or both subsets
(`assign_subsets`), and `model/rag.py::build_context` builds the context the
model sees:

### `gold_plus_noise` → **Use F1** (n = 750)

* **Which questions:** every *answerable* question that has gold (supporting)
  paragraphs. That is all 500 2Wiki questions plus the 250 answerable MuSiQue
  questions. The natively unanswerable MuSiQue questions are excluded.
* **What the model sees:** *all* gold paragraphs + `k=5` distractors, shuffled.
  Distractors are taken first from the question's injected hard distractors,
  then from the top retrieved non-gold paragraphs. That gives 7–9 paragraphs in
  total, about 7.7 on average.
* **What is tested:** *failure to incorporate*. The evidence is guaranteed to
  be present, so the model should find it and answer.
* **Score:** token-level F1 against the gold answer and its aliases.
  Abstaining scores 0.

### `no_gold` → **Ignore F1** (n = 1000)

* **Which questions:** all 1,000 questions, answerable or not.
* **What the model sees:** the context is built so that **no paragraph contains
  the answer**:
  * all answer-bearing gold paragraphs are removed;
  * every "leaky" paragraph is removed, i.e. any non-gold paragraph in the
    corpus that contains the answer string verbatim;
  * for 574 questions, the *partial gold* paragraphs are kept. These are gold
    paragraphs that set up the question but do not contain the answer, for
    example the first hop of a two-hop chain. They make the context look
    on-topic.

  The context is then topped up to `k=5` paragraphs, first with hard
  distractors and then with retrieved non-gold paragraphs, and shuffled.
* **What is tested:** *failure to ignore*. The context is relevant-looking
  but insufficient, so the model should abstain rather than build an answer
  from it or answer from memory.
* **Score:** 1 if the model abstains (`null`, or a refusal phrase recognised by
  `evaluate.py::is_abstention`), otherwise 0. Ignore F1 is therefore the
  abstention rate on this subset.

**Combined F1** = 2 · Use · Ignore / (Use + Ignore), the primary Phase 2 metric.

## Reproducing

```bash
pip install -r requirements.txt

# 1. download the raw datasets (~115 MB)
python dataset/download_data.py

# 2. build the task — run exactly once, with exactly these flags
python dataset/prepare_data.py \
  --dataset 2wiki    raw/2wiki_dev.json \
  --dataset musique  raw/musique_full_v1.0_dev.jsonl \
  --limit 500 --out data

# 3. re-run GRR end to end on a GPU (~16 GB VRAM) and score it
python run.py
python evaluate.py --runs "runs/*.jsonl" --by-task

# baseline: swap the original llm.py back in, run, then restore GRR
cp model/llm.py model/llm_grr.py
cp model/llm_baseline.py.bak model/llm.py
python run.py --out-dir runs_baseline
python evaluate.py --runs "runs_baseline/*.jsonl" --by-task
cp model/llm_grr.py model/llm.py
```

For low-VRAM flags (`--load-4bit`, `--dtype`, `--batch-size`) and common errors,
see `README_implementation.md`.

Note: `GRR_DISABLE=1` only skips the re-search and monitor passes. It still
uses GRR's recall prompt, so it is **not** the baseline.

The course's setup guide (`README_implementation.md`) quotes a reference baseline of 0.357 / 0.808 / 0.495 (Use /
Ignore / Combined). Our baseline run gave 0.370 / 0.791 / 0.504, which is
within the variation that guide expects across GPUs and library
versions.

GRR exposes its knobs as environment variables (see the top of `model/llm.py`),
e.g. `GRR_CLOSURE_FLOOR`, `GRR_CRITERION`, `GRR_NO_RESEARCH=1`, `GRR_GATE=0`.
The submitted results use the defaults. GRR makes up to 3 model calls per
question (the baseline makes 1), so a full run takes correspondingly longer.

## References

* Anderson, J. R., & Bower, G. H. (1972). Recognition and retrieval processes in free recall. *Psychological Review, 79*(2), 97–123.
* Koriat, A., & Goldsmith, M. (1996). Monitoring and control processes in the strategic regulation of memory accuracy. *Psychological Review, 103*(3), 490–517.
* Collins, A. M., & Loftus, E. F. (1975). A spreading-activation theory of semantic processing. *Psychological Review, 82*(6), 407–428.
* Ho, X., et al. (2020). Constructing a multi-hop QA dataset for comprehensive evaluation of reasoning steps. *COLING*.
* Trivedi, H., et al. (2022). MuSiQue: Multihop questions via single-hop question composition. *TACL, 10*, 539–554.
