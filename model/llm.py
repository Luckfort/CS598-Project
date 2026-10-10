"""
llm.py — generation side of the pipeline.

METHOD: "Generate–Recognize–Report" (GRR), a two-process account of grounded
answering, after Anderson & Bower (1972) and Koriat & Goldsmith (1996).

The baseline collapses three cognitively distinct operations into a single
forward pass: it must simultaneously *retrieve* a candidate, *assess* whether
that candidate is warranted, and *decide* whether to volunteer it. Anderson &
Bower's two-process theory of free recall holds that generation and recognition
are separable stages, and that a large class of recall failures are in fact
*recognition* failures of items the generation stage could have produced. The
baseline shows exactly that signature: on `gold_plus_noise`, where the evidence
is guaranteed present, it declines on 33% of questions, while scoring F1 0.554
on the questions it does answer. It is not failing to find answers. It is
refusing to emit answers it is capable of finding.

Koriat & Goldsmith then supply the missing half. They show that memory accuracy
is *strategically regulated*: a monitoring process assigns subjective confidence
to a candidate, and a separate control process compares that confidence to a
**report criterion** set by the payoff structure, volunteering the answer above
criterion and withholding it below. Accuracy and informativeness trade off along
that criterion, which is precisely the Use F1 / Ignore F1 tradeoff here. Their
framework also predicts **grain-size control**: when a fine-grained answer is
risky, report the coarsest grain still likely to be correct. Token-level F1
inverts that pressure (verbose answers lose precision), so the same control
process is used here to tighten answers to the minimal supporting span.

The three stages, run as separate model passes:

  1. RECALL      generate a candidate by chaining evidence across documents.
                 Refusal is *forbidden* at this stage — monitoring happens later.
  2. MONITOR     re-examine the documents and check each link of the chain
                 against them; emit a verdict and a confidence.
  3. CONTROL     compare confidence to the report criterion (in code, not in the
                 model's head), plus a source-monitoring check that the answer is
                 actually attested in the context. Report or withhold.

Separating (1) from (2) is what raises Use F1; making (2) strict and (3)
explicit is what protects Ignore F1. REPORT_CRITERION is the single knob that
moves the system along the accuracy/informativeness curve.

The model is asked for JSON with an explicit citation list, indicating what
chunks it looked at:

    {"answer": "...", "cited": [2, 5]}

Citations are what make failure modes measurable. Without them you only see
that the answer was wrong, not where the wrong answer came from.
"""
import json
import os
import re
import string
from typing import Any, Dict, List, Sequence

from model.cognitive import accumulate, decide, order_by_activation, spread

# --- control parameters ------------------------------------------------------
# The report criterion of Koriat & Goldsmith (1996): the confidence a candidate
# must reach before the control process volunteers it. Raising it buys Ignore F1
# at the cost of Use F1; lowering it does the reverse. Overridable from the
# environment so the criterion can be swept without editing the method.
# Empirically the model's graded confidence carries almost no information here
# (its median is 100 on answerable and unanswerable questions alike), while its
# binary judgement of whether the asked-for property is recorded does. The
# criterion is therefore 0 by default: the support judgement alone controls
# reporting. Raising it re-introduces a confidence threshold for ablation.
REPORT_CRITERION = int(os.environ.get("GRR_CRITERION", "0"))

# Require the reported answer to be attested in the context (source monitoring).
SOURCE_CHECK = os.environ.get("GRR_SOURCE_CHECK", "1") != "0"

# Token budgets per stage. Recall needs room for the evidence chain; the
# monitoring verdict is short.
RECALL_TOKENS = int(os.environ.get("GRR_RECALL_TOKENS", "256"))
RESEARCH_TOKENS = int(os.environ.get("GRR_RESEARCH_TOKENS", "256"))
DISABLE_RESEARCH = os.environ.get("GRR_NO_RESEARCH") == "1"
VERIFY_RECOVERIES = os.environ.get("GRR_NO_VERIFY") != "1"
# Monitor every answer, not only the ones re-search recovered.
VERIFY_ALL = os.environ.get("GRR_VERIFY_ALL", "1") == "1"
# Record the monitor's features but report every candidate regardless, so the
# decision rule can be chosen offline from one run instead of one run per rule.
GATE = os.environ.get("GRR_GATE", "1") == "1"
MONITOR_TOKENS = int(os.environ.get("GRR_MONITOR_TOKENS", "160"))

# Present the context in activation order, so bridge documents are not buried.
USE_SPREAD = os.environ.get("GRR_SPREAD", "0") == "1"
# Minimum chain closure required to report. Activation that reaches a document
# only through another document is the signature of a bridge; its absence says
# the context has no chain, independently of anything the model claims about its
# own confidence. Flat optimum: 0.02/0.05/0.10 all score within 0.002.
CLOSURE_FLOOR = float(os.environ.get("GRR_CLOSURE_FLOOR", "0.05"))
# Replace the single-threshold control rule with bounded evidence accumulation.
USE_DDM = os.environ.get("GRR_DDM", "0") == "1"
# Boundaries for the accumulator, in log-odds.
REPORT_BOUND = float(os.environ.get("GRR_REPORT_BOUND", "0.8"))
WITHHOLD_BOUND = float(os.environ.get("GRR_WITHHOLD_BOUND", "0.8"))

# Answers that are judgements rather than spans, and so cannot be quoted.
_UNQUOTABLE = {"yes", "no", "true", "false"}


# --- stage 1: RECALL ---------------------------------------------------------

RECALL_SYSTEM = (
    "You answer questions using provided evidence as reference. "
    "Reply with a single JSON object and nothing else."
)

RECALL_TEMPLATE = """Documents:
{documents}

Question: {question}

Decide which shape the question has before answering.

EXTRACTION ("where did X die", "who is Y's mother", "when was Z founded"):
the answer is a short span copied from whichever document states the property
being asked about. It is often reached in two steps — one document ties the
entity named in the question to an intermediate entity, and another states that
intermediate entity's property.

COMPARISON ("which came first", "are X and Y from the same country", "who was
born earlier"): look up the same property for BOTH entities, then compare. The
answer is the RESULT of the comparison — the winning entity's name, or exactly
"yes" or "no" — and it will not appear as a span anywhere. Never answer with one
of the looked-up values instead of the result: for "which film is older" the
answer is a film title, not a year.

The answer must match the type the question asks for: a place for "where", a
person for "who", a date for "when", a title for "which work", exactly "yes" or
"no" for a yes/no question. Give the shortest form that answers it — no
explanation, no restating the question, no clause describing when or how
something happened.

If the documents do not support an answer, use null. Do not guess from your own
knowledge, and do not treat a document that merely mentions the right people or
places as support for a property it never states.

Reply with JSON only, in this exact form:
{{"shape": "extraction" or "comparison", "evidence": "<the fact(s) you used, quoted>", "answer": "<answer, or null>", "cited": [<document numbers used>]}}"""

RECALL_NO_CONTEXT = """Question: {question}

Reply with JSON only, in this exact form:
{{"shape": "extraction", "evidence": "", "answer": "<short answer string, or null>", "cited": []}}"""


# --- stage 2: RE-SEARCH (only where the first reading found nothing) ---------

RESEARCH_SYSTEM = (
    "You are the second retrieval attempt of a question-answering system. A "
    "first reading of these documents did not find an answer. Most of the time "
    "that verdict is right and you should confirm it; but a first reading also "
    "misses answers that are split across two documents, and finding those is "
    "why you exist. Reply with a single JSON object and nothing else."
)

RESEARCH_TEMPLATE = """Documents:
{documents}

Question: {question}

A first reading concluded these documents cannot answer this question. Before
accepting that, make one focused second attempt, because the most common reason
a first reading fails is that the answer was never in any single document:

1. Name the entity the question is ultimately about. If the question refers to
   it indirectly — "the performer of song X", "Y's husband", "the director of
   Z" — then the question never names it, and you must first find the document
   that identifies it. Write down that intermediate entity.
2. Now look through every document again for that intermediate entity by name,
   not for the wording of the question. The document that answers the question
   is usually about the intermediate entity and shares almost no wording with
   the question, which is exactly why the first reading skipped it.
3. If the question is a comparison, check whether both compared values are
   present; a comparison is answerable even when no document compares them.

Answer only if the chain is complete and every link is stated in a document.
If after this second look there is still no document stating the property being
asked about, answer null — that is the correct outcome most of the time here,
and a guess is worse than null.

Reply with JSON only, in this exact form:
{{"intermediate": "<the entity the question is really about, or null>", "evidence": "<the fact(s) you used, quoted>", "answer": "<answer, or null>", "cited": [<document numbers used>]}}"""


# --- stage 2: MONITOR --------------------------------------------------------

MONITOR_SYSTEM = (
    "You check whether a set of documents actually states a particular fact. "
    "You are not judging whether an answer sounds plausible or whether the "
    "documents are on-topic — retrieved documents are always on-topic. You are "
    "answering one narrow, checkable question: is the specific property being "
    "asked about written down for the specific entity it is asked about? "
    "Reply with a single JSON object and nothing else."
)

MONITOR_TEMPLATE = """Documents:
{documents}

Question: {question}

A candidate answer of "{candidate}" was proposed, from: {chain}

Do not evaluate that candidate directly. Instead establish, from the documents
alone, whether the fact the question asks for is actually recorded:

1. "target": which entity is the question ultimately asking about? If the
   question names it indirectly ("the performer of song X", "Y's mother"),
   resolve it using the documents and write the resolved name. If the documents
   do not let you resolve it, write null.
2. "property": what is being asked about that entity — place of death, date of
   birth, spouse, director, country?
3. "stated": find the document sentence that records THAT property for THAT
   entity, and quote it. The sentence must be about the target entity and must
   state the property. A sentence about a relative, about a different work, or
   about the entity's other attributes does not count. If no sentence does
   this, write null.
4. For a comparison question, step 3 succeeds only if the property is recorded
   for BOTH compared entities.

Then: "supported" is true only if "stated" is a real quotation from the
documents. "confidence" is how sure you are that the recorded fact makes
"{candidate}" exactly right, 0 to 100.

Reply with JSON only, in this exact form:
{{"target": "<entity or null>", "property": "<property>", "stated": "<quoted sentence or null>", "supported": true or false, "answer": "<answer implied by the quoted sentence, or null>", "cited": [<document numbers>], "confidence": <0-100>}}"""


def format_documents(chunks: Sequence[dict]) -> str:
    if not chunks:
        return "(none)"
    out = []
    for i, c in enumerate(chunks, start=1):
        title = c.get("title") or ""
        head = f"[{i}] {title}".rstrip()
        out.append(f"{head}\n{c['text']}")
    return "\n\n".join(out)


class PromptBundle(list):
    """A message list that also carries the question and the chunks it was built
    from. `run.py` only ever passes this object straight through to
    `model.generate`, so it is a safe place to keep what the later stages need
    without touching the harness."""

    def __init__(self, messages: List[Dict[str, str]], question: str,
                 chunks: Sequence[dict], order: List[int] | None = None,
                 closure: float = 0.0):
        super().__init__(messages)
        self.question = question
        self.chunks = list(chunks)          # display order
        # order[i] is the position in run.py's original chunk list that display
        # slot i came from, so citations can be reported in the harness's terms.
        self.order = list(order if order is not None else range(len(chunks)))
        self.closure = closure


def build_messages(question: str, chunks: Sequence[dict]) -> List[Dict[str, str]]:
    """Stage-1 (recall) prompt. Later stages are built inside the runner."""
    order, closure = list(range(len(chunks))), 0.0
    if chunks:
        act, closure = spread(question, chunks)
        if USE_SPREAD:
            order = order_by_activation(chunks, act)
            chunks = [chunks[i] for i in order]
    if chunks:
        user = RECALL_TEMPLATE.format(documents=format_documents(chunks),
                                      question=question)
    else:
        user = RECALL_NO_CONTEXT.format(question=question)
    return PromptBundle([{"role": "system", "content": RECALL_SYSTEM},
                         {"role": "user", "content": user}], question, chunks,
                        order, closure)


def _build_research_messages(bundle: PromptBundle) -> List[Dict[str, str]]:
    user = RESEARCH_TEMPLATE.format(documents=format_documents(bundle.chunks),
                                    question=bundle.question)
    return [{"role": "system", "content": RESEARCH_SYSTEM},
            {"role": "user", "content": user}]


def _build_monitor_messages(bundle: PromptBundle, candidate: str,
                            chain: List[str]) -> List[Dict[str, str]]:
    chain_text = "\n".join(f"  - {s}" for s in chain) if chain else "  (none given)"
    user = MONITOR_TEMPLATE.format(documents=format_documents(bundle.chunks),
                                   question=bundle.question,
                                   candidate=candidate, chain=chain_text)
    return [{"role": "system", "content": MONITOR_SYSTEM},
            {"role": "user", "content": user}]


# --- parsing -----------------------------------------------------------------

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_PUNCT = set(string.punctuation)


def _strip_fences(raw: str) -> str:
    text = raw.strip()
    return re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()


def _salvage(text: str) -> Dict[str, Any] | None:
    """A turn that ran out of token budget mid-object still usually contains the
    fields that matter. Pull them out rather than discarding the whole pass."""
    am = re.search(r'"answer"\s*:\s*(?:"((?:[^"\\]|\\.)*)"|(null))', text)
    if not am:
        return None
    obj: Dict[str, Any] = {"answer": None if am.group(2) else am.group(1)}
    cm = re.search(r'"confidence"\s*:\s*(\d+)', text)
    if cm:
        obj["confidence"] = int(cm.group(1))
    sm = re.search(r'"supported"\s*:\s*(true|false)', text)
    if sm:
        obj["supported"] = sm.group(1) == "true"
    return obj


def _loads(raw: str) -> Dict[str, Any] | None:
    """Best-effort JSON object extraction from a model turn."""
    text = _strip_fences(raw)
    m = _JSON_RE.search(text)
    if not m:
        # No closing brace at all — the turn was cut off mid-object.
        return _salvage(text) if "{" in text else None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return _salvage(m.group(0))
    return obj if isinstance(obj, dict) else None


def _clean_answer(ans: Any) -> str | None:
    if isinstance(ans, (int, float)):
        return str(ans)
    if isinstance(ans, list):
        ans = ", ".join(str(a) for a in ans if a is not None) or None
    if ans is None:
        return None
    if not isinstance(ans, str):
        ans = str(ans)
    ans = ans.strip().strip('"').strip()
    if not ans or ans.lower() in {"null", "none", "unknown", "n/a", "na"}:
        return None
    return ans


def _clean_cited(raw_cited: Any, n_chunks: int) -> List[int]:
    cited: List[int] = []
    for c in raw_cited if isinstance(raw_cited, list) else []:
        try:
            ci = int(c)
        except (TypeError, ValueError):
            continue
        if 1 <= ci <= n_chunks:
            cited.append(ci)
    return sorted(set(cited))


def parse_output(raw: str, n_chunks: int) -> Dict[str, Any]:
    """Best-effort JSON extraction. Returns answer (str|None), cited (list[int]),
    and parse_ok so malformed-output rate is itself a tracked metric."""
    obj = _loads(raw)
    if obj is not None:
        return {"answer": _clean_answer(obj.get("answer")),
                "cited": _clean_cited(obj.get("cited"), n_chunks),
                "parse_ok": True}
    text = _strip_fences(raw)
    return {"answer": text or None, "cited": [], "parse_ok": False}


# --- stage 3: CONTROL --------------------------------------------------------

def _norm(s: str | None) -> str:
    s = (s or "").lower()
    s = "".join(ch if ch not in _PUNCT else " " for ch in s)
    return " ".join(s.split())


def _attested(answer: str, chunks: Sequence[dict]) -> bool:
    """Source monitoring: is this answer actually present in the context, or did
    it come from the model's own store? Judgements ('yes'/'no') are exempt
    because they are inferred rather than quoted."""
    a = _norm(answer)
    if not a or a in _UNQUOTABLE:
        return True
    ctx = _norm(" ".join(f"{c.get('title') or ''} {c['text']}" for c in chunks))
    if a in ctx:
        return True
    # Tolerate minor rewording: accept if nearly every content token is attested.
    toks = [t for t in a.split() if len(t) > 2]
    if not toks:
        return True
    ctx_toks = set(ctx.split())
    return sum(t in ctx_toks for t in toks) / len(toks) >= 0.8


class GenerateRecognizeReport:
    """Runs RECALL -> MONITOR -> CONTROL over a batch. Presents the same
    `.generate(batch) -> list[str]` interface the harness expects, so `run.py`
    is untouched; the extra passes happen inside."""

    def __init__(self, backend):
        self.backend = backend
        self.name = backend.name

    def generate(self, batch: Sequence[List[Dict[str, str]]]) -> List[str]:
        bundles = list(batch)

        # --- stage 1: recall, with the model's own report criterion intact ---
        recalls = self.backend.generate(bundles, max_new_tokens=RECALL_TOKENS)
        stage1: List[Dict[str, Any]] = []
        for raw in recalls:
            obj = _loads(raw) or {}
            stage1.append({"answer": _clean_answer(obj.get("answer")),
                           "cited": obj.get("cited"),
                           "evidence": obj.get("evidence") if isinstance(
                               obj.get("evidence"), str) else "",
                           "shape": obj.get("shape")})

        # --- stage 2: re-search, only where the first reading came up empty ---
        # Koriat & Goldsmith's control process governs not only whether to
        # report but how much retrieval effort to spend before giving up. A
        # first-pass null is a decision to stop searching, and it is that
        # decision -- not the report criterion -- that costs the baseline most
        # of its Use F1.
        redo_idx = [i for i, s in enumerate(stage1)
                    if not s["answer"] and bundles[i].chunks]
        if redo_idx and not DISABLE_RESEARCH:
            msgs = [_build_research_messages(bundles[i]) for i in redo_idx]
            outs = self.backend.generate(msgs, max_new_tokens=RESEARCH_TOKENS)
            for i, raw in zip(redo_idx, outs):
                obj = _loads(raw) or {}
                rec = _clean_answer(obj.get("answer"))
                if rec:
                    stage1[i] = {"answer": rec, "cited": obj.get("cited"),
                                 "evidence": obj.get("evidence") if isinstance(
                                     obj.get("evidence"), str) else "",
                                 "shape": stage1[i]["shape"], "recovered": True}

        # --- stage 3: verify recoveries only ---------------------------------
        # An answer the first reading produced carries the model's own
        # endorsement already. An answer produced only after being pressed to
        # look again does not, so that is where verification is worth its cost.
        verify_idx = [i for i, s in enumerate(stage1)
                      if s["answer"] and bundles[i].chunks
                      and (VERIFY_ALL or s.get("recovered"))]
        verdicts: Dict[int, Dict[str, Any]] = {}
        if verify_idx and VERIFY_RECOVERIES:
            msgs = [_build_monitor_messages(
                bundles[i], stage1[i]["answer"],
                [s for s in (stage1[i].get("evidence"),) if s]) for i in verify_idx]
            outs = self.backend.generate(msgs, max_new_tokens=MONITOR_TOKENS)
            for i, raw in zip(verify_idx, outs):
                verdicts[i] = _loads(raw) or {}

        return [json.dumps(self._report(i, b, stage1[i], verdicts.get(i)),
                           ensure_ascii=False) for i, b in enumerate(bundles)]

    def _report(self, i: int, bundle: PromptBundle, s1: Dict[str, Any],
                verdict: Dict[str, Any] | None) -> Dict[str, Any]:
        n = len(bundle.chunks)
        withheld = lambda why: {"answer": None, "cited": [], "_why": why}
        remap = lambda cs: sorted({bundle.order[c - 1] + 1
                                   for c in _clean_cited(cs, n)})

        if not s1["answer"]:
            return withheld("no answer after re-search")
        if not n:
            return {"answer": s1["answer"], "cited": []}

        if verdict is None and not s1.get("recovered"):
            return {"answer": s1["answer"], "cited": remap(s1["cited"]),
                    "_src": "recall"}
        if verdict is None:
            return {"answer": s1["answer"], "cited": remap(s1["cited"]),
                    "_src": "recovered", "_ver": "skipped"}

        conf = verdict.get("confidence")
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 0.0
        supported = verdict.get("supported")
        if isinstance(supported, str):
            supported = supported.strip().lower() in {"true", "yes", "1"}

        answer = _clean_answer(verdict.get("answer")) or s1["answer"]
        dbg = {"_src": "recovered" if s1.get("recovered") else "recall",
               "_cand": s1["answer"], "_clo": round(bundle.closure, 4),
               "_conf": conf, "_sup": supported,
               "_tgt": bool(verdict.get("target")),
               "_stated": bool(verdict.get("stated")),
               "_vans": _clean_answer(verdict.get("answer"))}
        if GATE and (supported is False or conf < REPORT_CRITERION):
            return {**withheld(f"rejected (conf {conf:.0f})"), **dbg}
        if GATE and bundle.closure < CLOSURE_FLOOR:
            return {**withheld(f"no chain closure ({bundle.closure:.3f})"), **dbg}
        return {"answer": answer, "cited": remap(verdict.get("cited") or s1["cited"]),
                **dbg}


class HFChatModel:
    """
    Defalut runner for the LLM model. You should only try to load in 4 bit
    if you're having trouble fitting the model on your computer.
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
                 device: str = "auto", max_new_tokens: int = 128,
                 dtype: str = "auto", load_4bit: bool = False,
                 max_input_tokens: int = 8192, **_):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers import __version__ as tf_version

        self.name = model_name
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        kwargs: Dict[str, Any] = {"device_map": device}
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            key = ("dtype" if tuple(int(x) for x in tf_version.split(".")[:2]) >= (4, 56)
                   else "torch_dtype")
            kwargs[key] = getattr(torch, dtype) if dtype != "auto" else "auto"

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.eval()
        self.torch = torch

    def generate(self, batch: Sequence[List[Dict[str, str]]],
                 max_new_tokens: int | None = None) -> List[str]:
        texts = [self.tokenizer.apply_chat_template(
            m, tokenize=False, add_generation_prompt=True) for m in batch]
        enc = self.tokenizer(texts, return_tensors="pt", padding=True,
                             padding_side="left", truncation=True,
                             max_length=self.max_input_tokens).to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**enc,
                                      max_new_tokens=max_new_tokens
                                      or self.max_new_tokens,
                                      do_sample=False,
                                      pad_token_id=self.tokenizer.pad_token_id
                                      or self.tokenizer.eos_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(gen, skip_special_tokens=True)


def get_model(model_name: str, **kwargs):
    backend = HFChatModel(model_name, **kwargs)
    if os.environ.get("GRR_DISABLE") == "1":       # baseline passthrough
        return backend
    return GenerateRecognizeReport(backend)
