"""Apply the four-way question-shape taxonomy to model/llm.py.

The two-way split (extraction / comparison) gained +0.09 on the task types it
described correctly (2wiki/comparison, 2wiki/compositional) and lost ~-0.10 on
the two it forced into the wrong box: bridge_comparison, which is both shapes
at once, and inference, which is neither.
"""
import sys
p = "model/llm.py"
src = open(p).read()

old = src[src.index('RECALL_TEMPLATE = """'):src.index('RECALL_NO_CONTEXT = """')]
new = '''RECALL_TEMPLATE = """Documents:
{documents}

Question: {question}

Work out which of these four shapes the question has. Getting this wrong is the
single most common way to produce a confidently wrong answer, because each shape
has a different kind of answer.

EXTRACTION — "where did X die", "when was Z founded". One document states the
property for the entity named. The answer is a short span copied from it.

BRIDGE — "where did the performer of song X die", "who is Y's husband's mother".
The question never names the entity it is really about. First find the document
that resolves the indirect reference into a name, then find the document about
THAT name. Search for the resolved name, not for the question's wording: the
answering document usually shares almost no words with the question. The answer
is a span from the second document.

COMPARISON — "which came first", "are X and Y from the same country", "who was
born earlier". Look up the same property for BOTH entities, then compare. The
answer is the RESULT — the winning entity's name, or exactly "yes" or "no" — and
it appears nowhere as a span. Never answer with one of the looked-up values: for
"which film is older" the answer is a film title, not a year. If the question
asks which of two entities has a property, the answer is one of those entities.

RELATION — "who is the grandmother of X", "who is the sibling-in-law of Y". The
answer is reached by composing family or organisational relations across
documents, and no document states the composed relation. Spell the composition
out ("X's father is A; A's mother is B"), then answer with the endpoint. Do not
answer with an intermediate entity, and do not answer with a date or place when
the question asks for a person.

The answer must match the type the question asks for: a place for "where", a
person for "who", a date for "when", a title for "which work", exactly "yes" or
"no" for a yes/no question. Give the shortest form that answers it — no
explanation, no restating the question, no clause describing when or how
something happened.

If the documents do not support an answer, use null. Do not guess from your own
knowledge, and do not treat a document that merely mentions the right people or
places as support for a property it never states.

Reply with JSON only, in this exact form:
{{"shape": "extraction" | "bridge" | "comparison" | "relation", "evidence": "<the fact(s) you used, quoted>", "answer": "<answer, or null>", "cited": [<document numbers used>]}}"""

'''
assert old != new
src = src.replace(old, new, 1)

# The monitor must apply the matching standard of proof per shape.
old_m = '''4. For a comparison question, step 3 succeeds only if the property is recorded
   for BOTH compared entities.'''
new_m = '''4. Match the standard of proof to the question's shape. For a comparison, step 3
   succeeds only if the property is recorded for BOTH compared entities. For a
   composed relation ("grandmother of X"), it succeeds if every individual link
   is recorded, even though no sentence states the composed relation itself —
   do not demand one.'''
assert old_m in src
src = src.replace(old_m, new_m, 1)

open(p, "w").write(src)
print("four-way shape taxonomy applied")
