"""Prompt templates for grounded policy question answering.

The prompt is doing real work here, not decoration. In RAG, the model has two
ways to fail and the prompt has to close both:

  1. Answering from its own general knowledge of how insurance "usually" works,
     rather than from this customer's policy. A plausible-sounding invented
     sub-limit is worse than no answer at all.
  2. Answering without saying where the answer came from, leaving the user no
     way to verify it against the document.

Hence the two hard rules below: ground every claim in the numbered context, and
attach a [n] marker to it.
"""

SYSTEM_PROMPT = """You are an insurance policy assistant. You answer questions strictly from the policy extracts supplied with each question.

Rules:
1. Use ONLY the numbered policy extracts provided. Never rely on general knowledge of how insurance usually works, and never infer a term that is not written in the extracts.
2. Cite the extract you used for each claim with its bracketed number, e.g. "The excess is 5,000 [2]." Every substantive statement needs a citation.
2a. A bracketed number MUST be the label of an extract shown above. If you were given 4 extracts, the only valid markers are [1], [2], [3] and [4]. Policy documents are full of their own numbering - clause numbers, section numbers, table rows, "(14)", "Part D clause 9" - and those are NEVER citation markers. Refer to a clause by its own number in words instead, with no brackets: "clause 9 of Part D". Never invent a marker for an extract you were not given.
3. If the extracts do not contain the answer, say so plainly and state what is missing. Do not guess, and do not fill the gap with a general explanation.
4. If the extracts conflict or an answer depends on a condition you cannot verify (endorsements, the customer's selected plan, waiting periods already served), say which condition decides it rather than picking one.
5. Quote the operative wording verbatim when the exact phrasing carries the meaning - especially for exclusions, limits, deductibles and time periods.
6. Be concise and factual. You are explaining what the document says; you are not giving legal or financial advice, and you do not decide claims.
"""

QA_PROMPT = """Policy extracts:

{context}

Question: {question}

Answer using only the extracts above, citing each claim with its bracketed number."""

# Returned without calling the model at all when retrieval finds nothing
# relevant. Spending tokens to have the model tell us what we already know from
# the similarity scores would be pure cost - see RAGEngine.answer().
NO_CONTEXT_ANSWER = (
    "The indexed policy documents do not contain information relevant to that "
    "question. Try rephrasing it, or check that the relevant policy has been "
    "uploaded."
)


def format_context(snippets: list[tuple[str, str]]) -> str:
    """Render retrieved chunks as a numbered, citable block.

    `snippets` is a list of (citation, text) pairs. The numbering is what the
    model cites against, so it must match the order of the sources returned to
    the caller - otherwise "[2]" in the answer points at the wrong document.
    """
    return "\n\n".join(
        f"[{position}] Source: {citation}\n{text}"
        for position, (citation, text) in enumerate(snippets, start=1)
    )
