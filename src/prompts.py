"""Prompt builders + answer extraction.

One baseline format is shared by (a) zero-shot baseline inference, (b) the
fine-tuning source text (Conditions B & C), and (c) the generalization pass on
TabFact (statement fed as the question). The CoT builder prepends hand-written
exemplars. Keeping a single source format means a model fine-tuned on WTQ sees
the same surface form when evaluated on TabFact.
"""

from __future__ import annotations

from . import config
from .cot_exemplars import format_exemplar, get_exemplars
from .data import serialize_table

BASELINE_INSTRUCTION = "Answer the question based on the table."
ANSWER_TAG = "Answer:"


def build_baseline_prompt(example: dict) -> str:
    """Zero-shot / fine-tune source: instruction + question + serialized table."""
    return "\n".join(
        [
            BASELINE_INSTRUCTION,
            f"Question: {example['question']}",
            "Table:",
            serialize_table(example["table"]),
            ANSWER_TAG,
        ]
    )


TABFACT_INSTRUCTION = "Read the table and decide whether the statement is true or false."


def build_tabfact_prompt(example: dict) -> str:
    """Zero-shot TabFact prompt: instruction + statement + table, asking for true/false.

    TabFact is true/false verification, not open QA. Feeding it through the WTQ
    'answer the question' prompt never tells the model to output true or false,
    so the generation is unmappable and the score collapses. This prompt asks for
    the label directly; no TabFact training happens, so it stays a fair OOD test.
    """
    return "\n".join(
        [
            TABFACT_INSTRUCTION,
            f"Statement: {example['question']}",
            "Table:",
            serialize_table(example["table"]),
            "Answer (true or false):",
        ]
    )


def build_cot_prompt(example: dict, style: str = "plain", n_shots: int = 6) -> str:
    """Few-shot CoT: n exemplars (with reasoning) then the query ending in 'Reasoning:'.

    style in {"plain", "structured"} selects the chain format (proposal Condition A).
    """
    shots = get_exemplars(style)[:n_shots]
    blocks = [format_exemplar(ex, style) for ex in shots]
    # Put the question right before "Reasoning:" so it survives left-truncation
    # of long prompts (the table can be big; the question must not be cut).
    query = "\n".join(
        [
            "Table:",
            serialize_table(example["table"]),
            f"Question: {example['question']}",
            "Reasoning:",
        ]
    )
    return "\n\n".join(blocks + [query])


def extract_answer(generated_text: str) -> str:
    """Pull the final answer out of a model generation.

    CoT outputs end with 'Answer: X'; take the text after the last tag. Plain
    generations (no tag) are returned stripped as-is.
    """
    if ANSWER_TAG in generated_text:
        return generated_text.rsplit(ANSWER_TAG, 1)[-1].strip()
    return generated_text.strip()


def build_train_source(example: dict) -> str:
    """Fine-tuning source text (same as the baseline prompt)."""
    return build_baseline_prompt(example)


def build_train_target(example: dict, trace: str | None = None) -> str:
    """Fine-tuning target.

    Condition B (answers-only): target is the gold answer.
    Condition C (with trace):   target is the reasoning chain (which already
                                ends in 'Answer: <gold>').
    """
    if trace is not None:
        return trace
    return example["answer"]
