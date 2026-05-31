"""Shape tests for prompt builders and answer extraction."""

from src.prompts import (
    build_baseline_prompt,
    build_cot_prompt,
    build_tabfact_prompt,
    build_train_source,
    build_train_target,
    extract_answer,
)

EXAMPLE = {
    "question": "What is Bob's score?",
    "table": {"header": ["Name", "Score"], "rows": [["Alice", "92"], ["Bob", "78"]]},
    "answer": "78",
}


def test_baseline_prompt_shape():
    p = build_baseline_prompt(EXAMPLE)
    assert "Answer the question based on the table." in p
    assert "Question: What is Bob's score?" in p
    assert "Name | Score" in p
    assert p.rstrip().endswith("Answer:")


def test_tabfact_prompt_asks_for_true_false():
    statement = {"question": "Bob scored more than Alice.",
                 "table": EXAMPLE["table"], "answer": "false"}
    p = build_tabfact_prompt(statement)
    assert "true or false" in p.lower()
    assert "Statement: Bob scored more than Alice." in p
    assert "Name | Score" in p
    # must NOT carry the WTQ answer tag that extract_answer keys on
    assert "Answer:" not in p


def test_cot_prompt_has_shots_and_ends_with_reasoning():
    p = build_cot_prompt(EXAMPLE, style="plain", n_shots=2)
    # 2 exemplars each carry a "Reasoning:" line, plus the query's trailing "Reasoning:"
    assert p.count("Reasoning:") == 3
    assert p.rstrip().endswith("Reasoning:")
    assert "What is Bob's score?" in p


def test_cot_prompt_structured_style():
    p = build_cot_prompt(EXAMPLE, style="structured", n_shots=3)
    assert p.count("Reasoning:") == 4  # 3 shots + query


def test_extract_answer_takes_last_tag():
    assert extract_answer("some reasoning Answer: 78") == "78"
    assert extract_answer("Answer: a then Answer: b") == "b"
    assert extract_answer("no tag here") == "no tag here"


def test_train_source_matches_baseline():
    assert build_train_source(EXAMPLE) == build_baseline_prompt(EXAMPLE)


def test_train_target_answer_vs_trace():
    assert build_train_target(EXAMPLE) == "78"
    assert build_train_target(EXAMPLE, trace="steps... Answer: 78") == "steps... Answer: 78"
