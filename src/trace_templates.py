"""Rule-based reasoning-trace generator for Condition C (trace fine-tuning).

Only returns a trace when the derivation is genuinely unambiguous. A wrong
trace used as a training target poisons the experiment, so conservative is
correct.  When in doubt, return None.

Public API
----------
generate_trace(example: dict) -> str | None
    Returns a short reasoning-chain string ending in "Answer: <gold>", or None.

trace_coverage(examples: list[dict]) -> dict
    Returns {"total": N, "with_trace": k, "coverage": k/N}.
"""

from __future__ import annotations


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _gold(example: dict) -> str:
    """Return the canonical gold answer string for an example."""
    answers = example.get("answers")
    if answers and len(answers) > 0:
        return answers[0]
    return example["answer"]


def _normalize(text: str) -> str:
    """Lowercase and strip for cell-matching purposes."""
    return text.lower().strip()


def _find_cell_matches(table: dict, norm_gold: str) -> list[tuple[int, int]]:
    """Return list of (row_index, col_index) where the normalized cell == norm_gold.

    Row index is 0-based into table["rows"]; col_index into table["header"].
    """
    matches: list[tuple[int, int]] = []
    for row_idx, row in enumerate(table["rows"]):
        for col_idx, cell in enumerate(row):
            if _normalize(str(cell)) == norm_gold:
                matches.append((row_idx, col_idx))
    return matches


# --------------------------------------------------------------------------- #
# Rule 1: Unique single-cell lookup
# --------------------------------------------------------------------------- #

def _rule_unique_lookup(example: dict) -> str | None:
    """Emit a trace when the gold value appears in exactly one cell."""
    gold = _gold(example)
    norm_gold = _normalize(gold)
    if not norm_gold:
        return None

    table = example["table"]
    matches = _find_cell_matches(table, norm_gold)

    if len(matches) != 1:
        # Zero matches (e.g. computed answer not literally in table) or >1, ambiguous.
        return None

    _row_idx, col_idx = matches[0]
    header = table["header"]
    col_name = str(header[col_idx])

    return (
        f"Scan the table for '{gold}'. "
        f"It appears once, in column '{col_name}'. "
        f"Answer: {gold}"
    )


# --------------------------------------------------------------------------- #
# Rule 2: Count-all rows
# --------------------------------------------------------------------------- #

def _rule_count_all_rows(example: dict) -> str | None:
    """Emit a trace when the question asks 'how many' and gold == total row count."""
    question = example.get("question", "")
    if not question.lower().startswith("how many"):
        return None

    gold = _gold(example)
    norm_gold = gold.strip()

    # Gold must be an integer string.
    try:
        gold_int = int(norm_gold)
    except ValueError:
        return None

    n_rows = len(example["table"]["rows"])
    if gold_int != n_rows:
        # Only emit when the answer unambiguously equals the full row count.
        return None

    return (
        f"The question asks how many. "
        f"Count the rows in the table: there are {n_rows}. "
        f"Answer: {norm_gold}"
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def generate_trace(example: dict) -> str | None:
    """Return a verifiable reasoning trace, or None if no rule applies.

    Rules are tried in priority order; the first that fires wins.

    Parameters
    ----------
    example : dict
        A unified WTQ example dict (see src/data.py for schema).

    Returns
    -------
    str
        A short reasoning chain ending in "Answer: <gold>".
    None
        When no rule yields an unambiguous derivation.
    """
    # Rule 1: unique single-cell lookup
    trace = _rule_unique_lookup(example)
    if trace is not None:
        return trace

    # Rule 2: count-all rows
    trace = _rule_count_all_rows(example)
    if trace is not None:
        return trace

    # No rule fired.
    return None


def trace_coverage(examples: list[dict]) -> dict:
    """Report what fraction of examples received a trace.

    Parameters
    ----------
    examples : list[dict]
        A list of unified example dicts.

    Returns
    -------
    dict with keys "total", "with_trace", "coverage".
    """
    total = len(examples)
    with_trace = sum(1 for ex in examples if generate_trace(ex) is not None)
    coverage = with_trace / total if total > 0 else 0.0
    return {"total": total, "with_trace": with_trace, "coverage": coverage}
