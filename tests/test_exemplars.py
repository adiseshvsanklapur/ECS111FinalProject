"""Tests for src/cot_exemplars.py.

All expected values are hand-verified against the tables defined in the module.
"""

import sys
import os

# Make the src package importable when running pytest from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.cot_exemplars import (
    EXEMPLARS_PLAIN,
    EXEMPLARS_STRUCTURED,
    format_exemplar,
    get_exemplars,
)
from src.data import serialize_table


# ---------------------------------------------------------------------------
# List-level structural tests
# ---------------------------------------------------------------------------

class TestListSizes:
    def test_plain_has_at_least_8(self):
        assert len(EXEMPLARS_PLAIN) >= 8

    def test_structured_has_at_least_8(self):
        assert len(EXEMPLARS_STRUCTURED) >= 8


# ---------------------------------------------------------------------------
# Per-exemplar validity: every field must be non-empty and table non-trivial
# ---------------------------------------------------------------------------

class TestExemplarValidity:
    @pytest.mark.parametrize("ex", EXEMPLARS_PLAIN, ids=[f"plain-{i}" for i in range(len(EXEMPLARS_PLAIN))])
    def test_plain_fields_non_empty(self, ex):
        assert ex["question"].strip(), "question must not be empty"
        assert ex["reasoning"].strip(), "reasoning must not be empty"
        assert ex["answer"].strip(), "answer must not be empty"
        assert len(ex["table"]["rows"]) >= 1, "table must have at least one row"

    @pytest.mark.parametrize("ex", EXEMPLARS_STRUCTURED, ids=[f"structured-{i}" for i in range(len(EXEMPLARS_STRUCTURED))])
    def test_structured_fields_non_empty(self, ex):
        assert ex["question"].strip(), "question must not be empty"
        assert ex["reasoning"].strip(), "reasoning must not be empty"
        assert ex["answer"].strip(), "answer must not be empty"
        assert len(ex["table"]["rows"]) >= 1, "table must have at least one row"


# ---------------------------------------------------------------------------
# format_exemplar: output structure
# ---------------------------------------------------------------------------

class TestFormatExemplar:
    def test_plain_ends_with_answer(self):
        ex = EXEMPLARS_PLAIN[0]
        result = format_exemplar(ex, "plain")
        assert result.endswith(f"Answer: {ex['answer']}")

    def test_structured_ends_with_answer(self):
        ex = EXEMPLARS_STRUCTURED[0]
        result = format_exemplar(ex, "structured")
        assert result.endswith(f"Answer: {ex['answer']}")

    def test_plain_last_exemplar_ends_with_answer(self):
        ex = EXEMPLARS_PLAIN[-1]
        result = format_exemplar(ex, "plain")
        assert result.endswith(f"Answer: {ex['answer']}")

    def test_structured_last_exemplar_ends_with_answer(self):
        ex = EXEMPLARS_STRUCTURED[-1]
        result = format_exemplar(ex, "structured")
        assert result.endswith(f"Answer: {ex['answer']}")

    @pytest.mark.parametrize("ex", EXEMPLARS_PLAIN, ids=[f"plain-fmt-{i}" for i in range(len(EXEMPLARS_PLAIN))])
    def test_plain_format_ends_with_answer_all(self, ex):
        result = format_exemplar(ex, "plain")
        assert result.endswith(f"Answer: {ex['answer']}")

    @pytest.mark.parametrize("ex", EXEMPLARS_STRUCTURED, ids=[f"structured-fmt-{i}" for i in range(len(EXEMPLARS_STRUCTURED))])
    def test_structured_format_ends_with_answer_all(self, ex):
        result = format_exemplar(ex, "structured")
        assert result.endswith(f"Answer: {ex['answer']}")

    def test_serialized_table_in_plain_output(self):
        ex = EXEMPLARS_PLAIN[0]
        result = format_exemplar(ex, "plain")
        expected_table = serialize_table(ex["table"])
        assert expected_table in result

    def test_serialized_table_in_structured_output(self):
        ex = EXEMPLARS_STRUCTURED[0]
        result = format_exemplar(ex, "structured")
        expected_table = serialize_table(ex["table"])
        assert expected_table in result

    @pytest.mark.parametrize("ex", EXEMPLARS_PLAIN, ids=[f"plain-table-{i}" for i in range(len(EXEMPLARS_PLAIN))])
    def test_plain_serialized_table_present_all(self, ex):
        result = format_exemplar(ex, "plain")
        assert serialize_table(ex["table"]) in result

    @pytest.mark.parametrize("ex", EXEMPLARS_STRUCTURED, ids=[f"structured-table-{i}" for i in range(len(EXEMPLARS_STRUCTURED))])
    def test_structured_serialized_table_present_all(self, ex):
        result = format_exemplar(ex, "structured")
        assert serialize_table(ex["table"]) in result

    def test_format_contains_question(self):
        ex = EXEMPLARS_PLAIN[0]
        result = format_exemplar(ex, "plain")
        assert ex["question"] in result

    def test_format_contains_reasoning(self):
        ex = EXEMPLARS_PLAIN[0]
        result = format_exemplar(ex, "plain")
        assert ex["reasoning"] in result


# ---------------------------------------------------------------------------
# get_exemplars
# ---------------------------------------------------------------------------

class TestGetExemplars:
    def test_get_plain_returns_plain_list(self):
        assert get_exemplars("plain") is EXEMPLARS_PLAIN

    def test_get_structured_returns_structured_list(self):
        assert get_exemplars("structured") is EXEMPLARS_STRUCTURED

    def test_invalid_style_raises_value_error(self):
        with pytest.raises(ValueError):
            get_exemplars("unknown")


# ---------------------------------------------------------------------------
# Spot-check answer correctness for a sample of exemplars
# ---------------------------------------------------------------------------

class TestAnswerCorrectnessSpotCheck:
    """Verify a selection of known answer values by re-deriving them from the table."""

    def test_alice_score_lookup(self):
        # Ex 1 plain / structured: simple cell lookup
        ex = EXEMPLARS_PLAIN[0]
        rows_for_alice = [r for r in ex["table"]["rows"] if r[0] == "Alice"]
        assert len(rows_for_alice) == 1
        assert rows_for_alice[0][1] == ex["answer"]

    def test_city_population_filter(self):
        # Ex 2: row filtering
        ex = EXEMPLARS_PLAIN[1]
        qualifying = [r[0] for r in ex["table"]["rows"] if float(r[1]) > 2.0]
        assert len(qualifying) == 1
        assert qualifying[0] == ex["answer"]

    def test_product_count(self):
        # Ex 3: counting
        ex = EXEMPLARS_PLAIN[2]
        count = sum(1 for r in ex["table"]["rows"] if float(r[1]) > 20)
        assert str(count) == ex["answer"]

    def test_revenue_sum(self):
        # Ex 4: sum aggregation
        ex = EXEMPLARS_PLAIN[3]
        total = sum(int(r[1]) for r in ex["table"]["rows"])
        assert str(total) == ex["answer"]

    def test_max_temperature(self):
        # Ex 5: max aggregation
        ex = EXEMPLARS_PLAIN[4]
        max_temp = max(int(r[1]) for r in ex["table"]["rows"])
        assert str(max_temp) == ex["answer"]

    def test_lions_vs_tigers_comparison(self):
        # Ex 6: comparison — Lions wins > Tigers wins => answer "yes"
        ex = EXEMPLARS_PLAIN[5]
        lions_wins = int([r[1] for r in ex["table"]["rows"] if r[0] == "Lions"][0])
        tigers_wins = int([r[1] for r in ex["table"]["rows"] if r[0] == "Tigers"][0])
        assert lions_wins > tigers_wins
        assert ex["answer"] == "yes"

    def test_smallest_country_area(self):
        # Ex 7: superlative min
        ex = EXEMPLARS_PLAIN[6]
        min_row = min(ex["table"]["rows"], key=lambda r: int(r[1]))
        assert min_row[0] == ex["answer"]

    def test_cheapest_product_rating(self):
        # Ex 8: multi-hop — cheapest product's rating
        ex = EXEMPLARS_PLAIN[7]
        cheapest = min(ex["table"]["rows"], key=lambda r: float(r[1]))
        assert cheapest[2] == ex["answer"]
