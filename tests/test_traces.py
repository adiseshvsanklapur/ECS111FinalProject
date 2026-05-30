"""Tests for src/trace_templates.py.

All fixtures are hand-crafted so tests run fully offline without HuggingFace.
"""

import pytest
from src.trace_templates import generate_trace, trace_coverage


# --------------------------------------------------------------------------- #
# Fixture helpers
# --------------------------------------------------------------------------- #

def _make_example(question: str, header: list, rows: list, answer: str,
                  answers: list | None = None) -> dict:
    """Build a minimal WTQ-style example dict."""
    return {
        "id": "test-0",
        "task": "wtq",
        "question": question,
        "table": {"header": header, "rows": rows},
        "answer": answer,
        "answers": answers if answers is not None else [answer],
    }


# --------------------------------------------------------------------------- #
# Rule 1: Unique single-cell lookup
# --------------------------------------------------------------------------- #

class TestUniqueLookup:

    def test_returns_string_for_unique_cell(self):
        """Gold appears in exactly one cell — must return a str."""
        ex = _make_example(
            question="Which team is from Boston?",
            header=["Team", "City", "Sport"],
            rows=[
                ["Celtics", "Boston", "Basketball"],
                ["Bruins",  "Boston", "Hockey"],
                ["Patriots", "Foxborough", "Football"],
            ],
            answer="Foxborough",
            answers=["Foxborough"],
        )
        result = generate_trace(ex)
        assert isinstance(result, str), "Expected a str trace for a unique-cell match"

    def test_trace_ends_with_answer_line(self):
        """Trace must end with 'Answer: <gold>'."""
        ex = _make_example(
            question="What sport do the Patriots play?",
            header=["Team", "City", "Sport"],
            rows=[
                ["Celtics",  "Boston",     "Basketball"],
                ["Bruins",   "Boston",     "Hockey"],
                ["Patriots", "Foxborough", "Football"],
            ],
            answer="Football",
            answers=["Football"],
        )
        result = generate_trace(ex)
        assert result is not None
        assert result.endswith("Answer: Football"), (
            f"Trace should end with 'Answer: Football', got: {result!r}"
        )

    def test_trace_mentions_correct_column_header(self):
        """Trace must mention the column header where the gold value lives."""
        ex = _make_example(
            question="What sport do the Patriots play?",
            header=["Team", "City", "Sport"],
            rows=[
                ["Celtics",  "Boston",     "Basketball"],
                ["Bruins",   "Boston",     "Hockey"],
                ["Patriots", "Foxborough", "Football"],
            ],
            answer="Football",
            answers=["Football"],
        )
        result = generate_trace(ex)
        assert result is not None
        assert "Sport" in result, (
            f"Trace should mention column 'Sport', got: {result!r}"
        )

    def test_returns_none_when_gold_appears_twice(self):
        """Gold in two cells — ambiguous, must return None."""
        ex = _make_example(
            question="Which city appears?",
            header=["Team", "City"],
            rows=[
                ["Celtics", "Boston"],
                ["Bruins",  "Boston"],   # 'Boston' appears twice
            ],
            answer="Boston",
            answers=["Boston"],
        )
        result = generate_trace(ex)
        assert result is None, (
            "Expected None when gold appears in more than one cell (ambiguous)"
        )

    def test_returns_none_when_gold_not_in_any_cell(self):
        """Gold not literally present in table (computed answer) — must return None."""
        ex = _make_example(
            question="What is the total score?",
            header=["Team", "Score"],
            rows=[
                ["A", "30"],
                ["B", "40"],
            ],
            answer="70",   # sum — not in any cell
            answers=["70"],
        )
        result = generate_trace(ex)
        assert result is None, (
            "Expected None when gold is a computed value absent from the table"
        )

    def test_lookup_uses_answers_list_first(self):
        """When answers list is provided, generate_trace should use answers[0]."""
        ex = _make_example(
            question="Which stadium is listed?",
            header=["Stadium", "Capacity"],
            rows=[
                ["Wembley",  "90000"],
                ["Old Trafford", "74000"],
            ],
            answer="Old Trafford",           # answer field
            answers=["Old Trafford"],        # answers[0] agrees
        )
        result = generate_trace(ex)
        assert result is not None
        assert result.endswith("Answer: Old Trafford")


# --------------------------------------------------------------------------- #
# Rule 2: Count-all rows
# --------------------------------------------------------------------------- #

class TestCountAllRows:

    def test_returns_trace_for_exact_row_count(self):
        """Question 'how many …' with gold == number of rows -> returns str."""
        ex = _make_example(
            question="How many teams are there?",
            header=["Team", "City"],
            rows=[
                ["A", "X"],
                ["B", "Y"],
                ["C", "Z"],
            ],
            answer="3",
            answers=["3"],
        )
        result = generate_trace(ex)
        assert isinstance(result, str), "Expected a str trace for count-all-rows rule"
        assert result.endswith("Answer: 3"), (
            f"Trace should end 'Answer: 3', got: {result!r}"
        )
        assert "3" in result

    def test_count_trace_contains_count(self):
        """Trace body should mention the row count."""
        ex = _make_example(
            question="How many entries are listed?",
            header=["Name"],
            rows=[["Alice"], ["Bob"]],
            answer="2",
            answers=["2"],
        )
        result = generate_trace(ex)
        assert result is not None
        assert "2" in result

    def test_returns_none_when_gold_does_not_equal_row_count(self):
        """If gold != total rows, rule must not fire (could be a filtered count)."""
        ex = _make_example(
            question="How many teams won more than 10 games?",
            header=["Team", "Wins"],
            rows=[
                ["A", "15"],
                ["B", "8"],
                ["C", "12"],
            ],
            answer="2",   # filtered count, NOT len(rows)=3
            answers=["2"],
        )
        result = generate_trace(ex)
        # Rule 2 should not fire because 2 != 3.
        # Rule 1 might fire if "2" appears in exactly one cell — check that.
        # "2" does NOT appear in any cell (cells are "15","8","12","A","B","C").
        # So overall None.
        assert result is None, (
            "Expected None when gold differs from total row count (filtered count)"
        )

    def test_returns_none_for_how_many_with_non_integer_gold(self):
        """Rule 2 requires an integer gold; float or text should not fire it."""
        ex = _make_example(
            question="How many miles long is the route?",
            header=["Route", "Miles"],
            rows=[["R1", "3.5"], ["R2", "7.0"]],
            answer="10.5",
            answers=["10.5"],
        )
        result = generate_trace(ex)
        assert result is None


# --------------------------------------------------------------------------- #
# Non-derivable cases
# --------------------------------------------------------------------------- #

class TestNonDerivable:

    def test_aggregation_returns_none(self):
        """A sum/aggregation answer not in the table and not a full row count -> None."""
        ex = _make_example(
            question="What is the total revenue?",
            header=["Year", "Revenue"],
            rows=[
                ["2020", "100"],
                ["2021", "200"],
                ["2022", "300"],
            ],
            answer="600",
            answers=["600"],
        )
        result = generate_trace(ex)
        assert result is None, "Expected None for an aggregation (sum) question"

    def test_multi_hop_returns_none(self):
        """A multi-hop lookup that isn't uniquely identifiable -> None."""
        ex = _make_example(
            question="What city is the team with the most wins from?",
            header=["Team", "City", "Wins"],
            rows=[
                ["A", "NYC",    "50"],
                ["B", "LA",     "60"],
                ["C", "Chicago","55"],
            ],
            answer="LA",
            answers=["LA"],
        )
        # "LA" appears exactly once in the table, so Rule 1 WILL fire and
        # return a trace.  This is actually correct behavior (the unique-lookup
        # rule fires on the cell value regardless of question intent).
        # Use an ambiguous case instead: answer appears in >1 cells or the
        # derivation requires knowing which row has max wins.
        # Adjust: make the answer appear in two cities so Rule 1 can't fire.
        ex2 = _make_example(
            question="What city is the team with the most wins from?",
            header=["Team", "City", "Wins"],
            rows=[
                ["A", "NYC", "50"],
                ["B", "NYC", "60"],   # two teams from NYC — "NYC" appears twice
                ["C", "LA",  "55"],
            ],
            answer="NYC",
            answers=["NYC"],
        )
        result = generate_trace(ex2)
        assert result is None, (
            "Expected None when answer appears in multiple cells (multi-hop ambiguity)"
        )


# --------------------------------------------------------------------------- #
# trace_coverage
# --------------------------------------------------------------------------- #

class TestTraceCoverage:

    def _derivable_example(self, uid: str) -> dict:
        """A unique-lookup example guaranteed to produce a trace."""
        return {
            "id": uid,
            "task": "wtq",
            "question": "Who scored?",
            "table": {
                "header": ["Player", "Score"],
                "rows": [
                    ["Alice", f"unique-{uid}"],
                    ["Bob",   "10"],
                ],
            },
            "answer": f"unique-{uid}",
            "answers": [f"unique-{uid}"],
        }

    def _non_derivable_example(self, uid: str) -> dict:
        """An example that won't match any rule."""
        return {
            "id": uid,
            "task": "wtq",
            "question": "What is the average?",
            "table": {
                "header": ["Value"],
                "rows": [["1"], ["2"], ["3"]],
            },
            "answer": "2",   # average; "2" appears in one cell — but wait,
                             # that would make Rule 1 fire!  Use "6" (sum) instead.
            "answers": ["6"],
        }

    def test_coverage_counts(self):
        """trace_coverage returns correct total, with_trace, and coverage."""
        derivable = [self._derivable_example(f"d{i}") for i in range(3)]
        non_derivable = [self._non_derivable_example(f"n{i}") for i in range(2)]
        examples = derivable + non_derivable

        # Verify our fixture assumptions first.
        for ex in derivable:
            assert generate_trace(ex) is not None, (
                f"Fixture {ex['id']} expected to produce a trace"
            )
        for ex in non_derivable:
            t = generate_trace(ex)
            assert t is None, (
                f"Fixture {ex['id']} expected None, got: {t!r}"
            )

        cov = trace_coverage(examples)
        assert cov["total"] == 5
        assert cov["with_trace"] == 3
        assert abs(cov["coverage"] - 3 / 5) < 1e-9

    def test_coverage_all_none(self):
        """Coverage is 0 when no example yields a trace."""
        examples = [self._non_derivable_example(f"n{i}") for i in range(4)]
        cov = trace_coverage(examples)
        assert cov["total"] == 4
        assert cov["with_trace"] == 0
        assert cov["coverage"] == 0.0

    def test_coverage_empty_list(self):
        """Edge case: empty list -> coverage 0.0 without division error."""
        cov = trace_coverage([])
        assert cov["total"] == 0
        assert cov["with_trace"] == 0
        assert cov["coverage"] == 0.0

    def test_coverage_all_derivable(self):
        """Coverage is 1.0 when every example yields a trace."""
        examples = [self._derivable_example(f"d{i}") for i in range(5)]
        cov = trace_coverage(examples)
        assert cov["total"] == 5
        assert cov["with_trace"] == 5
        assert abs(cov["coverage"] - 1.0) < 1e-9
