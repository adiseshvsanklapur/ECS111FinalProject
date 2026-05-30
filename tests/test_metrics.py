"""Tests for src/metrics.py.

All expected values are hand-computed and documented inline.
"""

import sys
import os

# Make the src package importable when running pytest from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.metrics import (
    normalize_answer,
    exact_match,
    exact_match_score,
    token_f1,
    token_f1_score,
    map_tabfact_label,
    classification_accuracy,
    mcnemar_test,
    cohen_kappa,
)


# --------------------------------------------------------------------------- #
# normalize_answer
# --------------------------------------------------------------------------- #

class TestNormalizeAnswer:
    def test_lowercase_and_punctuation(self):
        # "New York!" -> lowercase -> "new york!" -> strip punct -> "new york"
        assert normalize_answer("New York!") == "new york"

    def test_extra_whitespace_collapsed(self):
        assert normalize_answer("  hello   world  ") == "hello world"

    def test_punctuation_only(self):
        assert normalize_answer("!!!") == ""

    def test_articles_kept(self):
        # Articles should NOT be removed per spec
        assert normalize_answer("The answer is a cat") == "the answer is a cat"

    def test_mixed_punctuation(self):
        assert normalize_answer("U.S.A.") == "usa"

    def test_empty_string(self):
        assert normalize_answer("") == ""


# --------------------------------------------------------------------------- #
# exact_match
# --------------------------------------------------------------------------- #

class TestExactMatch:
    def test_true_case_insensitive(self):
        assert exact_match("New York", "new york") is True

    def test_true_with_punctuation(self):
        assert exact_match("New York!", "new york") is True

    def test_false_different_answers(self):
        assert exact_match("Paris", "London") is False

    def test_false_partial(self):
        assert exact_match("the cat sat", "the cat") is False

    def test_true_identical(self):
        assert exact_match("hello", "hello") is True


# --------------------------------------------------------------------------- #
# exact_match_score
# --------------------------------------------------------------------------- #

class TestExactMatchScore:
    def test_all_correct(self):
        preds = ["New York!", "Paris", "London"]
        golds = ["new york", "paris", "london"]
        assert exact_match_score(preds, golds) == pytest.approx(1.0)

    def test_none_correct(self):
        preds = ["Paris", "Berlin"]
        golds = ["London", "Rome"]
        assert exact_match_score(preds, golds) == pytest.approx(0.0)

    def test_partial(self):
        # 2 of 3 correct -> 2/3
        preds = ["New York!", "Paris", "Berlin"]
        golds = ["new york", "paris", "london"]
        assert exact_match_score(preds, golds) == pytest.approx(2 / 3)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            exact_match_score(["a", "b"], ["a"])


# --------------------------------------------------------------------------- #
# token_f1
# --------------------------------------------------------------------------- #

class TestTokenF1:
    def test_identical(self):
        # All tokens match: common=2, P=1, R=1, F1=1
        assert token_f1("the cat", "the cat") == pytest.approx(1.0)

    def test_disjoint(self):
        # No tokens in common -> F1=0
        assert token_f1("dog", "cat") == pytest.approx(0.0)

    def test_partial_overlap(self):
        # pred: ["the","cat","sat"] gold: ["the","cat"]
        # common = 2, P = 2/3, R = 2/2 = 1.0
        # F1 = 2 * (2/3) * 1.0 / (2/3 + 1.0) = (4/3) / (5/3) = 4/5 = 0.8
        assert token_f1("the cat sat", "the cat") == pytest.approx(0.8)

    def test_both_empty(self):
        assert token_f1("", "") == pytest.approx(1.0)

    def test_pred_empty(self):
        assert token_f1("", "cat") == pytest.approx(0.0)

    def test_gold_empty(self):
        assert token_f1("cat", "") == pytest.approx(0.0)

    def test_repeated_tokens(self):
        # pred: ["cat","cat"] gold: ["cat"]
        # common (multiset intersection) = min(2,1) = 1
        # P = 1/2, R = 1/1 = 1.0
        # F1 = 2 * 0.5 * 1.0 / (0.5 + 1.0) = 1.0 / 1.5 = 2/3
        assert token_f1("cat cat", "cat") == pytest.approx(2 / 3)


# --------------------------------------------------------------------------- #
# token_f1_score
# --------------------------------------------------------------------------- #

class TestTokenF1Score:
    def test_mean_of_three(self):
        # token_f1("the cat","the cat")=1.0, ("dog","cat")=0.0, ("the cat sat","the cat")=0.8
        # mean = (1.0 + 0.0 + 0.8) / 3 = 1.8/3 = 0.6
        preds = ["the cat", "dog", "the cat sat"]
        golds = ["the cat", "cat", "the cat"]
        assert token_f1_score(preds, golds) == pytest.approx(0.6)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            token_f1_score(["a"], ["a", "b"])


# --------------------------------------------------------------------------- #
# map_tabfact_label
# --------------------------------------------------------------------------- #

class TestMapTabfactLabel:
    def test_true_keyword(self):
        assert map_tabfact_label("true") == "true"

    def test_false_keyword(self):
        assert map_tabfact_label("It is false") == "false"

    def test_yes_maps_to_true(self):
        assert map_tabfact_label("yes, this is correct") == "true"

    def test_no_maps_to_false(self):
        assert map_tabfact_label("No, that is wrong") == "false"

    def test_entail_maps_to_true(self):
        assert map_tabfact_label("this statement is entailed") == "true"

    def test_refut_maps_to_false(self):
        assert map_tabfact_label("The claim is refuted") == "false"

    def test_contradict_maps_to_false(self):
        assert map_tabfact_label("This contradicts the table") == "false"

    def test_supported_maps_to_true(self):
        assert map_tabfact_label("The claim is supported") == "true"

    def test_incorrect_maps_to_false(self):
        assert map_tabfact_label("That is incorrect") == "false"

    def test_garbage_returns_none(self):
        # "I have no idea" contains "no" -> would map to "false"; use a
        # string with zero keyword signals to test the None path.
        assert map_tabfact_label("the data is ambiguous") is None

    def test_ambiguous_both_true_and_false_returns_none(self):
        # Contains both "true" and "false" -> ambiguous -> None
        assert map_tabfact_label("true or false?") is None

    def test_empty_returns_none(self):
        assert map_tabfact_label("") is None


# --------------------------------------------------------------------------- #
# classification_accuracy
# --------------------------------------------------------------------------- #

class TestClassificationAccuracy:
    def test_basic(self):
        # preds: "It is true" -> "true" (correct), "false" -> "false" (correct),
        #        "garbage" -> None (wrong, gold="true")
        # 2/3 correct
        preds = ["It is true", "false", "garbage"]
        golds = ["true", "false", "true"]
        assert classification_accuracy(preds, golds) == pytest.approx(2 / 3)

    def test_all_correct(self):
        preds = ["yes", "no"]
        golds = ["true", "false"]
        assert classification_accuracy(preds, golds) == pytest.approx(1.0)

    def test_unmappable_counts_as_wrong(self):
        preds = ["I dunno"]
        golds = ["true"]
        assert classification_accuracy(preds, golds) == pytest.approx(0.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            classification_accuracy(["yes", "no"], ["true"])


# --------------------------------------------------------------------------- #
# mcnemar_test
# --------------------------------------------------------------------------- #

class TestMcnemarTest:
    def test_known_b_c(self):
        # Construct: a correct / b wrong on indices 0 and 1 -> b=2
        #            a wrong  / b correct on index 2        -> c=1
        # b=2, c=1, n_discordant=3
        # binomtest(min(2,1)=1, 3, 0.5) -> compute expected pvalue
        from scipy.stats import binomtest as _binomtest
        expected_pvalue = float(_binomtest(1, 3, 0.5).pvalue)

        correct_a = [True,  True,  False, True]
        correct_b = [False, False, True,  True]
        result = mcnemar_test(correct_a, correct_b)
        assert result["b"] == 2
        assert result["c"] == 1
        assert result["n_discordant"] == 3
        assert result["pvalue"] == pytest.approx(expected_pvalue)

    def test_equal_discordant_pvalue_one(self):
        # b=2, c=2 -> binomtest(2, 4, 0.5).pvalue == 1.0
        correct_a = [True,  True,  False, False]
        correct_b = [False, False, True,  True]
        result = mcnemar_test(correct_a, correct_b)
        assert result["b"] == 2
        assert result["c"] == 2
        assert result["n_discordant"] == 4
        assert result["pvalue"] == pytest.approx(1.0)

    def test_no_discordant_pvalue_one(self):
        # No discordant pairs -> pvalue = 1.0
        correct_a = [True, False]
        correct_b = [True, False]
        result = mcnemar_test(correct_a, correct_b)
        assert result["b"] == 0
        assert result["c"] == 0
        assert result["n_discordant"] == 0
        assert result["pvalue"] == pytest.approx(1.0)

    def test_asymmetric_b_1_c_3(self):
        # b=1, c=3 -> min(1,3)=1, n=4 -> pvalue=0.625
        correct_a = [True,  False, False, False]
        correct_b = [False, True,  True,  True]
        result = mcnemar_test(correct_a, correct_b)
        assert result["b"] == 1
        assert result["c"] == 3
        assert result["pvalue"] == pytest.approx(0.625)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            mcnemar_test([True, False], [True])


# --------------------------------------------------------------------------- #
# cohen_kappa
# --------------------------------------------------------------------------- #

class TestCohenKappa:
    def test_perfect_agreement(self):
        rater_a = [0, 1, 2, 0, 1, 2]
        rater_b = [0, 1, 2, 0, 1, 2]
        assert cohen_kappa(rater_a, rater_b) == pytest.approx(1.0)

    def test_partial_agreement_less_than_one(self):
        rater_a = [0, 1, 2, 0]
        rater_b = [0, 2, 2, 1]
        kappa = cohen_kappa(rater_a, rater_b)
        assert kappa < 1.0

    def test_returns_float(self):
        result = cohen_kappa([0, 1], [0, 1])
        assert isinstance(result, float)
