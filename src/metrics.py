"""Evaluation metrics for ECS 111 Final Project.

Pure Python (stdlib + scipy + scikit-learn only). Safe to import without
torch/transformers/datasets.
"""

from __future__ import annotations

import re
import string
from collections import Counter

from scipy.stats import binomtest
from sklearn.metrics import cohen_kappa_score


# --------------------------------------------------------------------------- #
# Text normalization
# --------------------------------------------------------------------------- #

def normalize_answer(s: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace, strip.

    Faithful to proposal: lowercasing and punctuation removal.
    Articles are intentionally kept (not removed).
    """
    s = s.lower()
    # Remove all punctuation characters
    s = s.translate(str.maketrans("", "", string.punctuation))
    # Collapse any run of whitespace to a single space and strip ends
    s = " ".join(s.split())
    return s


# --------------------------------------------------------------------------- #
# Exact Match
# --------------------------------------------------------------------------- #

def exact_match(pred: str, gold: str) -> bool:
    """Return True iff the normalized prediction equals the normalized gold."""
    return normalize_answer(pred) == normalize_answer(gold)


def exact_match_score(preds: list[str], golds: list[str]) -> float:
    """Mean exact_match over two equal-length lists (returns 0.0 to 1.0).

    Raises ValueError if the lists have different lengths.
    """
    if len(preds) != len(golds):
        raise ValueError(
            f"preds and golds must have the same length, "
            f"got {len(preds)} vs {len(golds)}"
        )
    if not preds:
        return 0.0
    return sum(exact_match(p, g) for p, g in zip(preds, golds)) / len(preds)


# --------------------------------------------------------------------------- #
# Token F1 (SQuAD-style)
# --------------------------------------------------------------------------- #

def token_f1(pred: str, gold: str) -> float:
    """SQuAD-style token-level F1 over normalized whitespace tokens.

    If either side has zero tokens:
      - both empty -> 1.0
      - one empty  -> 0.0

    common = size of the multiset intersection;
    precision = common / len(pred_tokens);
    recall    = common / len(gold_tokens);
    F1        = 2 * P * R / (P + R), or 0.0 when common == 0.
    """
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    common = sum((pred_counter & gold_counter).values())

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def token_f1_score(preds: list[str], golds: list[str]) -> float:
    """Mean token_f1 over two equal-length lists (returns 0.0 to 1.0).

    Raises ValueError if the lists have different lengths.
    """
    if len(preds) != len(golds):
        raise ValueError(
            f"preds and golds must have the same length, "
            f"got {len(preds)} vs {len(golds)}"
        )
    if not preds:
        return 0.0
    return sum(token_f1(p, g) for p, g in zip(preds, golds)) / len(preds)


# --------------------------------------------------------------------------- #
# TabFact label mapping
# --------------------------------------------------------------------------- #

_TRUE_KEYWORDS = ("true", "yes", "entail", "supported", "correct")
_FALSE_KEYWORDS = ("false", "no", "refut", "contradict", "incorrect")

# Pre-compiled patterns anchored at word boundaries (start of keyword only).
# Using \b at the start prevents partial suffix matches (e.g. "correct" inside
# "incorrect"), while omitting \b at the end allows prefix/stem matches
# (e.g. "entail" matches "entailed", "refut" matches "refuted").
_TRUE_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(kw) for kw in _TRUE_KEYWORDS) + r")"
)
_FALSE_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(kw) for kw in _FALSE_KEYWORDS) + r")"
)


def map_tabfact_label(text: str) -> str | None:
    """Map free-form model output to "true", "false", or None.

    Searches the lowercased text for keyword signals using word boundaries:
      "true" side : true / yes / entail / supported / correct
      "false" side: false / no / refut / contradict / incorrect

    If both sides match (ambiguous) or neither matches, returns None.
    Word-boundary matching prevents "correct" from matching inside "incorrect",
    or "no" from matching inside "know".
    """
    lowered = text.lower()
    has_true = bool(_TRUE_PATTERN.search(lowered))
    has_false = bool(_FALSE_PATTERN.search(lowered))

    if has_true and not has_false:
        return "true"
    if has_false and not has_true:
        return "false"
    # ambiguous (both) or neither -> unmappable
    return None


# --------------------------------------------------------------------------- #
# Classification accuracy (TabFact)
# --------------------------------------------------------------------------- #

def classification_accuracy(preds: list[str], golds: list[str]) -> float:
    """Accuracy for TabFact binary classification.

    golds must already be "true" or "false".
    Each pred is mapped via map_tabfact_label; unmappable preds count as wrong.

    Raises ValueError if the lists have different lengths.
    """
    if len(preds) != len(golds):
        raise ValueError(
            f"preds and golds must have the same length, "
            f"got {len(preds)} vs {len(golds)}"
        )
    if not preds:
        return 0.0
    correct = sum(
        map_tabfact_label(p) == g for p, g in zip(preds, golds)
    )
    return correct / len(preds)


# --------------------------------------------------------------------------- #
# McNemar test (paired system comparison)
# --------------------------------------------------------------------------- #

def mcnemar_test(
    correct_a: list[bool], correct_b: list[bool]
) -> dict:
    """Paired McNemar significance test between two systems.

    b = count(a correct, b wrong)
    c = count(a wrong,   b correct)

    Uses an exact binomial test on the discordant pairs:
      scipy.stats.binomtest(min(b, c), b + c, 0.5)

    If b + c == 0 (no discordant pairs), pvalue is 1.0.

    Returns {"b": b, "c": c, "n_discordant": b+c, "pvalue": float}.

    Raises ValueError if the lists have different lengths.
    """
    if len(correct_a) != len(correct_b):
        raise ValueError(
            f"correct_a and correct_b must have the same length, "
            f"got {len(correct_a)} vs {len(correct_b)}"
        )

    b = sum(1 for a, bb in zip(correct_a, correct_b) if a and not bb)
    c = sum(1 for a, bb in zip(correct_a, correct_b) if not a and bb)
    n_discordant = b + c

    if n_discordant == 0:
        pvalue = 1.0
    else:
        result = binomtest(min(b, c), n_discordant, 0.5)
        pvalue = float(result.pvalue)

    return {"b": b, "c": c, "n_discordant": n_discordant, "pvalue": pvalue}


# --------------------------------------------------------------------------- #
# Cohen's kappa
# --------------------------------------------------------------------------- #

def cohen_kappa(rater_a: list[int], rater_b: list[int]) -> float:
    """Inter-rater agreement using sklearn.metrics.cohen_kappa_score.

    Ratings are expected to be 0, 1, or 2.
    Returns kappa in [-1, 1]; perfect agreement -> 1.0.
    """
    return float(cohen_kappa_score(rater_a, rater_b))
