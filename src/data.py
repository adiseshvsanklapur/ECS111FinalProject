"""Dataset loading + table serialization.

Unified example schema used everywhere downstream:

    {
        "id":       str,
        "task":     "wtq" | "tabfact",
        "question": str,                       # WTQ question  OR  TabFact statement
        "table":    {"header": [str, ...],
                     "rows":   [[str, ...], ...]},
        "answer":   str,                       # WTQ: gold answer text
                                               # TabFact: "true" | "false"
        "answers":  [str, ...],                # WTQ raw gold list (>=1); TabFact: [answer]
    }

`serialize_table` and the TabFact text parser are pure Python (no `datasets`
needed) so they are unit-testable offline. The HF loaders import `datasets`
lazily inside the function, so importing this module never requires the ML stack.
"""

from __future__ import annotations

import ast
import random
from typing import Any

from . import config


# --------------------------------------------------------------------------- #
# Table serialization  (proposal: "headers first, then each row in sequence")
# --------------------------------------------------------------------------- #
def serialize_table(table: dict[str, Any]) -> str:
    """Flatten a {header, rows} table to a compact, model-friendly string.

    Format: header line then one line per row, cells joined by " | ".
    """
    header = [str(h) for h in table["header"]]
    lines = [" | ".join(header)]
    for row in table["rows"]:
        lines.append(" | ".join(str(c) for c in row))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# TabFact corpus table parser  (pure)
# --------------------------------------------------------------------------- #
def parse_corpus_table(table: Any) -> dict[str, list]:
    """Convert a TabFact-corpus table (list-of-lists, header first) to {header, rows}.

    Accepts either a real list or its string repr (defensive).
    """
    if isinstance(table, str):
        table = ast.literal_eval(table)
    if not table:
        return {"header": [], "rows": []}
    header = [str(c) for c in table[0]]
    rows = [[str(c) for c in r] for r in table[1:]]
    return {"header": header, "rows": rows}


# --------------------------------------------------------------------------- #
# Seeded sampling  (pure)
# --------------------------------------------------------------------------- #
def sample_indices(n_total: int, n_take: int, seed: int) -> list[int]:
    """Deterministic sample of up to n_take indices from range(n_total)."""
    if n_take >= n_total:
        return list(range(n_total))
    rng = random.Random(seed)
    return sorted(rng.sample(range(n_total), n_take))


def disjoint_train_eval_indices(
    total: int, eval_n: int, eval_seed: int, train_n: int, train_seed: int
) -> tuple[list[int], list[int]]:
    """Fixed eval slice first, then a train slice drawn from the remainder.

    Guarantees train and eval indices never overlap, regardless of seeds. Used
    to carve a clean train/eval partition out of the single WTQ pool (the
    parquet mirror ships one split; see config note).
    """
    eval_idxs = sample_indices(total, eval_n, eval_seed)
    eval_set = set(eval_idxs)
    pool = [i for i in range(total) if i not in eval_set]
    if train_n >= len(pool):
        train_idxs = pool
    else:
        rng = random.Random(train_seed)
        train_idxs = sorted(rng.sample(pool, train_n))
    return train_idxs, eval_idxs


# --------------------------------------------------------------------------- #
# Normalizers (raw HF row -> unified schema)
# --------------------------------------------------------------------------- #
def _normalize_wtq(row: dict[str, Any], idx: int) -> dict[str, Any]:
    answers = [str(a) for a in row.get("answers", [])]
    return {
        "id": str(row.get("id", f"wtq-{idx}")),
        "task": "wtq",
        "question": row["question"],
        "table": {
            "header": [str(h) for h in row["table"]["header"]],
            "rows": [[str(c) for c in r] for r in row["table"]["rows"]],
        },
        "answer": ", ".join(answers),
        "answers": answers,
    }


def _normalize_tabfact(query_row: dict[str, Any], table: dict[str, list], idx: int) -> dict[str, Any]:
    answer = "true" if str(query_row["answer"]).strip().lower() == "true" else "false"
    return {
        "id": str(query_row.get("query_id", f"tabfact-{idx}")),
        "task": "tabfact",
        "question": query_row["query"],   # the statement to verify
        "table": table,
        "answer": answer,
        "answers": [answer],
    }


# --------------------------------------------------------------------------- #
# HF loaders  (import `datasets` lazily)
# --------------------------------------------------------------------------- #
def _wtq_pool():
    """The single WTQ pool from the parquet mirror (inline tables)."""
    from datasets import load_dataset

    return load_dataset(config.WTQ_ID, split="test")


def load_wtq_eval(n: int | None = config.EVAL_N, seed: int = config.EVAL_SEED) -> list[dict]:
    """Seeded WTQ evaluation slice (None = full pool)."""
    ds = _wtq_pool()
    idxs = range(len(ds)) if n is None else sample_indices(len(ds), n, seed)
    return [_normalize_wtq(ds[int(i)], int(i)) for i in idxs]


def load_wtq_train(
    n: int,
    seed: int,
    eval_n: int = config.EVAL_N,
    eval_seed: int = config.EVAL_SEED,
) -> list[dict]:
    """Seeded WTQ training slice, guaranteed disjoint from the eval slice."""
    ds = _wtq_pool()
    train_idxs, _ = disjoint_train_eval_indices(len(ds), eval_n, eval_seed, n, seed)
    return [_normalize_wtq(ds[int(i)], int(i)) for i in train_idxs]


def load_tabfact(split: str = "test", n: int | None = config.EVAL_N, seed: int = config.EVAL_SEED) -> list[dict]:
    """Load TabFact (OOD test), joining queries to corpus tables on table_id."""
    from datasets import load_dataset

    queries = load_dataset(config.TABFACT_QUERIES_ID, split=split)
    corpus = load_dataset(config.TABFACT_CORPUS_ID, split=split)
    table_by_id = {row["table_id"]: parse_corpus_table(row["table"]) for row in corpus}

    idxs = range(len(queries)) if n is None else sample_indices(len(queries), n, seed)
    out = []
    for i in idxs:
        q = queries[int(i)]
        table = table_by_id.get(q["table_id"])
        if table is None:
            continue
        out.append(_normalize_tabfact(q, table, int(i)))
    return out
