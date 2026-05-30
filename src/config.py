"""Single source of truth for models, datasets, hyperparameters, and paths.

Pure-Python and dependency-light on purpose: importing this module must NOT
require torch/transformers, so the pure-logic modules (metrics, data
serialization, traces) can import it without the heavy ML stack. `get_device`
imports torch lazily, only when actually called.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEEDS = [13, 42]  # every condition runs twice; report mean +/- std

# --------------------------------------------------------------------------- #
# Models (HuggingFace ids)
# --------------------------------------------------------------------------- #
FLAN_SMALL = "google/flan-t5-small"  # 80M  -- SMOKE / fast local checks only
FLAN_BASE = "google/flan-t5-base"    # 250M -- fine-tuned + prompted
FLAN_LARGE = "google/flan-t5-large"  # 780M -- prompting only (large OOMs when fine-tuned on T4)

# Conditions that fine-tune use base only; prompting conditions use both.
PROMPT_MODELS = [FLAN_BASE, FLAN_LARGE]
FINETUNE_MODEL = FLAN_BASE

# --------------------------------------------------------------------------- #
# Datasets (HuggingFace ids) -- verified by real download (see scripts/smoke_local.py)
#
# NOTE: `datasets` >= 4.0 removed support for script-based datasets, so the
# canonical `wikitablequestions` and `tab_fact` ids no longer load. We use
# parquet-native mirrors with identical content:
#   * WTQ     -> lighteval/wikitablequestions  (inline header+rows tables;
#                single 18,486-example pool -> we make a fixed SEEDED disjoint
#                train/eval partition, so train and eval never overlap).
#   * TabFact -> target-benchmark/{tabfact-queries, tabfact-corpus}
#                (queries carry statement + True/False; corpus carries the
#                 table as a list-of-lists, joined on table_id).
# --------------------------------------------------------------------------- #
WTQ_ID = "lighteval/wikitablequestions"          # Pasupat & Liang 2015, primary
TABFACT_QUERIES_ID = "target-benchmark/tabfact-queries"  # Chen et al. 2020, OOD test
TABFACT_CORPUS_ID = "target-benchmark/tabfact-corpus"

# --------------------------------------------------------------------------- #
# Evaluation sizing (seeded caps keep every condition < 2 hr on a Colab T4)
# --------------------------------------------------------------------------- #
EVAL_N = 1000        # baseline / fine-tune / TabFact eval slice
EVAL_N_COT = 500     # CoT prompting is slower (long prompts) -> smaller slice
EVAL_SEED = 13       # seed used to sample the eval slice (held fixed across conditions)

# --------------------------------------------------------------------------- #
# Fine-tuning hyperparameters (proposal section 5)
# --------------------------------------------------------------------------- #
LEARNING_RATE = 3e-4
TRAIN_BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 4          # effective batch size = 8 * 4 = 32
EPOCHS = 3
TRAIN_N = 8000               # cap on fine-tune training examples (keeps a run < 2 hr on a T4)
N_SHOTS = 6                  # CoT exemplars prepended per prompt (proposal: 6-8)
MAX_SOURCE_LEN = 512          # serialized table + question
MAX_TARGET_LEN = 128          # answer, or reasoning chain + answer
GEN_MAX_NEW_TOKENS = 128

# Decoding: greedy, temperature 0 (deterministic) for all reported runs.
GREEDY = True

# --------------------------------------------------------------------------- #
# SMOKE mode -- tiny real end-to-end run to prove the pipeline executes.
# Used both as the first Colab sanity cell and for local MPS verification.
# --------------------------------------------------------------------------- #
SMOKE_MODEL = FLAN_SMALL
SMOKE_TRAIN_N = 16
SMOKE_EVAL_N = 8
SMOKE_EPOCHS = 1
SMOKE_MAX_STEPS = 2

# --------------------------------------------------------------------------- #
# Error-type labels (proposal section 6)
# --------------------------------------------------------------------------- #
ERROR_TYPES = ["lookup", "aggregation", "multi_hop", "correct"]

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"


def get_device() -> str:
    """Return the best available torch device: cuda (Colab) > mps (Apple) > cpu.

    Imports torch lazily so this module stays importable without the ML stack.
    """
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        # Let unsupported MPS ops fall back to CPU instead of crashing.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return "mps"
    return "cpu"


def eval_n_for(condition: str, smoke: bool = False) -> int:
    """Eval slice size for a condition. CoT gets the smaller cap."""
    if smoke:
        return SMOKE_EVAL_N
    return EVAL_N_COT if "cot" in condition.lower() else EVAL_N
