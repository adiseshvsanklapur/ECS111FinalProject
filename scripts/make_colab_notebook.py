"""Generate a single self-contained Colab notebook that runs the full study.

The notebook needs NO git clone: it writes the src/ modules and the runner into
the Colab filesystem from %%writefile cells, installs the deps, and runs
scripts/run_all_local.py at full scale. A SHARD variable lets five teammates split
the work across free Colab accounts so it finishes in ~30 min each instead of one
~2-hour serial run. Because the code is read from the local files at generation
time, the notebook always matches this repo without needing a push.

Run:
    python scripts/make_colab_notebook.py
Writes ECS111_Colab_FullRun.ipynb at the repo root.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "ECS111_Colab_FullRun.ipynb"

SRC_FILES = [
    "__init__.py", "config.py", "data.py", "prompts.py", "cot_exemplars.py",
    "trace_templates.py", "metrics.py", "trainer.py", "evaluate.py",
    "analysis.py", "report_fill.py",
]
SCRIPT_FILES = ["run_all_local.py"]

INTRO = """# ECS111 Final Project — Full Run on Colab (T4 GPU), self-contained

**No git clone needed.** This notebook carries all the project code inside it.

1. Runtime -> Change runtime type -> **T4 GPU**.
2. (Optional) set `SHARD` in the next code cell to split the work with teammates.
3. Runtime -> **Run all**. If Colab disconnects, just Run all again — the runner
   resumes and skips finished work.

`SHARD` options (one teammate each for a ~30-min run, or leave `all` for one
~2-hour run): `all` · `baseline` · `cot_plain` · `cot_structured` ·
`finetune_answers` · `finetune_traces`.

When it finishes, download `ecs111_results.zip` and send it back, or paste the
contents of `results/summary_table.md`.
"""

GPU_CELL = '''import torch

if torch.cuda.is_available():
    print("CUDA available:", True, "|", torch.cuda.get_device_name(0))
else:
    print("!!! No GPU detected.")
    print("!!! Runtime -> Change runtime type -> T4 GPU, then Run all again.")
'''

SHARD_CELL = '''SHARD = "all"  # all | baseline | cot_plain | cot_structured | finetune_answers | finetune_traces
import os
for d in ("src", "scripts", "results", "checkpoints"):
    os.makedirs(d, exist_ok=True)
print("SHARD =", SHARD)
'''

INSTALL_CELL = (
    "%pip install -q torch transformers datasets sentencepiece "
    "scipy scikit-learn pandas matplotlib tabulate"
)

RUN_CELL = '''import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# {SHARD} is expanded by IPython from the Python variable set above.
!python scripts/run_all_local.py --scale full --shard {SHARD}
'''

PRINT_CELL = '''import os

path = "results/summary_table.md"
if os.path.exists(path):
    print(open(path).read())
else:
    print("results/summary_table.md not found yet — the run may be partial.")
'''

DOWNLOAD_CELL = '''import shutil

shutil.make_archive("ecs111_results", "zip", "results")
try:
    from google.colab import files
    files.download("ecs111_results.zip")
except Exception as e:
    print("download unavailable (not on Colab?):", e)
print("Send ecs111_results.zip back, or paste results/summary_table.md.")
'''


def _code_cell(cid: str, source: str) -> dict:
    return {"cell_type": "code", "id": cid, "metadata": {},
            "execution_count": None, "outputs": [], "source": source}


def _md_cell(cid: str, source: str) -> dict:
    return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": source}


def _writefile_cell(cid: str, relpath: str) -> dict:
    body = (ROOT / relpath).read_text()
    return _code_cell(cid, f"%%writefile {relpath}\n{body}")


def build_notebook() -> dict:
    cells = [
        _md_cell("intro", INTRO),
        _code_cell("gpu", GPU_CELL),
        _code_cell("shard", SHARD_CELL),
    ]
    for i, name in enumerate(SRC_FILES):
        cells.append(_writefile_cell(f"src_{i}", f"src/{name}"))
    for i, name in enumerate(SCRIPT_FILES):
        cells.append(_writefile_cell(f"script_{i}", f"scripts/{name}"))
    cells += [
        _code_cell("install", INSTALL_CELL),
        _code_cell("run", RUN_CELL),
        _code_cell("print", PRINT_CELL),
        _code_cell("download", DOWNLOAD_CELL),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    nb = build_notebook()
    OUT.write_text(json.dumps(nb, indent=1))
    print(f"Wrote {OUT} with {len(nb['cells'])} cells "
          f"({len(SRC_FILES)} src + {len(SCRIPT_FILES)} script writefile cells).")


if __name__ == "__main__":
    main()
