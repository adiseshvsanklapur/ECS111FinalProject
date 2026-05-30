"""Resume a hung results run: run ONLY the remaining steps, then aggregate.

A full `run_all_local.py --scale quick` completed all six prompting conditions
(their result JSONs are in results/) but hung during fine-tuning when the Mac
slept and wedged the MPS GPU. We do NOT want to recompute the prompting
conditions. This thin runner reuses run_all_local's functions to run only:

  1. fine-tune (answers only)
  2. fine-tune (with reasoning traces)
  3. generalization (TabFact)
  4. aggregate  (reads ALL results/*.json, so the existing 6 prompting JSONs
                 get merged in automatically)

It never runs baseline or CoT.

Usage:
    python scripts/resume_finetune_local.py --scale quick
    python scripts/resume_finetune_local.py --device cpu   # MPS fallback
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))  # repo root (for `src`)
sys.path.insert(0, str(_SCRIPT_DIR))  # scripts/ (for sibling run_all_local)

from src import config  # noqa: E402
import run_all_local  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", choices=["quick", "full"], default="quick")
    ap.add_argument(
        "--device",
        default=None,
        help="Force a device (e.g. 'cpu'). Default: config.get_device().",
    )
    args = ap.parse_args()

    device = args.device if args.device is not None else config.get_device()
    s = run_all_local.get_settings(args.scale)
    run_all_local.log(f"RESUME device={device} scale={args.scale} settings={s}")
    t0 = time.time()

    run_all_local.run_finetune(s, device, "finetune_answers", use_traces=False)
    run_all_local.run_finetune(s, device, "finetune_traces", use_traces=True)
    run_all_local.run_generalization(s, device)
    run_all_local.aggregate()

    run_all_local.log(f"ALL DONE in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
