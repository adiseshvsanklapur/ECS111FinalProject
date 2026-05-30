"""Seq2seq fine-tuning + generation for Flan-T5.

A small, explicit training loop (rather than the framework Trainer) so behaviour
is identical across transformers versions and on cuda / mps / cpu. Used by
Conditions B (answers-only) and C (reasoning traces). `train` takes already-built
source/target strings, keeping this module independent of prompt/data specifics.
"""

from __future__ import annotations

import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from . import config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _Seq2SeqDataset(Dataset):
    def __init__(self, sources: list[str], targets: list[str]):
        assert len(sources) == len(targets)
        self.sources = sources
        self.targets = targets

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, i: int) -> dict:
        return {"source": self.sources[i], "target": self.targets[i]}


def _make_collate(tokenizer):
    def collate(batch: list[dict]) -> dict:
        sources = [b["source"] for b in batch]
        targets = [b["target"] for b in batch]
        model_inputs = tokenizer(
            sources,
            max_length=config.MAX_SOURCE_LEN,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        labels = tokenizer(
            text_target=targets,
            max_length=config.MAX_TARGET_LEN,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )["input_ids"]
        # Ignore pad tokens in the loss.
        labels[labels == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        return model_inputs

    return collate


def load_model_and_tokenizer(model_id: str, device: str | None = None):
    device = device or config.get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
    return model, tokenizer, device


def train(
    model_id: str,
    sources: list[str],
    targets: list[str],
    seed: int,
    smoke: bool = False,
    device: str | None = None,
):
    """Fine-tune `model_id` on (sources -> targets). Returns (model, tokenizer, device).

    smoke=True shrinks to flan-t5-small + a couple of steps for a real but fast
    end-to-end check (used locally on MPS and as the first Colab sanity cell).
    """
    set_seed(seed)
    if smoke:
        model_id = config.SMOKE_MODEL
        sources = sources[: config.SMOKE_TRAIN_N]
        targets = targets[: config.SMOKE_TRAIN_N]

    model, tokenizer, device = load_model_and_tokenizer(model_id, device)
    loader = DataLoader(
        _Seq2SeqDataset(sources, targets),
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=_make_collate(tokenizer),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    epochs = config.SMOKE_EPOCHS if smoke else config.EPOCHS
    max_steps = config.SMOKE_MAX_STEPS if smoke else None

    model.train()
    step = 0
    optimizer.zero_grad()
    for _epoch in range(epochs):
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / config.GRAD_ACCUM_STEPS
            loss.backward()
            if (i + 1) % config.GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                if max_steps is not None and step >= max_steps:
                    break
        if max_steps is not None and step >= max_steps:
            break
    # Flush any remaining accumulated gradients.
    optimizer.step()
    optimizer.zero_grad()
    return model, tokenizer, device


@torch.no_grad()
def generate(
    model,
    tokenizer,
    sources: list[str],
    device: str | None = None,
    batch_size: int = 16,
) -> tuple[list[str], float]:
    """Greedy-decode predictions for `sources`. Returns (predictions, seconds_per_example)."""
    device = device or config.get_device()
    model.eval()
    preds: list[str] = []
    start = time.perf_counter()
    for i in range(0, len(sources), batch_size):
        chunk = sources[i : i + batch_size]
        inputs = tokenizer(
            chunk,
            max_length=config.MAX_SOURCE_LEN,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)
        out = model.generate(
            **inputs,
            max_new_tokens=config.GEN_MAX_NEW_TOKENS,
            do_sample=False,  # greedy, deterministic
            num_beams=1,
        )
        preds.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
    elapsed = time.perf_counter() - start
    per_example = elapsed / max(len(sources), 1)
    return preds, per_example


def peak_memory_mb(device: str) -> float | None:
    """Peak allocated memory in MB for the run, if the backend reports it."""
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 1024**2
    if device == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        return torch.mps.current_allocated_memory() / 1024**2
    return None
