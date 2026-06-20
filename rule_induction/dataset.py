"""Dataset assembly: holdout split + on-disk layout (Sections 6 & 7).

Mandatory controls baked in here:
  * Holdout per level — train/test split; the inducer sees only train, the
    arbiter scores on test. A rule that only works on train is memorization.
  * Multiple seeds — generate each level over many seeds and report the
    distribution, never the best case.
  * Planted ground truth is written separately from the case-files, so the
    inducer-facing traces never leak ``planted_rule_id``.

On-disk layout::

    <out>/<level>/seed_<s>/
        train.jsonl        # one case-file per line (no planted_rule_id)
        test.jsonl
        ground_truth.json  # planted rule + params + label distribution
    <out>/manifest.json    # index of everything generated
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from .levels import ALL_LEVELS, GENERATORS


def split_train_test(cases: List[Dict[str, Any]], ratio: float, seed: int
                     ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Deterministic shuffle + split. ``ratio`` is the train fraction."""
    rng = random.Random(seed * 7919 + 1)   # decorrelated from the generator seed
    idx = list(range(len(cases)))
    rng.shuffle(idx)
    cut = int(round(len(cases) * ratio))
    train_idx, test_idx = set(idx[:cut]), set(idx[cut:])
    train, test = [], []
    for i, c in enumerate(cases):
        c = dict(c)
        c["split"] = "train" if i in train_idx else "test"
        (train if i in train_idx else test).append(c)
    return train, test


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_level(level: str, n: int, seed: int):
    if level not in GENERATORS:
        raise KeyError(f"unknown level {level!r}; known: {ALL_LEVELS}")
    return GENERATORS[level](n, seed)


def write_level(out_dir: str, level: str, *, n: int, seed: int, ratio: float
                ) -> Dict[str, Any]:
    """Generate one (level, seed), split, and write the three files."""
    cases, ground_truth = generate_level(level, n, seed)
    train, test = split_train_test(cases, ratio, seed)

    seed_dir = os.path.join(out_dir, level, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    _write_jsonl(os.path.join(seed_dir, "train.jsonl"), train)
    _write_jsonl(os.path.join(seed_dir, "test.jsonl"), test)

    ground_truth = dict(ground_truth)
    ground_truth.update({"n_train": len(train), "n_test": len(test), "train_ratio": ratio})
    with open(os.path.join(seed_dir, "ground_truth.json"), "w", encoding="utf-8") as fh:
        json.dump(ground_truth, fh, indent=2, ensure_ascii=False)

    return {
        "level": level, "seed": seed, "dir": seed_dir,
        "planted_rule_id": ground_truth["planted_rule_id"],
        "n_train": len(train), "n_test": len(test),
        "label_distribution": ground_truth["label_distribution"],
    }


def build(out_dir: str, *, levels: Optional[List[str]] = None, n: int = 200,
          seeds: int = 3, ratio: float = 0.7, seed_start: int = 0) -> Dict[str, Any]:
    """Generate a full benchmark: ``levels`` x ``seeds``, write a manifest."""
    levels = levels or ALL_LEVELS
    os.makedirs(out_dir, exist_ok=True)
    entries = []
    for level in levels:
        for s in range(seed_start, seed_start + seeds):
            entries.append(write_level(out_dir, level, n=n, seed=s, ratio=ratio))
    manifest = {
        "out_dir": out_dir, "levels": levels, "n_per_seed": n,
        "seeds": seeds, "seed_start": seed_start, "train_ratio": ratio,
        "entries": entries,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    return manifest
