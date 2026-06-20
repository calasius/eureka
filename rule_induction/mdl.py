"""MDL scorer — the honesty anchor's measuring stick (Sections 2 & 4).

A rule is rewarded for **compression on holdout**, not accuracy. We use a
*prequential plug-in code*: the coding distribution is learned on TRAIN, and the
description length is paid on TEST. This cleanly enforces both hard defenses:

* **Holdout.** The bits are counted on test data the inducer never saw. A rule
  that only "works" on train buys nothing here.
* **Program length.** ``bits_saved`` subtracts the hypothesis's own description
  length, so a function that memorizes the trace (enormous program) loses to any
  short rule explaining the same data.

Definitions (all in bits)::

    L_null      = sum over test of  -log2 P_train_marginal(true_outcome)
    L_data|H    = sum over test of  -log2 P_train(true | predicted = H(case))
    L_H         = description length of the hypothesis
    bits_saved  = L_null - L_H - L_data|H

``P_train(true | predicted)`` is a smoothed confusion table built on train: a
*good* rule makes that distribution peaked (cheap), so it pays off on test. A
memorizer's train confusion is perfectly diagonal, but on test its predictions
are wrong, and the near-zero smoothed off-diagonal mass makes every holdout
mismatch expensive — exactly the overconfidence MDL is meant to punish.
"""

from __future__ import annotations

import io
import json
import math
import tokenize
from typing import Any, Dict, List, Optional, Sequence

LAPLACE = 0.5          # additive smoothing for the plug-in codes
BITS_PER_ATOM = 6.0    # description cost per leaf/structure node of a rule spec
BITS_PER_TOKEN = 8.0   # description cost per code token


# --------------------------------------------------------------------------- #
# Description length of a hypothesis (L_H)                                      #
# --------------------------------------------------------------------------- #
def _count_atoms(obj: Any) -> int:
    """Count the information-bearing *leaves* of a spec.

    Dict keys are the fixed grammar (shared between sender and receiver), so they
    are free; only the chosen values cost bits. This charges a rule for its
    content, not for the scaffolding of whatever syntax expresses it — so a
    legible decision list is not penalised relative to an equivalent compact rule.
    """
    if isinstance(obj, dict):
        return sum(_count_atoms(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_count_atoms(x) for x in obj)
    return 1


def program_bits_spec(spec: Any) -> float:
    """Description length of a *symbolic* rule (params dict)."""
    return _count_atoms(spec) * BITS_PER_ATOM


def program_bits_code(source: str) -> float:
    """Description length of a *code* hypothesis = meaningful token count * cost.

    Token-counting (not raw bytes / zlib) keeps short rules cheap and makes a
    memorizer — whose source embeds the whole training set — genuinely huge.
    """
    skip = {tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT,
            tokenize.COMMENT, tokenize.ENCODING, tokenize.ENDMARKER}
    n = 0
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type not in skip and tok.string != "":
                n += 1
    except (tokenize.TokenError, IndentationError):
        # Untokenizable source: fall back to a byte estimate (still large).
        n = max(1, len(source) // 4)
    return n * BITS_PER_TOKEN


# --------------------------------------------------------------------------- #
# Plug-in codes                                                                #
# --------------------------------------------------------------------------- #
def _smoothed(counts: Dict[Any, float], classes: Sequence[Any], alpha: float) -> Dict[Any, float]:
    total = sum(counts.get(c, 0.0) for c in classes) + alpha * len(classes)
    return {c: (counts.get(c, 0.0) + alpha) / total for c in classes}


def _marginal(train_true: Sequence[Any], classes: Sequence[Any]) -> Dict[Any, float]:
    counts: Dict[Any, float] = {}
    for t in train_true:
        counts[t] = counts.get(t, 0.0) + 1.0
    return _smoothed(counts, classes, LAPLACE)


def _confusion(train_pred: Sequence[Any], train_true: Sequence[Any],
               classes: Sequence[Any]) -> Dict[Any, Dict[Any, float]]:
    """P_train(true | predicted), smoothed; rows for unseen predictions back off."""
    rows: Dict[Any, Dict[Any, float]] = {}
    raw: Dict[Any, Dict[Any, float]] = {}
    for p, t in zip(train_pred, train_true):
        raw.setdefault(p, {})[t] = raw.setdefault(p, {}).get(t, 0.0) + 1.0
    for p in classes:
        rows[p] = _smoothed(raw.get(p, {}), classes, LAPLACE)
    return rows


def _bits(p: float) -> float:
    return -math.log2(max(p, 1e-12))


# --------------------------------------------------------------------------- #
# The score                                                                    #
# --------------------------------------------------------------------------- #
def score(train_true: Sequence[Any], train_pred: Sequence[Any],
          test_true: Sequence[Any], test_pred: Sequence[Any],
          program_bits: float) -> Dict[str, Any]:
    """Compute the MDL verdict. Pure and deterministic."""
    classes = sorted(set(train_true) | set(test_true)
                     | set(train_pred) | set(test_pred), key=lambda x: (x is None, str(x)))
    marginal = _marginal(train_true, classes)
    confusion = _confusion(train_pred, train_true, classes)

    l_null = sum(_bits(marginal[t]) for t in test_true)
    l_data_given_h = sum(_bits(confusion[p][t]) for p, t in zip(test_pred, test_true))
    bits_saved = l_null - program_bits - l_data_given_h

    n_test = len(test_true)
    test_acc = (sum(1 for p, t in zip(test_pred, test_true) if p == t) / n_test
                if n_test else 0.0)
    n_train = len(train_true)
    train_acc = (sum(1 for p, t in zip(train_pred, train_true) if p == t) / n_train
                 if n_train else 0.0)

    return {
        "bits_saved": bits_saved,
        "l_null": l_null,
        "program_bits": program_bits,
        "l_data_given_h": l_data_given_h,
        "compression_ratio": (l_null / (program_bits + l_data_given_h))
        if (program_bits + l_data_given_h) > 0 else float("inf"),
        "test_accuracy": test_acc,
        "train_accuracy": train_acc,
        "n_test": n_test,
        "n_train": n_train,
        "classes": classes,
    }
