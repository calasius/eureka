"""The difficulty ladder — one generator per level (Section 6).

Each generator builds random event sequences, then labels every case with the
level's planted rule via ``rules.make_labeler`` (single source of truth). It
returns ``(cases, ground_truth)`` where ``ground_truth`` is pure JSON recording
the planted_rule_id + params + a human description + label distribution.

Design discipline that keeps ground truth *exact*:
  * Filler/neutral event types are disjoint from the trigger/marker/confirm
    types a rule reacts to, so neutrals can never accidentally fire a rule.
  * Positive and negative scenarios are planted deliberately (including hard
    negatives — e.g. trigger-without-confirm, or confirm-too-far) so a level
    distinguishes the real structure from cheap proxies like mere presence.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

from .model import Event, event
from .rules import NO_RULE, make_labeler

# Neutral filler — never referenced by any planted rule.
NEUTRAL = ["evt_P", "evt_Q", "evt_R", "evt_S"]
# Extra distractors for the noise level.
DISTRACTOR = ["evt_N1", "evt_N2", "evt_N3", "evt_N4"]

CHANNELS = ["web", "phone", "branch"]


def _open(rng: random.Random, **attrs: Any) -> Event:
    return event(0, "open", attrs)


def _fillers(rng: random.Random, n: int, vocab: List[str] = NEUTRAL) -> List[Event]:
    return [event(0, rng.choice(vocab)) for _ in range(n)]


def _label_all(cases: List[Tuple[str, List[Event]]], rule_id: str,
               params: Dict[str, Any], level: str) -> List[Dict[str, Any]]:
    """Apply the planted labeler to every (case_id, events) pair."""
    labeler = make_labeler(rule_id, params)
    out = []
    for cid, events in cases:
        out.append({"case_id": cid, "events": events,
                    "outcome": labeler(events), "level": level})
    return out


def _distribution(cases: List[Dict[str, Any]]) -> Dict[str, int]:
    dist: Dict[str, int] = {}
    for c in cases:
        dist[c["outcome"]] = dist.get(c["outcome"], 0) + 1
    return dict(sorted(dist.items()))


def _ground_truth(level: str, rule_id: str, params: Dict[str, Any],
                  description: str, cases: List[Dict[str, Any]], seed: int) -> Dict[str, Any]:
    return {
        "level": level,
        "planted_rule_id": rule_id,
        "params": params,
        "description": description,
        "seed": seed,
        "n_cases": len(cases),
        "label_distribution": _distribution(cases),
        "scorer_note": (
            "Outcome == make_labeler(planted_rule_id, params)(case['events']). "
            "A recovered rule is correct iff logically equivalent to this."
        ),
    }


# --------------------------------------------------------------------------- #
# Level 0 — Sanity: marker presence -> reject                                  #
# --------------------------------------------------------------------------- #
def gen_level0(n: int, seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(seed)
    params = {"marker": "evt_X", "hit": "reject", "miss": "approve"}
    raw = []
    for i in range(n):
        body = _fillers(rng, rng.randint(2, 6))
        if rng.random() < 0.5:
            body.insert(rng.randint(0, len(body)), event(0, "evt_X"))
        events = [_open(rng, channel=rng.choice(CHANNELS))] + body + [event(0, "close")]
        raw.append((f"c_{i:05d}", events))
    cases = _label_all(raw, "R0_marker_presence", params, "level0")
    gt = _ground_truth("level0", "R0_marker_presence", params,
                       "If evt_X appears anywhere -> reject, else approve.",
                       cases, seed)
    return cases, gt


# --------------------------------------------------------------------------- #
# Level 1 — Recombination: typed successor within lag, over several pairs       #
# (the Galois test: 3 concrete instances are shadows of one general schema)     #
# --------------------------------------------------------------------------- #
_L1_PAIRS = [["T1", "F1"], ["T2", "F2"], ["T3", "F3"]]
_L1_LAG = 2


def _plant_successor(rng: random.Random, body: List[Event], pairs, lag) -> None:
    """Insert one of several scenarios that exercise the successor-within-lag rule."""
    a, b = rng.choice(pairs)
    scenario = rng.choice(["pos", "pos", "trigger_only", "confirm_only", "too_far", "none"])
    if scenario == "pos":
        i = rng.randint(0, len(body))
        gap = rng.randint(1, lag)
        body.insert(i, event(0, a))
        body.insert(min(i + gap, len(body)), event(0, b))
    elif scenario == "trigger_only":
        body.insert(rng.randint(0, len(body)), event(0, a))
    elif scenario == "confirm_only":
        body.insert(rng.randint(0, len(body)), event(0, b))
    elif scenario == "too_far":
        i = rng.randint(0, len(body))
        body.insert(i, event(0, a))
        body.insert(min(i + lag + rng.randint(1, 3), len(body)), event(0, b))
    # "none": leave neutral


def gen_level1(n: int, seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(seed)
    params = {"pairs": _L1_PAIRS, "lag": _L1_LAG, "hit": "reject", "miss": "approve"}
    raw = []
    for i in range(n):
        body = _fillers(rng, rng.randint(2, 6))
        _plant_successor(rng, body, _L1_PAIRS, _L1_LAG)
        events = [_open(rng, channel=rng.choice(CHANNELS))] + body + [event(0, "close")]
        raw.append((f"c_{i:05d}", events))
    cases = _label_all(raw, "R1_typed_successor", params, "level1")
    gt = _ground_truth(
        "level1", "R1_typed_successor", params,
        "For some k, if Tk is followed by Fk within lag=2 -> reject. The three "
        "concrete pairs (T1->F1, T2->F2, T3->F3) instantiate one general schema.",
        cases, seed)
    return cases, gt


# --------------------------------------------------------------------------- #
# Level 2 — Noise: Level-1 rule + random distractor events injected             #
# --------------------------------------------------------------------------- #
def gen_level2(n: int, seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(seed)
    params = {"pairs": _L1_PAIRS, "lag": _L1_LAG, "hit": "reject", "miss": "approve"}
    raw = []
    for i in range(n):
        body = _fillers(rng, rng.randint(3, 8))
        # Sprinkle distractors first so they pad the trace but never fire the rule.
        for _ in range(rng.randint(2, 6)):
            body.insert(rng.randint(0, len(body)), event(0, rng.choice(DISTRACTOR)))
        # Plant the real signal with gap=1 so noise rarely breaks a true positive.
        a, b = rng.choice(_L1_PAIRS)
        scenario = rng.choice(["pos", "pos", "trigger_only", "confirm_only", "none"])
        if scenario == "pos":
            j = rng.randint(0, len(body))
            body.insert(j, event(0, a))
            body.insert(j + 1, event(0, b))
        elif scenario == "trigger_only":
            body.insert(rng.randint(0, len(body)), event(0, a))
        elif scenario == "confirm_only":
            body.insert(rng.randint(0, len(body)), event(0, b))
        events = [_open(rng, channel=rng.choice(CHANNELS))] + body + [event(0, "close")]
        raw.append((f"c_{i:05d}", events))
    cases = _label_all(raw, "R2_typed_successor_noisy", params, "level2")
    gt = _ground_truth(
        "level2", "R2_typed_successor_noisy", params,
        "Same successor-within-lag rule as Level 1, but each case is padded with "
        "random distractor events (evt_N*) that never affect the outcome.",
        cases, seed)
    return cases, gt


# --------------------------------------------------------------------------- #
# Level 3 — Multimodal: two rules on disjoint subsets, split by channel          #
# (this is what justifies the multi-agent layer)                                #
# --------------------------------------------------------------------------- #
def gen_level3(n: int, seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(seed)
    params = {
        "by": "channel",
        "default": "approve",
        "branches": {
            "web": {"rule_id": "R0_marker_presence",
                    "params": {"marker": "evt_X", "hit": "reject", "miss": "approve"}},
            "phone": {"rule_id": "R1_typed_successor",
                      "params": {"pairs": [["T1", "F1"]], "lag": 2,
                                 "hit": "review", "miss": "approve"}},
        },
    }
    raw = []
    for i in range(n):
        channel = rng.choice(CHANNELS)
        body = _fillers(rng, rng.randint(2, 6))
        if channel == "web" and rng.random() < 0.5:
            body.insert(rng.randint(0, len(body)), event(0, "evt_X"))
        elif channel == "phone":
            _plant_successor(rng, body, [["T1", "F1"]], 2)
        events = [_open(rng, channel=channel)] + body + [event(0, "close")]
        raw.append((f"c_{i:05d}", events))
    cases = _label_all(raw, "R3_channel_split", params, "level3")
    gt = _ground_truth(
        "level3", "R3_channel_split", params,
        "Two rules govern disjoint subsets by channel: web -> marker(evt_X)=reject; "
        "phone -> successor(T1->F1 within lag 2)=review; branch -> approve. A single "
        "global rule cannot fit both subsets — rival theories must be separated.",
        cases, seed)
    return cases, gt


# --------------------------------------------------------------------------- #
# Level 4 — Compositional: outcome depends on sub-rules (2-3 level hierarchy)    #
# --------------------------------------------------------------------------- #
def gen_level4(n: int, seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(seed)
    params = {
        "predicates": {
            "P1": {"pred": "typed_successor", "params": {"pairs": [["T1", "F1"]], "lag": 2}},
            "P2": {"pred": "marker", "params": {"marker": "evt_Y"}},
            "P3": {"pred": "count_at_least", "params": {"type": "evt_C", "n": 2}},
        },
        "table": [
            {"cond": {"P1": True, "P2": True}, "outcome": "reject"},
            {"cond": {"P1": True, "P2": False}, "outcome": "review"},
            {"cond": {"P1": False, "P3": True}, "outcome": "review"},
        ],
        "default": "approve",
    }
    raw = []
    for i in range(n):
        body = _fillers(rng, rng.randint(2, 5))
        if rng.random() < 0.5:                       # P1: plant T1->F1 (gap 1)
            j = rng.randint(0, len(body))
            body.insert(j, event(0, "T1"))
            body.insert(j + 1, event(0, "F1"))
        if rng.random() < 0.5:                       # P2: marker evt_Y
            body.insert(rng.randint(0, len(body)), event(0, "evt_Y"))
        for _ in range(rng.choice([0, 0, 2, 3])):    # P3: >=2 evt_C
            body.insert(rng.randint(0, len(body)), event(0, "evt_C"))
        events = [_open(rng, channel=rng.choice(CHANNELS))] + body + [event(0, "close")]
        raw.append((f"c_{i:05d}", events))
    cases = _label_all(raw, "R4_hierarchical", params, "level4")
    gt = _ground_truth(
        "level4", "R4_hierarchical", params,
        "Outcome depends on sub-rules: P1=successor(T1->F1), P2=marker(evt_Y), "
        "P3=count(evt_C)>=2. (P1&P2)->reject; (P1&!P2)->review; (!P1&P3)->review; "
        "else approve. Recovery requires composing abstractions of abstractions.",
        cases, seed)
    return cases, gt


# --------------------------------------------------------------------------- #
# Level 5 — Analogy: same abstract structure in two vocabularies                 #
# --------------------------------------------------------------------------- #
def gen_level5(n: int, seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(seed)
    vocab_triggers = {"alpha": "evt_A", "beta": "ping"}
    params = {
        "by": "vocab",
        "k": 3,
        "classes": ["approve", "reject", "review"],
        "vocab_triggers": vocab_triggers,
    }
    raw = []
    for i in range(n):
        vocab = rng.choice(["alpha", "beta"])
        trig = vocab_triggers[vocab]
        body = _fillers(rng, rng.randint(2, 5))
        for _ in range(rng.randint(0, 5)):           # vary count -> vary residue
            body.insert(rng.randint(0, len(body)), event(0, trig))
        events = [_open(rng, channel=rng.choice(CHANNELS), vocab=vocab)] + body + [event(0, "close")]
        raw.append((f"c_{i:05d}", events))
    cases = _label_all(raw, "R5_counter_mod_k", params, "level5")
    gt = _ground_truth(
        "level5", "R5_counter_mod_k", params,
        "Outcome = class[count(trigger) mod 3]. The SAME counter-mod-k structure is "
        "expressed in two vocabularies (alpha:evt_A, beta:ping). Tests analogical "
        "transfer of structure across vocabularies — expect frailty (SOTA limit).",
        cases, seed)
    return cases, gt


# --------------------------------------------------------------------------- #
# NEG — Negative control: no rule, i.i.d. outcomes independent of the events     #
# --------------------------------------------------------------------------- #
def gen_neg(n: int, seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(seed)
    # Include rule-looking event types as bait, but outcomes ignore them entirely.
    bait = NEUTRAL + ["evt_X", "T1", "F1", "evt_Y", "evt_C"]
    cases = []
    for i in range(n):
        body = _fillers(rng, rng.randint(2, 8), vocab=bait)
        events = [_open(rng, channel=rng.choice(CHANNELS))] + body + [event(0, "close")]
        # Outcome drawn independently of events -> no recoverable rule exists.
        outcome = rng.choice(["approve", "reject"])
        cases.append({"case_id": f"c_{i:05d}", "events": events,
                      "outcome": outcome, "level": "neg"})
    gt = {
        "level": "neg",
        "planted_rule_id": NO_RULE,
        "params": {"distribution": "Bernoulli(0.5) over {approve, reject}"},
        "description": ("No rule. Outcomes are i.i.d. coin flips independent of the "
                        "events (despite rule-looking bait events). Any 'confident' "
                        "rule the system reports here is a false positive."),
        "seed": seed,
        "n_cases": len(cases),
        "label_distribution": _distribution(cases),
        "scorer_note": "Ground truth = NO RULE. Measure the false-positive rate here.",
    }
    return cases, gt


GENERATORS = {
    "level0": gen_level0,
    "level1": gen_level1,
    "level2": gen_level2,
    "level3": gen_level3,
    "level4": gen_level4,
    "level5": gen_level5,
    "neg": gen_neg,
}

ALL_LEVELS = list(GENERATORS.keys())
