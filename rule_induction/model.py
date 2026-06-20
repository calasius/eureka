"""Event / case-file data model (Section 6 of the plan).

A *case* is an ordered sequence of *events*. Each event has a discrete ``type``
and an optional ``attrs`` bag (the "group of bits"). The ``outcome`` is decided
by the level's planted rule. Everything is plain JSON-serialisable dicts so the
traces are trivially portable and diffable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

Event = Dict[str, Any]   # {"t": int, "type": str, "attrs": {...}}
Case = Dict[str, Any]    # see new_case()


def event(t: int, type: str, attrs: Optional[Dict[str, Any]] = None) -> Event:
    return {"t": t, "type": type, "attrs": attrs or {}}


def retime(events: List[Event]) -> List[Event]:
    """Re-assign monotonically increasing ``t`` after insertions/deletions.

    Generators build sequences by inserting into a list; ``t`` is the logical
    step index, so we normalise it once the order is final.
    """
    out = []
    for i, e in enumerate(events):
        e = dict(e)
        e["t"] = i
        out.append(e)
    return out


def new_case(
    case_id: str,
    events: List[Event],
    outcome: str,
    level: str,
    *,
    planted_rule_id: Optional[str] = None,
    split: Optional[str] = None,
) -> Case:
    """Build a case-file dict.

    ``planted_rule_id`` is the ground truth. By default the dataset writer keeps
    it OUT of the inducer-facing files and only records it in ground_truth.json
    (Section 6: "Never shown to the inducer — only to the scorer."). Pass it here
    only when you explicitly want the leak (e.g. debugging).
    """
    case: Case = {
        "case_id": case_id,
        "events": retime(events),
        "outcome": outcome,
        "level": level,
    }
    if split is not None:
        case["split"] = split
    if planted_rule_id is not None:
        case["planted_rule_id"] = planted_rule_id
    return case


def open_attr(events: List[Event], key: str, default: Any = None) -> Any:
    """Read ``attrs[key]`` from the first ``open`` event (else first event)."""
    for e in events:
        if e["type"] == "open":
            return e.get("attrs", {}).get(key, default)
    if events:
        return events[0].get("attrs", {}).get(key, default)
    return default
