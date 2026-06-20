"""Synthetic REST-API access-log generator (a real-world `--dataset` for the agent).

Each *case* is one client **session**: an ordered sequence of HTTP requests, in
the same event format the rest of the system speaks (``{case_id, events:[{t,
type, attrs}], outcome}``). The ``outcome`` is the API gateway's decision for the
session — ``allow`` / ``throttle`` / ``block`` — produced by a single hidden
**policy** (the planted rule).

The point: this is not toy ladder data, it's the shape of logs you actually have
(method, path, status, latency, client plan/region). Feed it to the investigator
skill via ``--dataset`` and let the agent recover the gateway policy, verified by
the deterministic arbiter on a holdout.

Hidden policy ``gateway-v1`` (priority order — first match wins):

  1. **block**    — the session performs a *privileged write* (``POST /admin/*``
                    or ``DELETE /users/*``) with **no successful login**
                    (``POST /login`` → 200) earlier in the session. Order- and
                    status-sensitive: a privilege-escalation guard.
  2. **throttle** — not blocked, but >= 6 requests in the session OR >= 2 responses
                    with status 429. A rate-abuse guard.
  3. **allow**    — otherwise.

The block clause is exactly the kind of structure a symbolic search struggles
with (precedence + an attribute test + absence) and the investigator recovers
cleanly with a ``code`` hypothesis — the honest showcase for this layer.

On-disk layout (flat, what ``load_dataset`` / ``present --dataset`` read)::

    <out>/train.jsonl
    <out>/test.jsonl
    <out>/ground_truth.json     # the policy + reference solution (omit with --no-ground-truth)

CLI::

    python -m rule_induction.apilog --out data/apilog --n 800 --seed 0
    python -m rule_induction.present --dataset data/apilog          # train-only view
    # then, in a Claude session:  /investigator --dataset data/apilog
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from typing import Any, Dict, List, Tuple

from .dataset import split_train_test
from .model import Event, event, retime

Case = Dict[str, Any]

# --------------------------------------------------------------------------- #
# Vocabulary                                                                   #
# --------------------------------------------------------------------------- #
PLANS = ["free", "free", "free", "pro", "pro", "enterprise"]
REGIONS = ["us-east", "eu-west", "ap-south"]
AGENTS = ["browser", "browser", "mobile", "curl", "bot"]
RESOURCES = ["orders", "invoices", "carts", "tickets", "profiles"]

# Privileged writes that require a prior successful login.
PRIV_WRITE = {"POST_admin", "DELETE_user"}


def _path(rng: random.Random, kind: str) -> str:
    res = rng.choice(RESOURCES)
    rid = rng.randint(1000, 9999)
    return {
        "GET_health": "/health",
        "GET_static": f"/static/{rng.choice(['app.js', 'logo.png', 'style.css'])}",
        "POST_login": "/auth/login",
        "GET_api": f"/api/v1/{res}/{rid}",
        "POST_api": f"/api/v1/{res}",
        "PUT_api": f"/api/v1/{res}/{rid}",
        "POST_admin": f"/admin/{res}",
        "DELETE_user": f"/users/{rid}",
    }[kind]


def _req(rng: random.Random, kind: str, status: int) -> Event:
    """One HTTP request as an event: type = METHOD_class, attrs = the log fields."""
    method = kind.split("_", 1)[0]
    ms = rng.choice([8, 12, 20, 35, 60, 90, 140, 220])
    if status >= 500:
        ms += rng.randint(200, 900)
    return event(0, kind, {
        "method": method,
        "path": _path(rng, kind),
        "status": status,
        "ms": ms,
        "bytes": rng.choice([0, 128, 512, 2048, 8192]),
    })


def _login(rng: random.Random, ok: bool) -> Event:
    return _req(rng, "POST_login", 200 if ok else 401)


def _read(rng: random.Random) -> Event:
    kind = rng.choice(["GET_api", "GET_api", "GET_health", "GET_static"])
    return _req(rng, kind, rng.choices([200, 304, 500], weights=[88, 8, 4])[0])


def _write_api(rng: random.Random) -> Event:
    kind = rng.choice(["POST_api", "PUT_api"])
    return _req(rng, kind, rng.choices([201, 200, 400, 500], weights=[70, 15, 10, 5])[0])


def _priv_write(rng: random.Random) -> Event:
    kind = rng.choice(list(PRIV_WRITE))
    return _req(rng, kind, rng.choices([200, 204, 403], weights=[70, 20, 10])[0])


# --------------------------------------------------------------------------- #
# The hidden policy (single source of truth for the label)                     #
# --------------------------------------------------------------------------- #
def policy_gateway_v1(events: List[Event]) -> str:
    """Gateway decision for a session. See module docstring for the spec."""
    seen_login_ok = False
    blocked = False
    n_req = n_429 = 0
    for e in events:
        t = e["type"]
        if t in ("open", "close"):
            continue
        n_req += 1
        st = (e.get("attrs") or {}).get("status")
        if st == 429:
            n_429 += 1
        if t == "POST_login" and st == 200:
            seen_login_ok = True
        if t in PRIV_WRITE and not seen_login_ok:
            blocked = True
    if blocked:
        return "block"
    if n_req >= 6 or n_429 >= 2:
        return "throttle"
    return "allow"


# A compact reference solution recorded in ground_truth.json (the *answer* file,
# never shown to the investigator) — what a correct `code` hypothesis converges to.
REFERENCE_SOLUTION = '''\
def predict(events):
    PRIV = {"POST_admin", "DELETE_user"}
    login_ok = blocked = False
    n = n429 = 0
    for e in events:
        t = e["type"]
        if t in ("open", "close"):
            continue
        n += 1
        st = (e.get("attrs") or {}).get("status")
        if st == 429:
            n429 += 1
        if t == "POST_login" and st == 200:
            login_ok = True
        if t in PRIV and not login_ok:
            blocked = True
    if blocked:
        return "block"
    if n >= 6 or n429 >= 2:
        return "throttle"
    return "allow"
'''


# --------------------------------------------------------------------------- #
# Session synthesis                                                            #
# --------------------------------------------------------------------------- #
SCENARIOS = [
    "benign",            # short, clean reads -> allow
    "benign",
    "burst",             # many requests / 429s -> throttle
    "burst",
    "escalation",        # priv write, no prior login -> block
    "authorized_admin",  # login_ok THEN priv write -> allow (hard negative)
    "failed_login_admin",# login 401 THEN priv write -> block (status-sensitive)
    "auth_then_burst",   # login_ok + many -> throttle
]


def _session_body(rng: random.Random, scenario: str, noise: float) -> List[Event]:
    body: List[Event] = []

    def sprinkle(lo: int, hi: int) -> None:
        for _ in range(rng.randint(lo, hi)):
            body.insert(rng.randint(0, len(body)), _read(rng))

    if scenario == "benign":
        sprinkle(1, 3)
        if rng.random() < 0.5:
            body.insert(0, _login(rng, ok=True))
    elif scenario == "burst":
        sprinkle(5, 11)
        # a real burst trips the limiter: later calls start returning 429.
        for _ in range(rng.randint(0, 3)):
            body.append(_req(rng, "GET_api", 429))
    elif scenario == "escalation":
        sprinkle(0, 2)
        body.append(_priv_write(rng))            # no login_ok anywhere -> block
    elif scenario == "authorized_admin":
        body.insert(0, _login(rng, ok=True))
        sprinkle(0, 2)
        body.append(_priv_write(rng))            # login precedes write -> allow
    elif scenario == "failed_login_admin":
        body.insert(0, _login(rng, ok=False))    # 401, not a successful login
        sprinkle(0, 2)
        body.append(_priv_write(rng))            # -> block
    elif scenario == "auth_then_burst":
        body.insert(0, _login(rng, ok=True))
        sprinkle(5, 10)

    # Optional benign noise: scattered reads that never change the decision.
    if noise > 0:
        for _ in range(int(rng.random() < noise) + rng.randint(0, 2 if rng.random() < noise else 0)):
            body.insert(rng.randint(0, len(body)), _read(rng))
    return body


def _make_case(rng: random.Random, i: int, noise: float) -> Case:
    scenario = rng.choice(SCENARIOS)
    plan = rng.choice(PLANS)
    opening = event(0, "open", {
        "client_id": f"cli_{rng.randint(0, 999):03d}",
        "plan": plan,
        "region": rng.choice(REGIONS),
        "agent": rng.choice(AGENTS),
    })
    body = _session_body(rng, scenario, noise)
    events = retime([opening] + body + [event(0, "close")])
    return {
        "case_id": f"s_{i:05d}",
        "events": events,
        "outcome": policy_gateway_v1(events),
        "level": "apilog",
    }


def generate(n: int, seed: int, *, noise: float = 0.3) -> List[Case]:
    rng = random.Random(seed)
    return [_make_case(rng, i, noise) for i in range(n)]


# --------------------------------------------------------------------------- #
# Write the flat dataset                                                       #
# --------------------------------------------------------------------------- #
def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _distribution(cases: List[Case]) -> Dict[str, int]:
    return dict(sorted(Counter(c["outcome"] for c in cases).items()))


def write_dataset(out_dir: str, *, n: int, seed: int, ratio: float, noise: float,
                  with_ground_truth: bool) -> Dict[str, Any]:
    cases = generate(n, seed, noise=noise)
    train, test = split_train_test(cases, ratio, seed)
    os.makedirs(out_dir, exist_ok=True)
    _write_jsonl(os.path.join(out_dir, "train.jsonl"), train)
    _write_jsonl(os.path.join(out_dir, "test.jsonl"), test)

    if with_ground_truth:
        gt = {
            "dataset": "apilog",
            "planted_policy": "gateway-v1",
            "description": (
                "Per-session API gateway decision. block = a privileged write "
                "(POST_admin or DELETE_user) with no prior successful login "
                "(POST_login status 200); throttle = >=6 requests or >=2 status-429 "
                "responses; else allow. Block is order- and status-sensitive."
            ),
            "outcome_classes": ["allow", "throttle", "block"],
            "label_distribution": _distribution(cases),
            "n_train": len(train),
            "n_test": len(test),
            "train_ratio": ratio,
            "noise": noise,
            "reference_solution": REFERENCE_SOLUTION,
            "scorer_note": (
                "Outcome == policy_gateway_v1(case['events']). A recovered "
                "hypothesis is correct iff it agrees on the holdout (test_accuracy)."
            ),
        }
        with open(os.path.join(out_dir, "ground_truth.json"), "w", encoding="utf-8") as fh:
            json.dump(gt, fh, indent=2, ensure_ascii=False)

    return {
        "out_dir": out_dir, "n": n, "seed": seed,
        "n_train": len(train), "n_test": len(test),
        "label_distribution": _distribution(cases),
        "ground_truth": with_ground_truth,
    }


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def _main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Generate synthetic REST-API session logs as a --dataset.")
    p.add_argument("--out", default="data/apilog", help="output folder [default: data/apilog]")
    p.add_argument("--n", type=int, default=800, help="number of sessions")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ratio", type=float, default=0.7, help="train fraction")
    p.add_argument("--noise", type=float, default=0.3, help="benign-noise intensity (0 = none)")
    p.add_argument("--no-ground-truth", action="store_true",
                   help="write only train/test (simulate logs whose rule you don't know)")
    args = p.parse_args(argv)

    info = write_dataset(args.out, n=args.n, seed=args.seed, ratio=args.ratio,
                         noise=args.noise, with_ground_truth=not args.no_ground_truth)
    print(f"wrote {info['n_train']} train + {info['n_test']} test sessions to {info['out_dir']}/")
    print(f"label distribution: {info['label_distribution']}")
    print(f"ground_truth.json : {'written' if info['ground_truth'] else 'omitted'}")
    print("\nnext:")
    print(f"  python -m rule_induction.present --dataset {info['out_dir']}")
    print(f"  # then in a Claude session:  /investigator --dataset {info['out_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
