"""Protocol state-machine inference from traces (UNSUPERVISED).

Given traces of a protocol — each a sequence of messages, all *valid* — recover the
finite state machine that generated them, and render it as Markdown + a Mermaid
diagram. No labels, no fabricated negatives: the structure is learned from the
positive traces alone, exactly the unsupervised setting protocols call for.

The honesty thread from the rest of Eureka carries over as **compression**: a
prefix-tree that just memorises every trace has one state per distinct prefix
(huge); merging states that share the same *future behaviour* collapses that into a
small machine that also **generalises** — it accepts traces never seen (e.g. more
loop iterations). Fewer states = shorter description = the real structure.

Method (classic passive automata learning):
  1. Build the **prefix-tree acceptor** (PTA) over the traces.
  2. Merge states by **k-tails**: two states are equivalent if the set of
     continuations up to length k (plus an end marker for accepting states) is the
     same. This folds repetition into loops.
  3. **Determinise by merging**: if a merged state has two transitions on the same
     symbol, union the targets; repeat to a fixpoint -> a clean DFA.

Division of labour with the skill: this tool produces the *structure* (states,
transitions, stats, Mermaid) deterministically; the agent renames states with
semantic names (CLOSED / ESTABLISHED / …) and writes the prose.

CLI::

    python -m rule_induction.protocol --dataset data/tcp_handshake
    python -m rule_induction.protocol --demo                       # built-in TCP-ish demo
    python -m rule_induction.protocol --dataset DIR --k 2 --out sm.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

Trace = List[str]
END = "$"   # end-of-trace marker used inside k-tails so terminal states stay distinct


# --------------------------------------------------------------------------- #
# Loading                                                                       #
# --------------------------------------------------------------------------- #
def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def traces_from_cases(cases: List[Dict[str, Any]], *, attr: Optional[str] = None) -> List[Trace]:
    """Each case -> a sequence of message symbols. Optionally fold an attribute into
    the symbol (e.g. type + ':' + status), and drop structural open/close events."""
    skip = {"open", "close"}
    out: List[Trace] = []
    for c in cases:
        seq: Trace = []
        for e in c["events"]:
            t = e["type"]
            if t in skip:
                continue
            if attr is not None:
                v = (e.get("attrs") or {}).get(attr)
                if v is not None:
                    t = f"{t}:{v}"
            seq.append(t)
        if seq:
            out.append(seq)
    return out


def load_traces(dataset_dir: str, *, attr: Optional[str] = None) -> List[Trace]:
    cases: List[Dict[str, Any]] = []
    for name in ("train.jsonl", "test.jsonl"):
        path = os.path.join(dataset_dir, name)
        if os.path.exists(path):
            cases += _read_jsonl(path)
    return traces_from_cases(cases, attr=attr)


# --------------------------------------------------------------------------- #
# Prefix-tree acceptor                                                          #
# --------------------------------------------------------------------------- #
def build_pta(traces: List[Trace]) -> Tuple[Dict[int, Dict[str, int]], set]:
    children: Dict[int, Dict[str, int]] = {0: {}}
    accepting: set = set()
    nxt = 1
    for tr in traces:
        node = 0
        for sym in tr:
            if sym not in children[node]:
                children[node][sym] = nxt
                children[nxt] = {}
                nxt += 1
            node = children[node][sym]
        accepting.add(node)
    return children, accepting


def _k_tails(children: Dict[int, Dict[str, int]], accepting: set,
             node: int, k: int) -> frozenset:
    tails: set = set()

    def dfs(n: int, path: List[str]) -> None:
        if accepting and n in accepting:
            tails.add(tuple(path + [END]))
        if path:
            tails.add(tuple(path))
        if len(path) >= k:
            return
        for sym, c in children[n].items():
            dfs(c, path + [sym])

    dfs(node, [])
    return frozenset(tails)


# --------------------------------------------------------------------------- #
# Union-find                                                                     #
# --------------------------------------------------------------------------- #
class _UF:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)   # keep lower id as representative


# --------------------------------------------------------------------------- #
# Inference                                                                     #
# --------------------------------------------------------------------------- #
class StateMachine:
    def __init__(self, start, states, accepting, transitions):
        self.start = start                       # state id
        self.states = states                     # ordered list of state ids
        self.accepting = accepting               # set of state ids
        self.transitions = transitions           # {state: {symbol: target}}
        self.names: Dict[int, str] = {}          # filled by naming

    def alphabet(self) -> List[str]:
        syms = {s for edges in self.transitions.values() for s in edges}
        return sorted(syms)


def infer(traces: List[Trace], *, k: int = 2) -> Tuple[StateMachine, Dict[str, Any]]:
    children, accepting = build_pta(traces)
    nodes = list(children)

    # 1. initial partition by k-tails signature
    uf = _UF(nodes)
    by_sig: Dict[frozenset, int] = {}
    for n in nodes:
        sig = _k_tails(children, accepting, n, k)
        if sig in by_sig:
            uf.union(by_sig[sig], n)
        else:
            by_sig[sig] = n

    # 2. determinise by merging conflicting targets to a fixpoint
    while True:
        trans: Dict[int, Dict[str, set]] = defaultdict(lambda: defaultdict(set))
        for node, edges in children.items():
            r = uf.find(node)
            for sym, c in edges.items():
                trans[r][sym].add(uf.find(c))
        changed = False
        for r, edges in trans.items():
            for sym, targets in edges.items():
                if len(targets) > 1:
                    ts = sorted(targets)
                    for t in ts[1:]:
                        uf.union(ts[0], t)
                    changed = True
        if not changed:
            break

    # 3. build the quotient DFA
    det: Dict[int, Dict[str, int]] = defaultdict(dict)
    for node, edges in children.items():
        r = uf.find(node)
        for sym, c in edges.items():
            det[r][sym] = uf.find(c)
    start = uf.find(0)
    acc = {uf.find(n) for n in accepting}

    # order states by BFS from start for stable, readable ids
    order: List[int] = []
    seen = set()
    q = deque([start])
    while q:
        s = q.popleft()
        if s in seen:
            continue
        seen.add(s); order.append(s)
        for sym in sorted(det.get(s, {})):
            q.append(det[s][sym])
    for s in det:            # any unreachable leftovers (shouldn't happen) appended
        if s not in seen:
            order.append(s); seen.add(s)

    sm = StateMachine(start, order, acc, {s: dict(det.get(s, {})) for s in order})
    stats = {
        "n_traces": len(traces),
        "pta_states": len(children),
        "dfa_states": len(order),
        "alphabet": sm.alphabet(),
        "k": k,
    }
    stats["compression"] = _mdl(traces, sm, children, accepting)
    return sm, stats


# --------------------------------------------------------------------------- #
# Compression (the honesty flavour): does the machine beat the raw prefix tree? #
# --------------------------------------------------------------------------- #
def _machine_bits(n_states: int, n_transitions: int, alphabet: int) -> float:
    per = math.log2(max(1, n_states)) + math.log2(max(1, alphabet))
    return n_transitions * per


def _data_bits(traces: List[Trace], step_choices) -> float:
    bits = 0.0
    for tr in traces:
        bits += step_choices(tr)
    return bits


def _mdl(traces, sm: StateMachine, children, accepting) -> Dict[str, Any]:
    """Honest comparison, in Eureka's style: the structured machine vs. a structureless
    baseline that codes each symbol under its global frequency (a 0-order code, the
    `L_null` analogue). A good state machine makes most next-symbols forced (≈0 bits),
    so it crushes the baseline — *that* is the structure paying for itself."""
    alpha = len(sm.alphabet())

    # L_null: code every symbol (plus a stop) under the marginal symbol distribution.
    freq: Dict[str, float] = defaultdict(float)
    n_sym = 0.0
    for tr in traces:
        for s in tr:
            freq[s] += 1; n_sym += 1
        n_sym += 1                      # an explicit stop symbol per trace
    vocab = len(freq) + 1               # + stop
    def _code(count): return -math.log2((count + 0.5) / (n_sym + 0.5 * vocab))
    l_null = 0.0
    for tr in traces:
        l_null += sum(_code(freq[s]) for s in tr) + _code(len(traces))   # stop per trace

    # L_data | machine: at each state pay log2(#admissible next symbols [+stop]).
    def dfa_choices(tr: Trace) -> float:
        s = sm.start; b = 0.0
        for sym in tr:
            opts = len(sm.transitions.get(s, {})) + (1 if s in sm.accepting else 0)
            b += math.log2(max(1, opts))
            s = sm.transitions.get(s, {}).get(sym, s)
        opts = len(sm.transitions.get(s, {})) + (1 if s in sm.accepting else 0)
        b += math.log2(max(1, opts))     # the stop decision
        return b

    dfa_trans = sum(len(e) for e in sm.transitions.values())
    l_machine = _machine_bits(len(sm.states), dfa_trans, alpha)
    l_data = _data_bits(traces, dfa_choices)
    total = l_machine + l_data

    return {
        "l_null_bits": round(l_null, 1),
        "machine_bits": round(l_machine, 1),
        "data_given_machine_bits": round(l_data, 1),
        "bits_saved": round(l_null - total, 1),
        "state_reduction": f"{len(children)} -> {len(sm.states)}",
    }


# --------------------------------------------------------------------------- #
# Naming + generalisation check                                                 #
# --------------------------------------------------------------------------- #
def autoname(sm: StateMachine) -> None:
    """Default names S0..Sn (start=S0). The skill/agent overrides with semantics."""
    for i, s in enumerate(sm.states):
        sm.names[s] = f"S{i}"


def accepts(sm: StateMachine, trace: Trace) -> bool:
    s = sm.start
    for sym in trace:
        nxt = sm.transitions.get(s, {}).get(sym)
        if nxt is None:
            return False
        s = nxt
    return s in sm.accepting


def detect_loops(sm: StateMachine) -> List[Tuple[str, str]]:
    """Self-reachable states = repetition the machine generalised into a loop."""
    loops = []
    for s in sm.states:
        # BFS forward; if we can return to s, it's on a cycle
        seen = set(); q = deque(sm.transitions.get(s, {}).values())
        while q:
            n = q.popleft()
            if n == s:
                loops.append(sm.names[s]); break
            if n in seen:
                continue
            seen.add(n); q.extend(sm.transitions.get(n, {}).values())
    return loops


# --------------------------------------------------------------------------- #
# Rendering                                                                      #
# --------------------------------------------------------------------------- #
def to_mermaid(sm: StateMachine) -> str:
    lines = ["stateDiagram-v2", f"    [*] --> {sm.names[sm.start]}"]
    for s in sm.states:
        for sym in sorted(sm.transitions.get(s, {})):
            t = sm.transitions[s][sym]
            lines.append(f"    {sm.names[s]} --> {sm.names[t]}: {sym}")
        if s in sm.accepting:
            lines.append(f"    {sm.names[s]} --> [*]")
    return "\n".join(lines)


def to_markdown(sm: StateMachine, stats: Dict[str, Any]) -> str:
    comp = stats["compression"]
    loops = detect_loops(sm)
    md = []
    md.append("# Recovered protocol state machine\n")
    md.append(f"Inferred from **{stats['n_traces']} traces** (unsupervised — valid traces only), "
              f"k-tails merging with k={stats['k']}.\n")
    md.append("## Overview\n")
    md.append(f"- **States:** {stats['dfa_states']}  ·  **Alphabet:** "
              + ", ".join(f"`{a}`" for a in stats["alphabet"]))
    md.append(f"- **Start:** `{sm.names[sm.start]}`  ·  **Accepting (clean end):** "
              + ", ".join(f"`{sm.names[s]}`" for s in sm.states if s in sm.accepting))
    md.append(f"- **Loops (repetition generalised):** "
              + (", ".join(f"`{l}`" for l in loops) if loops else "none"))
    md.append(f"- **Compression:** merged the {comp['state_reduction']}-state prefix tree; "
              f"the machine describes the traces in {comp['data_given_machine_bits']} bits "
              f"vs. {comp['l_null_bits']} with no structure "
              f"(**{comp['bits_saved']:+} saved** — structure pays for itself).\n")
    md.append("## State diagram\n")
    md.append("```mermaid")
    md.append(to_mermaid(sm))
    md.append("```\n")
    md.append("## Transition table\n")
    md.append("| State | On message | Goes to |")
    md.append("|---|---|---|")
    for s in sm.states:
        edges = sm.transitions.get(s, {})
        if not edges:
            md.append(f"| `{sm.names[s]}`{' ✅' if s in sm.accepting else ''} | — | — |")
        for j, sym in enumerate(sorted(edges)):
            label = f"`{sm.names[s]}`{' ✅' if s in sm.accepting else ''}" if j == 0 else ""
            md.append(f"| {label} | `{sym}` | `{sm.names[edges[sym]]}` |")
    md.append("\n_✅ = a state where a trace may validly end._")
    return "\n".join(md)


# --------------------------------------------------------------------------- #
# Built-in demo: a TCP-ish protocol with a data-transfer loop                    #
# --------------------------------------------------------------------------- #
def demo_traces(n: int = 300, seed: int = 0) -> List[Trace]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        tr = ["SYN", "SYN_ACK", "ACK"]               # handshake -> ESTABLISHED
        for _ in range(rng.randint(0, 5)):           # variable data transfer (the loop)
            tr += ["DATA", "DATA_ACK"]
        tr += ["FIN", "FIN_ACK", "ACK"]              # teardown -> CLOSED
        out.append(tr)
    return out


# --------------------------------------------------------------------------- #
# CLI                                                                           #
# --------------------------------------------------------------------------- #
def _main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Infer a protocol state machine from traces.")
    p.add_argument("--dataset", help="folder with train.jsonl/test.jsonl of protocol traces")
    p.add_argument("--demo", action="store_true", help="use the built-in TCP-ish demo traces")
    p.add_argument("--attr", help="fold an event attribute into the symbol (e.g. status)")
    p.add_argument("--k", type=int, default=2, help="k-tails depth [default: 2]")
    p.add_argument("--out", help="write the Markdown report to this file")
    args = p.parse_args(argv)

    if args.demo:
        traces = demo_traces()
    elif args.dataset:
        traces = load_traces(args.dataset, attr=args.attr)
    else:
        p.error("provide --dataset DIR or --demo")

    if not traces:
        print("no traces found"); return 1

    sm, stats = infer(traces, k=args.k)
    autoname(sm)
    md = to_markdown(sm, stats)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(md + "\n")
        print(f"wrote {args.out}  ({stats['dfa_states']} states, "
              f"{stats['compression']['bits_saved']:+} bits)")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
