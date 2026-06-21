"""Demo — a HARD context-free grammar abstraction, invented once, transferred for free.

The flagship of the abstraction thesis. A pushdown recogniser (typed nesting +
exact depth + equal pair counts) is **invented** on one alphabet (parentheses),
then **reused** on a totally different one (HTML-ish `div`/`span` tags) by supplying
new parameters. Because the primitive is parametric and already in the library, the
two-part code charges only a pointer to reuse it — so the expensive abstraction
becomes a bargain the moment it transfers across vocabularies.

  * invention (Mechanism 1, beyond the fixed grammar): coin a stack recogniser
  * analogy (Mechanism 4): the SAME grammar in parentheses and in tags
  * amortization (two-part MDL): cheap to invent once, ~free to reuse

Everything is judged blind on a holdout. Run::

    python -m rule_induction.demo_grammar_transfer
"""

from __future__ import annotations

import random
import tempfile
from typing import Callable, Dict, List, Tuple

from . import arbiter
from .dataset import split_train_test
from .librarian import Librarian
from .model import event, retime

# A reusable, PARAMETRIC pushdown recogniser. `pairs` (opener->closer) and `depth`
# are parameters, so the SAME primitive recognises any typed-nesting alphabet.
PRIM = '''
def typed_nest_balanced(events, pairs, depth):
    closers = {v: k for k, v in pairs.items()}
    st = []; md = 0; counts = {}
    for e in events:
        t = e["type"]
        if t in pairs:
            st.append(t)
            if len(st) > md: md = len(st)
        elif t in closers:
            if not st or st[-1] != closers[t]: return False
            o = st.pop(); counts[o] = counts.get(o, 0) + 1
    if st: return False
    if md != depth: return False
    return len(counts) == len(pairs) and len(set(counts.values())) == 1
'''


def hypothesis(pairs: Dict[str, str], depth: int = 3) -> Dict:
    return {
        "kind": "composed",
        "name": "typed_nest_balanced",
        "description": "typed nesting + exact depth + equal pair counts (parametric)",
        "primitives": {"typed_nest_balanced": PRIM},
        "clauses": [{"all": [{"prim": "typed_nest_balanced",
                              "params": {"pairs": pairs, "depth": depth}}], "outcome": "accept"}],
        "default": "reject",
    }


def make_generator(o1: str, c1: str, o2: str, c2: str,
                   noise: List[str], depth: int) -> Callable[[int, int], List[Dict]]:
    """Return a generator of the hard typed-nesting grammar over a given alphabet."""
    pairs, closers = {o1: c1, o2: c2}, {c1: o1, c2: o2}

    def rule(ev):
        st, md, ca, cb = [], 0, 0, 0
        for e in ev:
            t = e["type"]
            if t in pairs:
                st.append(t); md = max(md, len(st))
            elif t in closers:
                if not st or st[-1] != closers[t]:
                    return False
                ca, cb = (ca + 1, cb) if st.pop() == o1 else (ca, cb + 1)
        return (not st) and md == depth and ca == cb

    def maxd(tk):
        st, m = [], 0
        for t in tk:
            if t in pairs:
                st.append(t); m = max(m, len(st))
            elif t in closers and st:
                st.pop()
        return m

    def rand_valid(rng, ka, kb, cap):
        left, st, out = {o1: ka, o2: kb}, [], []
        while left[o1] + left[o2] > 0 or st:
            can = [t for t in left if left[t] > 0] if len(st) < cap else []
            acts = (["o"] if can else []) + (["c"] if st else [])
            if not acts:
                break
            if rng.choice(acts) == "o":
                t = rng.choice(can); left[t] -= 1; st.append(t); out.append(t)
            else:
                out.append(pairs[st.pop()])
        return out

    def valid_at(rng, ka, kb, d):
        best = []
        for _ in range(80):
            tk = rand_valid(rng, ka, kb, d)
            if maxd(tk) == d:
                return tk
            best = tk
        return best

    def tokens(rng):
        fl = rng.choice(["p", "p", "p", "wd", "wc", "mm", "unb", "cf"])
        k = rng.randint(2, 4)
        if fl == "p":
            return valid_at(rng, k, k, depth)
        if fl == "wd":
            return valid_at(rng, k, k, rng.choice([2, 4]))
        if fl == "wc":
            kb = max(1, k + rng.choice([-1, 1, 2])); kb = kb if kb != k else k + 1
            return valid_at(rng, k, kb, depth)
        if fl == "mm":
            t = valid_at(rng, k, k, depth)
            idx = [i for i, x in enumerate(t) if x in closers]
            if idx:
                i = rng.choice(idx); t[i] = c2 if t[i] == c1 else c1
            return t
        if fl == "unb":
            t = valid_at(rng, k, k, depth)
            idx = [i for i, x in enumerate(t) if x in closers]
            if idx:
                t.pop(rng.choice(idx))
            return t
        return [rng.choice([c1, c2])] + valid_at(rng, k, k, depth)

    def noisy(rng, tk):
        tk = list(tk)
        for _ in range(rng.randint(0, 3)):
            tk.insert(rng.randint(0, len(tk)), rng.choice(noise))
        return retime([event(0, t) for t in tk])

    def generate(n, seed):
        rng = random.Random(seed)
        pos, neg, guard = [], [], 0
        while (len(pos) < n // 2 or len(neg) < n // 2) and guard < n * 50:
            guard += 1
            ev = noisy(rng, tokens(rng))
            acc = rule(ev)
            bucket = pos if acc else neg
            if len(bucket) >= n // 2:
                continue
            bucket.append({"case_id": f"c_{len(pos) + len(neg):05d}", "events": ev,
                           "outcome": "accept" if acc else "reject", "level": "grammar"})
        cases = pos + neg
        rng.shuffle(cases)
        return cases

    return generate


# Two alphabets, ONE abstract grammar.
PARENS = ("kx", "kz", "wp", "wq", ["n1", "n2"])
TAGS = ("div_o", "div_c", "span_o", "span_c", ["txt", "img"])


def run(n: int = 5000) -> Tuple[Dict, Dict]:
    lib = Librarian(tempfile.mkdtemp(prefix="eureka_transfer_"))

    genA = make_generator(*PARENS, depth=3)
    trA, teA = split_train_test(genA(n, 0), 0.6, 0)
    vA = arbiter.adjudicate(hypothesis({"kx": "kz", "wp": "wq"}), trA, teA, librarian=lib,
                            known_primitives=lib.known_primitive_names(), level_origin="grammar/A")

    genB = make_generator(*TAGS, depth=3)
    trB, teB = split_train_test(genB(n, 1), 0.6, 1)
    vB = arbiter.adjudicate(hypothesis({"div_o": "div_c", "span_o": "span_c"}), trB, teB, librarian=lib,
                            known_primitives=lib.known_primitive_names(), level_origin="grammar/B")
    return vA, vB


def main() -> int:
    vA, vB = run()
    print(f"A · INVENT typed_nest_balanced (parentheses) : acc={vA['test_accuracy']:.3f}  "
          f"bits_saved={vA['bits_saved']:+.0f}  [L_program={vA['program_bits']:.0f}]  promoted={vA['promoted']}")
    print(f"B · REUSE typed_nest_balanced (div/span tags): acc={vB['test_accuracy']:.3f}  "
          f"bits_saved={vB['bits_saved']:+.0f}  [L_program={vB['program_bits']:.0f}]")
    saved = vA["program_bits"] - vB["program_bits"]
    print(f"\nThe pushdown recogniser cost {vA['program_bits']:.0f} bits to invent on parentheses; "
          f"reusing it on a different alphabet cost only {vB['program_bits']:.0f}. "
          f"Same hard grammar, new vocabulary, ~{saved:.0f} bits saved by the abstraction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
