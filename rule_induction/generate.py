"""CLI for the synthetic data generator (Skill 3).

Examples::

    # Full ladder (levels 0-5 + NEG), 3 seeds each, 200 cases per seed:
    python -m rule_induction.generate --out data

    # Just the build-order starter set (levels 0-3 + NEG):
    python -m rule_induction.generate --out data --levels level0 level1 level2 level3 neg

    # Bigger, more seeds, custom train ratio:
    python -m rule_induction.generate --out data -n 500 --seeds 10 --ratio 0.6
"""

from __future__ import annotations

import argparse

from .dataset import build
from .levels import ALL_LEVELS


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Generate the synthetic rule-induction ladder.")
    p.add_argument("--out", default="data", help="output directory (default: data)")
    p.add_argument("--levels", nargs="+", default=None, metavar="LEVEL",
                   help=f"levels to generate (default: all). Choices: {ALL_LEVELS}")
    p.add_argument("-n", type=int, default=200, help="cases per (level, seed) [default 200]")
    p.add_argument("--seeds", type=int, default=3, help="number of seeds per level [default 3]")
    p.add_argument("--seed-start", type=int, default=0, help="first seed [default 0]")
    p.add_argument("--ratio", type=float, default=0.7, help="train fraction [default 0.7]")
    args = p.parse_args(argv)

    if args.levels:
        unknown = [lv for lv in args.levels if lv not in ALL_LEVELS]
        if unknown:
            p.error(f"unknown levels {unknown}; choose from {ALL_LEVELS}")

    manifest = build(args.out, levels=args.levels, n=args.n,
                     seeds=args.seeds, ratio=args.ratio, seed_start=args.seed_start)

    print(f"Wrote {len(manifest['entries'])} (level, seed) datasets to {args.out!r}\n")
    print(f"{'level':8} {'seed':>4} {'train':>6} {'test':>5}  label_distribution")
    print("-" * 64)
    for e in manifest["entries"]:
        dist = ", ".join(f"{k}={v}" for k, v in e["label_distribution"].items())
        print(f"{e['level']:8} {e['seed']:>4} {e['n_train']:>6} {e['n_test']:>5}  {dist}")
    print(f"\nManifest: {args.out}/manifest.json")
    print("Ground truth lives in each seed_*/ground_truth.json (scorer-only; "
          "never feed it to the inducer).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
