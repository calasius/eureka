"""The Librarian — persistent store of accepted abstractions (Section 4).

This is NOT a skill. It is *state*: a git-versioned directory of accepted
abstractions, **written by the arbiter** (on promotion) and **read by the
inducer** (at generation time, to get primitives to recombine — Mechanism 2).

Each entry records the rule/function + its MDL score + metadata (what it
compresses, when accepted, provenance). Git gives auditability and a clean
revert path if a false abstraction ever contaminates the library.

Two thresholds (Section 4):
  * run-acceptance threshold — used once, by the arbiter, to accept a rule in a
    single run. (Enforced by the arbiter, not here.)
  * library threshold (STRICTER) — to be admitted as a permanent foundation.
    Enforced by ``Librarian.promote``: admitting a rule as a reusable primitive
    is a bigger commitment than using it once.

Layout::

    <root>/
        config.json            # thresholds + metadata
        index.json             # summary of every entry (fast read for the inducer)
        entries/<id>.json       # one abstraction per file (one fact per file)
        (git repo — every promote/revert is a commit)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

DEFAULT_CONFIG = {
    "run_acceptance_threshold_bits": 1.0,   # informational; enforced by the arbiter
    "library_threshold_bits": 8.0,          # STRICTER; enforced here on promote()
    "description": "Eureka rule-induction abstraction library.",
}

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(text: str) -> str:
    return _SLUG_RE.sub("-", text.strip().lower()).strip("-")[:60] or "entry"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class PromotionRejected(Exception):
    """Raised when a candidate falls below the (stricter) library threshold."""


class Librarian:
    """Read/write API over the git-versioned abstraction store."""

    def __init__(self, root: str = "library", *, use_git: bool = True):
        self.root = os.path.abspath(root)
        self.entries_dir = os.path.join(self.root, "entries")
        self.index_path = os.path.join(self.root, "index.json")
        self.config_path = os.path.join(self.root, "config.json")
        self.use_git = use_git and self._git_available()
        self._ensure_initialized()

    # ----------------------------- setup ---------------------------------- #
    @staticmethod
    def _git_available() -> bool:
        try:
            subprocess.run(["git", "--version"], capture_output=True, check=True)
            return True
        except (OSError, subprocess.CalledProcessError):
            return False

    def _git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", self.root, *args],
                              capture_output=True, text=True, check=check)

    def _ensure_initialized(self) -> None:
        os.makedirs(self.entries_dir, exist_ok=True)
        if not os.path.exists(self.config_path):
            self._write_json(self.config_path, DEFAULT_CONFIG)
        if not os.path.exists(self.index_path):
            self._write_json(self.index_path, {"entries": {}})
        if self.use_git and not os.path.isdir(os.path.join(self.root, ".git")):
            self._git("init", "-q")
            # Local identity so commits never fail in a fresh environment.
            self._git("config", "user.name", "Eureka Librarian")
            self._git("config", "user.email", "calasius@gmail.com")
            readme = os.path.join(self.root, "README.md")
            if not os.path.exists(readme):
                self._write_text(readme, _README)
            self._commit("init: abstraction library", paths=["."])

    # ----------------------------- io helpers ----------------------------- #
    @staticmethod
    def _write_json(path: str, obj: Any) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, ensure_ascii=False)
            fh.write("\n")

    @staticmethod
    def _write_text(path: str, text: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)

    @staticmethod
    def _read_json(path: str) -> Any:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    def _commit(self, message: str, paths: List[str]) -> Optional[str]:
        if not self.use_git:
            return None
        self._git("add", *paths)
        status = self._git("status", "--porcelain")
        if not status.stdout.strip():
            return None
        self._git("commit", "-q", "-m", message)
        return self._git("rev-parse", "--short", "HEAD").stdout.strip()

    # ----------------------------- config --------------------------------- #
    @property
    def config(self) -> Dict[str, Any]:
        return self._read_json(self.config_path)

    def library_threshold(self) -> float:
        return float(self.config.get("library_threshold_bits", DEFAULT_CONFIG["library_threshold_bits"]))

    # ----------------------------- reads (inducer) ------------------------ #
    def index(self) -> Dict[str, Any]:
        return self._read_json(self.index_path)

    def list(self, *, kind: Optional[str] = None, rule_id: Optional[str] = None
             ) -> List[Dict[str, Any]]:
        rows = list(self.index().get("entries", {}).values())
        if kind:
            rows = [r for r in rows if r.get("kind") == kind]
        if rule_id:
            rows = [r for r in rows if r.get("rule_id") == rule_id]
        return sorted(rows, key=lambda r: r.get("id", ""))

    def get(self, entry_id: str) -> Dict[str, Any]:
        path = os.path.join(self.entries_dir, f"{entry_id}.json")
        if not os.path.exists(path):
            raise KeyError(f"no library entry {entry_id!r}")
        return self._read_json(path)

    def known_primitive_names(self) -> frozenset:
        """The invented predicates already in the library — the shared vocabulary
        a composed hypothesis can reference for only a pointer's worth of bits."""
        return frozenset(r["name"] for r in self.list(kind="primitive"))

    # ----------------------------- validation ----------------------------- #
    @staticmethod
    def normalize(entry: Dict[str, Any]) -> Dict[str, Any]:
        """Validate + fill defaults. Returns a normalized copy (no id assigned)."""
        required = ["name", "kind", "description"]
        missing = [k for k in required if not entry.get(k)]
        if missing:
            raise ValueError(f"entry missing required fields: {missing}")
        if entry["kind"] not in ("rule", "function", "primitive"):
            raise ValueError("kind must be 'rule', 'function', or 'primitive'")
        mdl = entry.get("mdl") or {}
        if "bits_saved" not in mdl:
            raise ValueError("entry.mdl.bits_saved is required (the compression gain on holdout)")
        out = dict(entry)
        out.setdefault("spec", {})
        out.setdefault("rule_id", None)
        out.setdefault("program", None)
        out["mdl"] = {
            "bits_saved": float(mdl["bits_saved"]),
            "compresses": mdl.get("compresses", ""),
            "score": mdl.get("score"),
        }
        out["provenance"] = {
            "run_id": (entry.get("provenance") or {}).get("run_id"),
            "investigator": (entry.get("provenance") or {}).get("investigator"),
            "level_origin": (entry.get("provenance") or {}).get("level_origin"),
            "evaluated_on": (entry.get("provenance") or {}).get("evaluated_on", "holdout"),
        }
        return out

    # ----------------------------- writes (arbiter) ----------------------- #
    def promote(self, entry: Dict[str, Any], *, threshold: Optional[float] = None
                ) -> Dict[str, Any]:
        """Admit an abstraction as a permanent primitive — git-committed.

        Enforces the STRICTER library threshold against ``entry.mdl.bits_saved``
        (the compression gain measured on holdout). Raises ``PromotionRejected``
        below threshold. Re-promoting an existing id creates a new version only
        if it strictly improves the recorded compression.
        """
        norm = self.normalize(entry)
        thr = self.library_threshold() if threshold is None else float(threshold)
        bits = norm["mdl"]["bits_saved"]
        if bits < thr:
            raise PromotionRejected(
                f"{bits:.2f} bits saved < library threshold {thr:.2f} bits — "
                f"using a rule once is cheaper than admitting it as a foundation."
            )

        entry_id = entry.get("id") or _slug(norm["name"])
        path = os.path.join(self.entries_dir, f"{entry_id}.json")

        version = 1
        if os.path.exists(path):
            prev = self._read_json(path)
            prev_bits = prev.get("mdl", {}).get("bits_saved", float("-inf"))
            if bits <= prev_bits:
                raise PromotionRejected(
                    f"{entry_id!r} already promoted at {prev_bits:.2f} bits; "
                    f"{bits:.2f} is not an improvement."
                )
            version = int(prev.get("version", 1)) + 1

        record = dict(norm)
        record.update({
            "id": entry_id,
            "status": "promoted",
            "version": version,
            "promoted_at": _now_iso(),
            "promoted_threshold_bits": thr,
        })
        self._write_json(path, record)
        self._update_index(entry_id, record)

        verb = "promote" if version == 1 else f"promote (v{version})"
        sha = self._commit(
            f"{verb}: {entry_id} (+{bits:.2f} bits, {record['kind']}, "
            f"origin {record['provenance'].get('level_origin')})",
            paths=["entries", "index.json"],
        )
        record["commit"] = sha
        return record

    def revert(self, entry_id: str, *, reason: str) -> Dict[str, Any]:
        """Remove a (possibly false) abstraction — git-committed for the audit trail."""
        path = os.path.join(self.entries_dir, f"{entry_id}.json")
        if not os.path.exists(path):
            raise KeyError(f"no library entry {entry_id!r}")
        os.remove(path)
        idx = self.index()
        idx.get("entries", {}).pop(entry_id, None)
        self._write_json(self.index_path, idx)
        sha = self._commit(f"revert: {entry_id} ({reason})", paths=["entries", "index.json"])
        return {"id": entry_id, "reverted": True, "reason": reason, "commit": sha}

    def _update_index(self, entry_id: str, record: Dict[str, Any]) -> None:
        idx = self.index()
        idx.setdefault("entries", {})[entry_id] = {
            "id": entry_id,
            "name": record["name"],
            "kind": record["kind"],
            "rule_id": record.get("rule_id"),
            "description": record["description"],
            "bits_saved": record["mdl"]["bits_saved"],
            "compresses": record["mdl"]["compresses"],
            "version": record["version"],
            "promoted_at": record["promoted_at"],
            "level_origin": record["provenance"].get("level_origin"),
        }
        self._write_json(self.index_path, idx)

    def log(self, n: int = 20) -> str:
        if not self.use_git:
            return "(git unavailable; no history)"
        return self._git("log", f"-{n}", "--oneline", check=False).stdout.strip()


def _main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Inspect/manage the abstraction library.")
    p.add_argument("--root", default="library", help="library root [default: library]")
    p.add_argument("--no-git", action="store_true", help="disable git versioning")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("list", help="list promoted abstractions")
    sp.add_argument("--kind", choices=["rule", "function"])
    sp.add_argument("--rule-id")

    sp = sub.add_parser("show", help="print one entry in full")
    sp.add_argument("id")

    sp = sub.add_parser("promote", help="promote an abstraction from a JSON file")
    sp.add_argument("--file", required=True, help="JSON file describing the entry")
    sp.add_argument("--bits", type=float, help="override entry.mdl.bits_saved")
    sp.add_argument("--threshold", type=float, help="override the library threshold")

    sp = sub.add_parser("revert", help="remove a (false) abstraction")
    sp.add_argument("id")
    sp.add_argument("--reason", required=True)

    sub.add_parser("log", help="show the git history of the library")

    args = p.parse_args(argv)
    lib = Librarian(args.root, use_git=not args.no_git)

    if args.cmd == "list":
        rows = lib.list(kind=args.kind, rule_id=args.rule_id)
        if not rows:
            print("(empty library)")
            return 0
        for r in rows:
            print(f"{r['id']:<32} {r['bits_saved']:>7.2f} bits  v{r['version']}  "
                  f"{r['kind']:<8} {r['description']}")
        return 0

    if args.cmd == "show":
        print(json.dumps(lib.get(args.id), indent=2, ensure_ascii=False))
        return 0

    if args.cmd == "promote":
        entry = lib._read_json(args.file)
        if args.bits is not None:
            entry.setdefault("mdl", {})["bits_saved"] = args.bits
        try:
            rec = lib.promote(entry, threshold=args.threshold)
        except PromotionRejected as exc:
            print(f"REJECTED: {exc}")
            return 1
        print(f"promoted {rec['id']} (+{rec['mdl']['bits_saved']:.2f} bits, "
              f"v{rec['version']}, commit {rec.get('commit')})")
        return 0

    if args.cmd == "revert":
        rec = lib.revert(args.id, reason=args.reason)
        print(f"reverted {rec['id']} (commit {rec.get('commit')})")
        return 0

    if args.cmd == "log":
        print(lib.log())
        return 0

    return 0


_README = """# Abstraction Library (the Librarian)

Git-versioned store of **accepted abstractions** for the rule-induction agent.

- **Written by** the arbiter (Skill 2) on promotion.
- **Read by** the inducer (Skill 1) at generation time, to recombine primitives.
- Each `entries/<id>.json` is one abstraction: rule/function + MDL score + metadata.
- `index.json` is the fast-read summary the inducer loads.
- `config.json` holds the thresholds. The **library threshold** is stricter than
  the per-run acceptance threshold: admitting a permanent foundation is a bigger
  commitment than using a rule once.
- Every promote/revert is a commit — full audit trail, and `revert` cleanly
  removes a false abstraction before it contaminates future induction.

Do not hand-edit; use `python -m rule_induction.librarian` or the `Librarian` API.
"""


if __name__ == "__main__":
    raise SystemExit(_main())
