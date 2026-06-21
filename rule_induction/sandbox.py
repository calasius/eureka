"""Sandbox runner for code-hypotheses (Section 2, third defense).

Treat every code-hypothesis as **hostile** until a sandboxed run returns clean.
A candidate is a snippet defining ``def predict(events) -> outcome``. We run it in
a separate, locked-down subprocess and only trust the predictions it returns.

Defenses applied (Linux):
  * **Wall-clock timeout** — the parent kills the child past ``timeout_s``.
  * **CPU cap** — RLIMIT_CPU.
  * **Memory cap** — RLIMIT_AS.
  * **No filesystem writes** — RLIMIT_FSIZE = 0 (any file write raises SIGXFSZ;
    stdout is a pipe, not a file, so results still flow).
  * **Few file descriptors** — RLIMIT_NOFILE.
  * **Restricted builtins** — the candidate gets no ``open``/``__import__``/
    ``eval``/``exec``/``compile``, so it cannot touch the disk or import a
    network library in the first place.
  * **Isolated interpreter** — ``python -I -S -B`` (ignore env/user-site, no
    site, no .pyc), minimal environment.

Honest limit: full **network** isolation needs OS-level confinement
(network namespace / nsjail / firejail / a container). The restricted builtins
prevent importing ``socket`` from the candidate, but a hostile interpreter-level
escape is out of scope for a single-host runner. Run untrusted code under
nsjail/containers in production.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

DEFAULT_TIMEOUT_S = 5.0
DEFAULT_MEM_MB = 512
DEFAULT_CPU_S = 2

# Trusted wrapper that runs INSIDE the sandboxed child. It reads {source, cases}
# from stdin, execs the candidate with a restricted builtins namespace, runs
# predict() over the cases, and prints a JSON result to stdout. Only this wrapper
# has full builtins; the candidate (`source`) does not.
_CHILD = r"""
import sys, json

_ALLOWED = [
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter",
    "float", "frozenset", "int", "isinstance", "issubclass", "len", "list",
    "map", "max", "min", "next", "pow", "range", "reversed", "round", "set",
    "sorted", "str", "sum", "tuple", "zip", "True", "False", "None",
    "ValueError", "KeyError", "IndexError", "TypeError", "Exception",
]
import builtins as _b
_safe = {n: getattr(_b, n) for n in _ALLOWED if hasattr(_b, n)}

data = json.load(sys.stdin)
source, cases = data["source"], data["cases"]
glb = {"__builtins__": _safe}
try:
    exec(source, glb)
    predict = glb.get("predict")
    if not callable(predict):
        raise ValueError("hypothesis must define predict(events)")
    preds = [predict(c["events"]) for c in cases]
    # Predictions must be JSON-encodable labels.
    sys.stdout.write(json.dumps({"ok": True, "predictions": preds}))
except Exception as exc:  # candidate is hostile-until-clean: any error => dirty
    sys.stdout.write(json.dumps({"ok": False, "error": repr(exc)}))
"""


# Trusted wrapper for the agent's TRAIN-ONLY scratchpad: it execs an `analyze`
# function and runs it over the train cases handed in. Same lockdown as _CHILD, so
# the analysis code has no `open`/`__import__` — it physically cannot read
# test.jsonl or ground_truth.json; it only ever sees the train data passed in.
_CHILD_ANALYZE = r"""
import sys, json

_ALLOWED = [
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter",
    "float", "frozenset", "int", "isinstance", "issubclass", "len", "list",
    "map", "max", "min", "next", "pow", "range", "reversed", "round", "set",
    "sorted", "str", "sum", "tuple", "zip", "True", "False", "None",
    "ValueError", "KeyError", "IndexError", "TypeError", "Exception",
]
import builtins as _b
_safe = {n: getattr(_b, n) for n in _ALLOWED if hasattr(_b, n)}

data = json.load(sys.stdin)
source, train = data["source"], data["train"]
glb = {"__builtins__": _safe}
try:
    exec(source, glb)
    analyze = glb.get("analyze")
    if not callable(analyze):
        raise ValueError("scratchpad must define analyze(train)")
    result = analyze(train)
    json.dumps(result)  # must be JSON-serialisable
    sys.stdout.write(json.dumps({"ok": True, "result": result}))
except Exception as exc:
    sys.stdout.write(json.dumps({"ok": False, "error": repr(exc)}))
"""


class SandboxError(Exception):
    """The candidate did not return a clean run (timeout, crash, bad output)."""


def _preexec(cpu_s: int, mem_bytes: int):
    import resource

    def _apply():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))

    return _apply


def _run_sandboxed(child: str, payload: Dict[str, Any], *,
                   timeout_s: float, mem_mb: int, cpu_s: int) -> Dict[str, Any]:
    """Run one locked-down child over a JSON payload; return its parsed result dict."""
    preexec = _preexec(cpu_s, mem_mb * 1024 * 1024) if hasattr(os, "fork") else None
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-S", "-B", "-c", child],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
            preexec_fn=preexec,
        )
    except subprocess.TimeoutExpired:
        raise SandboxError(f"timeout after {timeout_s}s")

    if proc.returncode != 0:
        raise SandboxError(
            f"non-zero exit {proc.returncode} (likely cpu/memory limit). "
            f"stderr: {proc.stderr.strip()[:200]}"
        )
    out = proc.stdout.strip()
    if not out:
        raise SandboxError(f"no output. stderr: {proc.stderr.strip()[:200]}")
    try:
        result = json.loads(out)
    except json.JSONDecodeError:
        raise SandboxError(f"unparseable output: {out[:200]}")
    if not result.get("ok"):
        raise SandboxError(f"candidate raised: {result.get('error')}")
    return result


def run_code(source: str, cases: List[Dict[str, Any]], *,
             timeout_s: float = DEFAULT_TIMEOUT_S,
             mem_mb: int = DEFAULT_MEM_MB,
             cpu_s: int = DEFAULT_CPU_S) -> List[Any]:
    """Run ``predict`` over ``cases`` in the sandbox; return predictions.

    Raises ``SandboxError`` on timeout, non-zero exit, memory/cpu kill, malformed
    output, or any exception inside the candidate. Callers must treat that as an
    automatic rejection of the hypothesis.
    """
    result = _run_sandboxed(_CHILD, {"source": source, "cases": cases},
                            timeout_s=timeout_s, mem_mb=mem_mb, cpu_s=cpu_s)
    preds = result["predictions"]
    if len(preds) != len(cases):
        raise SandboxError("prediction count mismatch")
    return preds


def run_analysis(source: str, train: List[Dict[str, Any]], *,
                 timeout_s: float = DEFAULT_TIMEOUT_S,
                 mem_mb: int = DEFAULT_MEM_MB,
                 cpu_s: int = DEFAULT_CPU_S) -> Any:
    """Run the agent's ``analyze(train)`` scratchpad; return its JSON-serialisable result.

    The analysis code is sandboxed exactly like a hypothesis: no file access, so it
    can read only the train cases handed in — never test.jsonl or ground_truth.json.
    """
    result = _run_sandboxed(_CHILD_ANALYZE, {"source": source, "train": train},
                            timeout_s=timeout_s, mem_mb=mem_mb, cpu_s=cpu_s)
    return result["result"]
