#!/usr/bin/env python3
"""Subprocess timing worker for the khoj orgnode task. Stdlib only.

    usage: timer_runner.py <module_path>

Loads the given module by path - the frozen reference ``reactor/orgnode_ref.py``
or a candidate workspace's entry file - warms it up on the first gate document,
and times the workload with ``corpus.time_makelist``. Prints one JSON line:
``{"module": <path>, "seconds": <best>}``.

The evaluator runs the reference AND the candidate through this same script, in
separate processes of their own: neither side is ever timed in a process that
earlier work has bloated (GC arenas, allocator state, memory pressure), and the
two sides cannot share module state. In-process candidate timing was measured on
the authoring machine to drift ~25% slower after heavy in-process work while a
fresh reference subprocess stayed flat - an asymmetry that silently depresses
the score, hence this design.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: timer_runner.py <module_path>", file=sys.stderr)
        return 2
    module_path = pathlib.Path(argv[1]).resolve()

    sys.path.insert(0, str(HERE))
    import corpus

    spec = importlib.util.spec_from_file_location("timed_module", module_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Warm-up parse of the first gate document (~130 chars): primes linecache
    # and interned strings so the first timed repetition is not paying one-off
    # import/first-parse costs that the other side would not pay either.
    mod.makelist(corpus.GATE_DOCS[0], corpus.GATE_FILENAME)
    best = corpus.time_makelist(mod)
    print(json.dumps({"module": str(module_path), "seconds": best}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
