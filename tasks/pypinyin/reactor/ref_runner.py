"""Time the frozen reference package on the shared workload.

Run as a subprocess by the evaluator harness; prints one JSON line.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "ref_pkg"))
sys.path.insert(0, str(HERE))

import corpus  # noqa: E402
import pypinyin  # noqa: E402


def main() -> None:
    texts = corpus.workload_texts()
    best = float("inf")
    for _ in range(4):
        t0 = time.perf_counter()
        pypinyin.pinyin(texts[0])
        pypinyin.lazy_pinyin(texts[1])
        best = min(best, time.perf_counter() - t0)
    print(json.dumps({"ref_seconds": best}))


if __name__ == "__main__":
    main()
