"""Deterministic gate corpus and timing workload for the khoj orgnode task.

Everything here is frozen input: the gate documents, the filename every job is
parsed under, the JSON-stable node dump, and the shared timing protocol. The
eager freeze in ``.rt-build-khoj.py`` uses these same functions to produce
``expected.json``, and ``ref_runner.py`` / ``evaluator_harness.py`` both time
through ``time_makelist`` so the reference and the candidate are measured the
same way.
"""
from __future__ import annotations

import time

#: The filename handed to ``makelist`` for every gate job. Drive-rooted and
#: bracket-free, so ``normalize_filename`` leaves it untouched on any Windows
#: machine and the frozen ``expected.json`` does not depend on the authoring
#: user's home directory.
GATE_FILENAME = "C:/kb/notes.org"

#: Gate documents. Each one is ``makelist``-ed as a string (the reference
#: splits strings itself) under ``GATE_FILENAME``.
GATE_DOCS: list[str] = [
    # 0: empty input - no nodes at all
    "",
    # 1: pre-heading text only - the intro node path
    "just plain text\nwith two lines\n",
    # 2: a single heading - the last-node path
    "* Single heading\n",
    # 3: nesting, TODO keywords, priority, tags, body
    "* TODO first\n** [#A] priority child :tag@x:\nbody of child\n*** deep\n* second\n",
    # 4: SEQ_TODO declaration + TITLE + custom keywords
    "#+TITLE: My Notes\n#+SEQ_TODO: NEXT(n) DONE(d)\n* NEXT thing\n* DONE other\n",
    # 5: tags without other decoration
    "* heading :tags:more:\nbody\n",
    # 6: property drawers, incl. one left unclosed
    "* item\n:PROPERTIES:\n:ID: abc-1\n:END:\n* next\n:PROPERTIES:\n:ID: abc-2\n:UNCLOSED: yes\nafter drawer\n",
    # 7: clock lines in a logbook
    "* a\nCLOCK: [2026-10-01 Tue 10:00]--[2026-10-01 Tue 11:30] =>  1:30\n* b\n",
    # 8: Chinese headings, body and tags
    "* 中文标题与其他内容\n正文行，包含中文标点。\n** 子标题 :标签:\n",
    # 9: sparse levels and blank lines between headings
    "* h1\n\n* h2\n\n** h2a\n* h3\n\n",
    # 10: hash comments outside drawers are dropped from the body
    "* boundary hash\n# a hash comment line\n* end\n",
    # 11: indentation and tabs
    "* x\n   indented line\n\ttab line\n",
    # 12: "*" without a following space is not a heading - it is intro text
    "*nospace not a heading\n* real\n",
    # 13: a colon in body text and a "#" line that opens a drawer it never closes
    "* h\nNote: a colon in the body\n# tricky :PROPERTIES: opener\n* after\n",
]

#: Workload sizes for timing: ~5000 headings and ~65k lines in total.
_WORKLOAD_SIZES = (4000, 1000)


def _gen_sections(n: int) -> str:
    out = ["#+TITLE: Personal Knowledge Base\n#+SEQ_TODO: TODO ACTIVE WAITING | DONE CANCELLED FAILED\n\n"]
    for i in range(n):
        lvl = "*" * (1 + i % 3)
        out.append(f"{lvl} TODO Item {i} about [[file:notes/{i}.org][topic {i}]] :tag{i%17}:work:\n")
        out.append(
            f"SCHEDULED: <2026-10-{(i%28)+1:02d} Tue> DEADLINE: <2026-11-{(i%28)+1:02d} Wed> "
            f"CLOSED: [2026-09-{(i%28)+1:02d} Mon]\n")
        out.append(":PROPERTIES:\n:ID: item-%d\n:CREATED: [2026-09-01 Mon]\n:END:\n" % i)
        out.append(f"Some body text for item {i}, with a sentence or two of ordinary prose.\n")
        if i % 5 == 0:
            out.append(":LOGBOOK:\nCLOCK: [2026-10-01 Tue 10:00]--[2026-10-01 Tue 11:30] =>  1:30\n:END:\n")
        out.append(f"level {lvl} bullets:\n- point one {i}\n- point two {i}\n\n")
    return "".join(out)


def workload_docs() -> list[str]:
    return [_gen_sections(n) for n in _WORKLOAD_SIZES]


def gate_jobs() -> list[tuple[str, str]]:
    """Ordered (key, document) jobs. The trailing repeat re-parses doc 3 after
    every other job, so a candidate that wrongly memoises across calls (or
    across documents) fails the gate."""
    jobs = [(str(i), doc) for i, doc in enumerate(GATE_DOCS)]
    jobs.append(("repeat", GATE_DOCS[3]))
    return jobs


def dump(nodes) -> list[dict]:
    """Public-API dump of a node list - JSON-stable and locale-free.

    Reads only the properties Khoj's own importer consumes, through the public
    accessors. Dates are stringified in ISO form (``str(datetime)``), which does
    not depend on locale, so the dump is portable across machines.
    """
    out: list[dict] = []
    for n in nodes:
        out.append({
            "level": n.level,
            "heading": n.heading,
            "body": n.body,
            "tags": list(n.tags),
            "todo": n.todo,
            "priority": n.priority,
            "scheduled": str(n.scheduled),
            "deadline": str(n.deadline),
            "closed": str(n.closed),
            "properties": dict(n.properties),
            "logbook": [[str(a), str(b)] for a, b in n.logbook],
            "ancestors": [str(a) for a in n.ancestors],
        })
    return out


def time_makelist(mod, fn: str = GATE_FILENAME) -> float:
    """Best-of timing over the workload documents, shared by both sides.

    The protocol: at least ``MIN_REPS`` repetitions and at least
    ``MIN_TOTAL_S`` seconds in total, keeping the minimum. Taking the minimum
    of several runs rejects one-off scheduler noise; the reference runner and
    the evaluator call this exact function, so neither side gets a more
    favourable protocol than the other.
    """
    min_reps = 4
    min_total_s = 0.4
    max_reps = 30
    docs = workload_docs()
    best = float("inf")
    total = 0.0
    reps = 0
    while reps < max_reps:
        started = time.perf_counter()
        for d in docs:
            mod.makelist(d, fn)
        elapsed = time.perf_counter() - started
        best = min(best, elapsed)
        total += elapsed
        reps += 1
        if total >= min_total_s and reps >= min_reps:
            break
    return best


__all__ = [
    "GATE_DOCS",
    "GATE_FILENAME",
    "dump",
    "gate_jobs",
    "time_makelist",
    "workload_docs",
]
