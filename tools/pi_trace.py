#!/usr/bin/env python3
"""Read back a per-call Pi event trace as a timeline.

A discovery call is a black box that either produces a candidate or does not,
and when it does not, the difference that matters -- the agent was looping, the
agent was thinking, the provider went quiet, a tool never returned -- is not in
the result file. The worker writes every event it sees to
``<request>.events.jsonl`` as it sees them; this turns that file into the
timeline.

    python -I tools/pi_trace.py <events.jsonl> [--tools] [--gaps 5]

With no arguments it prints, for each trace it can find under ``.runtime/runs``,
one line: how long the call ran, what it did, and where the time went.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load(path: Path) -> list[dict]:
    """Every trace line, as ``(seconds, event)``. Unparseable lines are kept.

    Kept rather than skipped because a trace that stops mid-line is a trace
    whose writer was killed mid-write, which is itself the finding.

    Two on-disk shapes are read. The current one wraps each record as
    ``{"trace_s": <seconds>, "event": <record>}``; the older one, written before
    records were timestamped, is the record on its own. A bare record is treated
    as its own event with no time, because reading it as an *empty* event makes
    the summary report zero tool calls for a trace full of them -- a false
    reading of a real file, which is worse than a crash.
    """
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            rows.append({"trace_s": None, "event": {"type": "<unparseable>"}})
            continue
        if isinstance(row, dict) and "event" not in row:
            rows.append({"trace_s": None, "event": row})
        else:
            rows.append(row)
    return rows


def summarise(rows: list[dict]) -> dict:
    """Times and counts for one call.

    Two different questions are answered separately, because conflating them
    produced a false report the first time this ran. *How many tools have not
    finished* is ``starts - ends``: every ``tool_execution_end`` answers some
    start, so a difference is an outstanding call whatever the ids say. *How many
    ends could not be attributed to a start by id* is a statement about the trace
    rather than about the call -- Pi emits some ends with no ``toolCallId`` at
    all, and pairing by id alone reads those as tools that never returned. Only
    the second number is a caveat; the first is the finding.
    """
    times = [r["trace_s"] for r in rows if isinstance(r.get("trace_s"), (int, float))]
    kinds = Counter(str((r.get("event") or {}).get("type")) for r in rows)
    tools: list[list] = []
    index_of: dict[str, int] = {}
    ends = 0
    ends_without_id = 0
    for row in rows:
        event = row.get("event") or {}
        kind = event.get("type")
        at = row.get("trace_s")
        # Counting does not need a clock. Only durations do, and a trace that
        # predates the timestamped envelope still has every tool call in it --
        # so an untimed record is counted and simply contributes no duration.
        timed = isinstance(at, (int, float))
        # Keyed by toolCallId, never by position or name: Pi can have several
        # executions in flight, and matching ends to starts by order would
        # silently attribute one tool's duration to another.
        raw_id = event.get("toolCallId")
        call_id = str(raw_id)
        if kind == "tool_execution_start":
            index_of[call_id] = len(tools)
            tools.append([at if timed else None, None, str(event.get("toolName"))])
        elif kind == "tool_execution_end":
            ends += 1
            if raw_id is None:
                ends_without_id += 1
                continue
            index = index_of.pop(call_id, None)
            if index is not None and timed:
                tools[index][1] = at
    spans = [
        (float(began), ended, str(name))
        for began, ended, name in tools
        if isinstance(began, (int, float))
    ]
    generated = sum(int((r.get("event") or {}).get("chars") or 0) for r in rows)
    return {
        "seconds": (max(times) - min(times)) if times else 0.0,
        "records": len(rows),
        "kinds": kinds,
        "tools": spans,
        "starts": len(tools),
        "ends": ends,
        "ends_without_id": ends_without_id,
        # Every end answers some start, so the difference is what is outstanding
        # regardless of whether the ids could be paired.
        "open_tools": max(0, len(tools) - ends),
        "unpaired_by_id": len(index_of),
        "generated_chars": generated,
        # Nonzero means this trace predates the timestamped envelope, so every
        # duration in it is unavailable rather than zero.
        "untimed_records": len(rows) - len(times),
    }


def print_timeline(rows: list[dict], *, show_tools: bool, gap_s: float) -> None:
    previous: float | None = None
    for row in rows:
        event = row.get("event") or {}
        kind = str(event.get("type"))
        at = row.get("trace_s")
        # Streaming deltas are summarised by the worker already; the remaining
        # per-record deltas are noise at this level.
        if kind == "message_update":
            continue
        if show_tools and kind.startswith("tool_execution"):
            detail = json.dumps(event.get("args") or {})[:160]
            print(f"{at!s:>9}  {kind}  {event.get('toolName')}  {detail}")
            continue
        if kind.startswith("tool_execution"):
            if kind == "tool_execution_start":
                print(f"{at!s:>9}  tool start  {event.get('toolName')}  "
                      f"{json.dumps(event.get('args') or {})[:140]}")
            continue
        if isinstance(at, (int, float)) and previous is not None and at - previous >= gap_s:
            print(f"{at:>9}  ---- {at - previous:.1f}s with no event ----")
        previous = at if isinstance(at, (int, float)) else previous
        if kind == "trace_stream_summary":
            print(f"{at!s:>9}  generated {event.get('chars')} chars "
                  f"in {event.get('deltas')} deltas {event.get('kinds')}")
        else:
            print(f"{at!s:>9}  {kind}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", nargs="?", type=Path,
                        help="an events.jsonl, or a run directory, or nothing")
    parser.add_argument("--tools", action="store_true",
                        help="print the tool calls and their arguments")
    parser.add_argument("--gaps", type=float, default=10.0,
                        help="report silences of at least this many seconds")
    args = parser.parse_args(argv)

    if args.trace is None:
        root = Path(".runtime/runs")
        targets = sorted(root.glob("*/agent/*/*.events.jsonl"))
    elif args.trace.is_dir():
        targets = sorted(args.trace.glob("agent/*/*.events.jsonl")) or [args.trace]
    else:
        targets = [args.trace]

    if not targets:
        print("no traces found")
        return 0
    for path in targets:
        rows = load(path)
        info = summarise(rows)
        print(f"== {path}")
        span = (f"{info['seconds']:.1f}s" if not info["untimed_records"]
                else "no times (pre-envelope trace)")
        print(f"   {span}, {info['records']} records, "
              f"{info['generated_chars']} chars generated, "
              f"{info['starts']} tool calls, {info['open_tools']} without an end")
        if info["ends_without_id"]:
            print(f"   note: {info['ends_without_id']} tool ends carried no "
                  f"toolCallId, so {info['unpaired_by_id']} duration(s) below are "
                  f"unknown rather than missing")
        for name, count in info["kinds"].most_common():
            if name not in ("message_update",):
                print(f"      {count:>6}  {name}")
        for started, ended, name in info["tools"]:
            if ended is not None:
                span = f"{ended - started:.1f}s"
            elif info["ends_without_id"]:
                # An end arrived that carried no id, so this start not being
                # paired does not mean it never returned -- some unpaired start
                # did. "never returned" here would be a claim the file cannot
                # support.
                span = "no end with this id"
            else:
                span = "never returned"
            print(f"      tool {name:<6} at {started:>8.1f}s  {span}")
        if args.tools or len(targets) == 1:
            print()
            print_timeline(rows, show_tools=args.tools, gap_s=args.gaps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
