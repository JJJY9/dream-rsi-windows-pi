"""Tests for the trace reader.

The reader exists to make a call explain itself, so a reader that reports the
wrong thing is worse than no reader at all: its output goes into the run report
as evidence. Both of these tests are regressions against false readings that this
tool actually produced, and neither was found by reading it -- they were found by
running it over real traces and noticing that a file full of tool calls had been
summarised as having none.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"
if str(TOOLS) not in sys.path:  # pragma: no cover - import bootstrapping
    sys.path.insert(0, str(TOOLS))

import pi_trace  # noqa: E402


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    return path


def test_an_end_without_a_call_id_is_not_a_tool_that_never_returned(tmp_path):
    """Pi emits ``tool_execution_end`` with no ``toolCallId``. Pairing only by id
    reads each of those as a tool still running, which turns a finished call into
    one that appears to be stuck -- the exact misreading the tool exists to
    prevent."""
    path = _write(tmp_path / "t.jsonl", [
        {"trace_s": 1.0, "event": {"type": "tool_execution_start",
                                   "toolCallId": "a", "toolName": "bash"}},
        {"trace_s": 2.0, "event": {"type": "tool_execution_end", "toolCallId": "a"}},
        {"trace_s": 3.0, "event": {"type": "tool_execution_start",
                                   "toolCallId": "b", "toolName": "bash"}},
        {"trace_s": 4.0, "event": {"type": "tool_execution_end"}},  # no id
    ])
    info = pi_trace.summarise(pi_trace.load(path))

    assert info["starts"] == 2
    assert info["ends"] == 2
    assert info["open_tools"] == 0, "every start has an end, so nothing is outstanding"
    assert info["ends_without_id"] == 1, "and the caveat is reported separately"


def test_a_start_with_no_end_is_counted_as_outstanding(tmp_path):
    """The other direction, so the fix above cannot pass by counting nothing."""
    path = _write(tmp_path / "t.jsonl", [
        {"trace_s": 1.0, "event": {"type": "tool_execution_start",
                                   "toolCallId": "a", "toolName": "bash"}},
        {"trace_s": 2.0, "event": {"type": "tool_execution_end", "toolCallId": "a"}},
        {"trace_s": 3.0, "event": {"type": "tool_execution_start",
                                   "toolCallId": "b", "toolName": "bash"}},
    ])
    info = pi_trace.summarise(pi_trace.load(path))

    assert info["open_tools"] == 1


def test_a_trace_written_before_records_were_timestamped_still_reads(tmp_path):
    """Archived traces are in the older, unwrapped shape. Reading those as empty
    events reports zero tool calls for a file full of them -- and a zero reads as
    a finding, which is why the reader says "no times" instead of "0.0s"."""
    path = _write(tmp_path / "t.jsonl", [
        {"type": "tool_execution_start", "toolCallId": "a", "toolName": "bash"},
        {"type": "tool_execution_end", "toolCallId": "a"},
        {"trace_truncated_after": 4000},
    ])
    info = pi_trace.summarise(pi_trace.load(path))

    assert info["starts"] == 1, "the bare record is the event, not an empty one"
    assert info["open_tools"] == 0
    assert info["untimed_records"] == 3
    assert info["seconds"] == 0.0


def test_an_unparseable_line_is_kept_rather_than_dropped(tmp_path):
    """A trace that stops mid-line is a trace whose writer was killed mid-write."""
    (tmp_path / "t.jsonl").write_text(
        '{"trace_s": 1.0, "event": {"type": "turn_start"}}\n{"trace_s": 2.0, "eve',
        encoding="utf-8",
    )
    rows = pi_trace.load(tmp_path / "t.jsonl")

    assert len(rows) == 2
    assert rows[-1]["event"]["type"] == "<unparseable>"
