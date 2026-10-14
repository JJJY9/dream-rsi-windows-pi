# Task: speed up khoj's org-mode parser — a real third-party project

**The workspace is a copy of [khoj-ai/khoj](https://github.com/khoj-ai/khoj)'s
`src/khoj` package (pinned commit `ae229ca`, AGPL-3.0, vendored verbatim).**
Khoj is an open-source personal-AI search app; every content sync re-parses
the user's org-mode files — the format a large share of Emacs users keep
years of notes, TODOs and clocked work logs in. The parser is the classic
`orgnode.py` (Charles Cave, 2009) and it walks every line of every file
through a stack of `re.search` calls.

## What you may edit

**Exactly one file:**

```
processor/content/org_mode/orgnode.py
```

Everything else (the rest of the vendored package, the license) is read-only
reference. The grader verifies the hashes of every other file; touching any of
them scores 0. New scratch files you create for your own experiments are
allowed and ignored.

## What must stay true

- **Public behavior, exactly**: `makelist(document, filename)` must return
  nodes whose public attributes reproduce the shipped parser's on a frozen
  corpus of 15 jobs (14 small documents covering nesting, TODO keywords,
  priorities, tags, property and logbook drawers, clock lines, scheduled/
  deadline/closed dates, hash comments, indentation, `*nospace` non-headings,
  Chinese text — plus one re-parse of an earlier document after all the
  others, so any caching you add must be call-order independent). One
  mismatched job scores 0. The compared surface is: `level`, `heading`,
  `body`, `tags`, `todo`, `priority`, `scheduled`, `deadline`, `closed`,
  `properties`, `logbook`, `ancestors`.
- The grader loads your file **by path** as a standalone module and calls it
  directly — keep all imports standard-library, keep the file importable on
  its own, and keep the `makelist(file, filename, start_line=1,
  ancestry_lines=0)` signature. Documents are passed as **strings** (the
  reference splits them itself).
- **Standard library only** — no new dependencies.

## How you are scored

The grader times a fixed ~65k-line workload (two generated org files,
~5000 headings with drawers, schedules, clock lines and prose bodies) against
the **frozen reference module running in a separate process**, both under the
identical protocol (several repetitions, best kept):

```
speedup = time(reference) / time(yours)
score   = min(1.0, speedup / 2.5)
```

Points of reference measured on the authoring machine: the shipped
implementation scores ~0.4 (1x); precompiling the per-line regexes measures
~1.6x (0.6); adding substring guards so most lines never enter the regex
machinery measures ~2.0x (0.8); routing plain body lines past the whole
drawer/clock/property chain, hoisting the per-heading filename normalisation
and memoising clock parsing measures ~2.6–3.0x (1.0). Profiling shows where
the heat is: every single line is run through ~10 module-level `re.search`
calls (each paying the pattern-cache lookup), every heading re-normalises the
same filename through `pathlib`, and every clock line re-parses an identical
timestamp with `strptime`.

## Notes

- The workload's line mix mirrors a real GTD file: mostly prose and bullets,
  with headings, drawers and date lines scattered through it. Making the
  common line cheap matters more than micro-tuning the rare ones.
- **Keep the file importable at every save** — the call can end at any
  moment, and whatever the file contains then is what gets graded. Save your
  work as you go; don't hold a rewrite in a scratch buffer until the end.
- Anything you change must keep the parser's behavior for *every* line, not
  just the common ones: the gate corpus deliberately includes the odd corners
  (drawers left unclosed, comments that open drawers, `*` lines that aren't
  headings).
