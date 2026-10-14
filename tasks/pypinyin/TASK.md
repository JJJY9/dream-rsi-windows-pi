# Task: speed up pypinyin's per-character conversion — a real third-party project

**The workspace is a copy of [mozillazg/pypinyin](https://github.com/mozillazg/pypinyin)
0.55.0 (MIT, vendored from the pinned PyPI sdist).** This task asks for a real
performance improvement to a Chinese-NLP building block — the library that
powers pinyin lookup in search ranking, contact sorting, IME candidates and
speech front-ends.

## What you may edit

**Exactly one file:**

```
pypinyin/converter.py
```

Everything else (the package's other modules, the dictionaries, the license)
is read-only reference. The grader verifies the hashes of every other file;
touching any of them scores 0. New scratch files you create for your own
experiments are allowed and ignored.

## What must stay true

- **Public behavior, exactly**: `pypinyin.pinyin()` and
  `pypinyin.lazy_pinyin()` must reproduce the shipped implementation's output
  on a frozen corpus of 84 jobs (12 texts — multi-pronunciation phrases like
  "银行行长", mixed Chinese/English/digits, punctuation, emoji, rare
  characters, empty string — across 7 styles: default, NORMAL, INITIALS,
  FINALS, FIRST_LETTER, TONE2, TONE3). One mismatched job scores 0.
- The module must import cleanly, and `DefaultConverter` must keep working
  through the public `pypinyin` API (the grader never instantiates internals
  directly).
- **Standard library only** — no new dependencies.

## How you are scored

The grader times a fixed ~100k-character workload (`pinyin()` on a long text,
then `lazy_pinyin()` on another) and compares against the **frozen reference
package running in a separate process**:

```
speedup = time(reference) / time(yours)
score   = min(1.0, speedup / 2.0)
```

Points of reference measured on the authoring machine: the shipped
implementation scores ~0.5 (1x); a fix for the per-phrase `deepcopy` in
`_phrase_pinyin` measures ~1.4x (0.7); memoising the pure per-character
style conversion in `convert_style` measures ~2.3x (1.0). The code itself
shows where the heat is: every character walks the full
`convert_styles → convert_style → regex post-processing` machinery, and every
matched phrase is `deepcopy`-ed.

## Notes

- The style conversion is a pure function of `(style, pinyin, han, strict)`
  for the default converter — repeated characters across a document are the
  common case, not the exception.
- `PHRASES_DICT` entries are shared module state: whatever you do to them,
  the dictionary itself must never be mutated.
- **Keep the file importable at every save** — the call can end at any
  moment, and whatever the file contains then is what gets graded.
