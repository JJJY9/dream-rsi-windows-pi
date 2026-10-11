"""Policy development (§3.3): turn feedback on ``H_t`` into candidate revisions.

Two developers share one interface:

``ScriptedVariantDeveloper``
    Offline. Emits the bundled, hand-written policy variants. These are *real,
    distinct, deterministic policies* — they make the selection machinery
    exercisable with no model in the loop — but they are **not** model output
    and every record they create says so (``origin="revision"`` plus
    ``developer="scripted-offline"``). A run that uses them is an offline
    fixture run and is never reported as a live result.

``LlmPolicyDeveloper``
    Live. One real model call that rewrites the incumbent source given the
    measured feedback. Blocked, not faked, when no credentials are configured.

Every proposal — from either developer — passes the same gate before it may be
replayed: :func:`validate_policy_source` (static) and a load-and-call probe in
an isolated worker. A proposal that fails either is recorded ``REJECTED`` and
the incumbent is left untouched.
"""
from __future__ import annotations

import ast
import hashlib
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

#: The only modules a policy may import.
#:
#: This was a denylist, and a denylist of imports cannot be made to work: the
#: standard library is far larger than any list of names someone remembers to
#: forbid, and the misses are not exotic. ``io.open`` and
#: ``pathlib.Path(...).read_text()`` are both file I/O, neither was on the list,
#: and neither is caught by :data:`FORBIDDEN_CALLS` either, because that check
#: compares the *dotted* name and only bare ``open`` was listed. Everything here
#: is pure computation: no I/O, no clock, no randomness, no process access.
ALLOWED_IMPORTS = frozenset({
    "__future__", "typing", "dataclasses", "math", "itertools",
    "functools", "collections", "abc", "enum",
})

#: Calls that reach outside a pure function of the observation.
#:
#: ``open``/``exec``/``eval`` are the obvious ones. The reflection group —
#: ``getattr``, ``setattr``, ``delattr``, ``type``, ``dir``, ``super``,
#: ``object`` — is what makes a *denylist* of dunder names useless on its own:
#: ``getattr(x, "__class__")`` puts the dunder in a string literal, where the
#: AST sees an ``ast.Constant`` and never an ``ast.Attribute``, so the
#: dunder-attribute rule below has nothing to match. That was a working escape
#: from this gate until it was closed; see ``test_policy_gate.py``.
FORBIDDEN_CALLS = frozenset({
    "open", "exec", "eval", "compile", "__import__", "input", "breakpoint",
    "globals", "locals", "vars", "memoryview",
    "getattr", "setattr", "delattr", "type", "dir", "super", "object",
})

REQUIRED_ENTRY = "select"


@dataclass(frozen=True)
class StaticCheck:
    ok: bool
    reason: str | None
    digest: str


def _is_dunder(name: str) -> bool:
    """``__x__`` — the spelling of every attribute Python's data model exposes."""
    return len(name) > 4 and name.startswith("__") and name.endswith("__")


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """``id()`` of every string constant that is a docstring.

    Docstrings are inert: the interpreter stores them and never evaluates their
    contents, so text is not a route to reflection. Excluding them keeps the
    ``__`` string rule from rejecting a model that merely *mentions* a dunder
    while explaining its policy.
    """
    found: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            found.add(id(first.value))
    return found


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_dotted(node.value)}.{node.attr}"
    return ""


def validate_policy_source(path: Path) -> StaticCheck:
    """Parse and screen a policy file. Never imports or executes it.

    This is a *screen*, not a sandbox, and passing it does **not** make the file
    safe. It is a static check on source that a language model wrote, and static
    checks on a Turing-complete language are defeated by anyone who tries hard
    enough and knows the runtime. What it reliably does is reject the escapes
    that are *easy* to reach — including the ones a model might stumble into
    rather than choose — and reject the nondeterminism sources, which is what
    replay correctness actually depends on.

    The real containment is the isolated worker the policy runs in, and even
    that only limits blast radius: a policy executes as untrusted code with the
    user's own rights. Anything that describes this gate as making model output
    safe is describing it wrong.

    The rules are allowlists where an allowlist is possible (imports) and
    deny-lists elsewhere, and every rule that was added because a specific
    escape got through is named in the comment beside it.
    """
    source = Path(path).read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError as exc:
        return StaticCheck(False, f"policy source is not valid UTF-8: {exc}", digest)
    if "\r\n" in text:
        return StaticCheck(False, "policy source must use LF line endings, not CRLF", digest)

    try:
        tree = ast.parse(text, filename=path.name)
    except SyntaxError as exc:
        return StaticCheck(False, f"syntax error at line {exc.lineno}: {exc.msg}", digest)

    entry: ast.FunctionDef | None = None
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [(node.module or "")]
            )
            for name in names:
                root = name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    return StaticCheck(
                        False,
                        f"import {name!r} is not one of the modules a policy may use "
                        f"({', '.join(sorted(ALLOWED_IMPORTS))})",
                        digest,
                    )
            if isinstance(node, ast.ImportFrom) and node.level:
                return StaticCheck(False, "relative imports are not allowed in a policy", digest)
        elif isinstance(node, ast.FunctionDef) and node.name == REQUIRED_ENTRY:
            entry = node

    if entry is None:
        return StaticCheck(
            False, f"policy must define a module-level function {REQUIRED_ENTRY}(observation, rho)", digest
        )
    if entry.decorator_list:
        return StaticCheck(False, f"{REQUIRED_ENTRY} must not be decorated", digest)

    args = entry.args
    if len(args.args) < 2 or args.vararg is not None:
        return StaticCheck(
            False, f"{REQUIRED_ENTRY} must take exactly (observation, rho)", digest
        )

    docstrings = _docstring_nodes(tree)
    for sub in ast.walk(tree):
        if isinstance(sub, ast.Call):
            name = _dotted(sub.func)
            if name in FORBIDDEN_CALLS:
                return StaticCheck(False, f"forbidden call {name!r}()", digest)
        elif isinstance(sub, ast.Attribute) and sub.attr.startswith("__"):
            if sub.attr not in {"__name__", "__doc__", "__file__"}:
                return StaticCheck(False, f"forbidden dunder access {sub.attr!r}", digest)
        elif isinstance(sub, ast.Name) and _is_dunder(sub.id):
            # ``__builtins__["open"]`` needs no forbidden call and no attribute
            # access, so the two rules above both miss it. A policy has no
            # legitimate use for a dunder name.
            return StaticCheck(False, f"forbidden dunder name {sub.id!r}", digest)
        elif (
            isinstance(sub, ast.Constant)
            and isinstance(sub.value, str)
            and "__" in sub.value
            and id(sub) not in docstrings
        ):
            # The other half of the reflection route: the dunder as *data*.
            # Covers ``getattr(x, "__class__")`` and the concatenated
            # ``"__" + "class__"``, since each fragment is itself a constant
            # containing a double underscore. Docstrings are excluded -- they are
            # inert text, and a model explaining itself should not be rejected.
            return StaticCheck(
                False,
                "string literals may not contain '__' (it is only ever used to name a dunder)",
                digest,
            )

    return StaticCheck(True, None, digest)


# --------------------------------------------------------------------------
# Developers
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Proposal:
    """One candidate revision on disk, before any validation."""

    m: int
    path: Path
    developer: str
    is_live: bool
    summary: str = ""


class PolicyDeveloper(Protocol):
    """Produces up to ``count`` candidate policy files."""

    developer_id: str
    is_live: bool

    def propose(
        self,
        *,
        incumbent_path: Path,
        feedback: Sequence[str],
        output_dir: Path,
        count: int,
        iteration_t: int,
        start_m: int = 1,
    ) -> Sequence[Proposal]: ...


#: The offline variants. Each is a genuinely different, legal policy: they rank
#: legal actions differently and spend the root slot differently, so they reach
#: different points on the (n, a) plane and earn different V.
VARIANT_MODES: tuple[str, ...] = ("score_first", "breadth_first", "depth_first")


class ScriptedVariantDeveloper:
    """Offline: emit the bundled hand-written variants. Never claims to be a model."""

    developer_id = "scripted-offline"
    is_live = False

    def propose(
        self,
        *,
        incumbent_path: Path,
        feedback: Sequence[str],
        output_dir: Path,
        count: int,
        iteration_t: int,
        start_m: int = 1,
    ) -> Sequence[Proposal]:
        del feedback  # the variants are fixed; they do not adapt
        template = Path(__file__).resolve().parent / "_variant_template.py"
        source = template.read_text(encoding="utf-8")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Traceability: every revision records the exact incumbent it was
        # proposed against, so a selection can be re-derived later.
        incumbent_digest = hashlib.sha256(Path(incumbent_path).read_bytes()).hexdigest()

        proposals: list[Proposal] = []
        for m in range(start_m, start_m + max(0, count)):
            mode = VARIANT_MODES[(m - 1) % len(VARIANT_MODES)]
            path = output_dir / f"pi_{iteration_t}_{m}_{mode}.py"
            header = (
                f"# revision m={m} at iteration t={iteration_t}; "
                f"incumbent sha256={incumbent_digest}\n"
            )
            body = header + source.replace("{{MODE}}", mode)
            path.write_text(body, encoding="utf-8", newline="\n")
            proposals.append(
                Proposal(
                    m=m,
                    path=path,
                    developer=self.developer_id,
                    is_live=False,
                    summary=f"offline variant mode={mode} (not model output)",
                )
            )
        return proposals


def deploy_policy(source_path: Path, destination: Path) -> str:
    """Copy a chosen policy into its immutable deployment slot; return its digest."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(source_path), destination)
    return hashlib.sha256(destination.read_bytes()).hexdigest()


__all__ = [
    "ALLOWED_IMPORTS",
    "FORBIDDEN_CALLS",
    "VARIANT_MODES",
    "PolicyDeveloper",
    "Proposal",
    "ScriptedVariantDeveloper",
    "StaticCheck",
    "deploy_policy",
    "validate_policy_source",
]
