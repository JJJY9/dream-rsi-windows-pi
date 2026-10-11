"""Content-addressed store for immutable artifacts.

Layout: ``<root>/<sha[:2]>/<sha[2:4]>/<sha>``. Every write goes through a
temporary file that is flushed and ``fsync``-ed before an atomic
``os.replace``, so a crash can never leave a partial blob at a final path.
Every read re-hashes the stored bytes and raises
:class:`~dream_rsi.domain.errors.IntegrityError` when they do not match the
digest, so tampering is detected instead of propagated.

Only immutable artifacts live here: workspace snapshots, evaluator traces,
policy sources, JSON manifests. Credentials are never written to the store,
nothing is ever unpickled, and no path outside the store root is written.
"""
from __future__ import annotations

import contextlib
import fnmatch
import hashlib
import json
import os
import uuid
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

from ..domain.errors import IntegrityError

__all__ = ["ContentAddressedStore"]

_CHUNK_SIZE = 1024 * 1024
_STAGING_DIR = "_staging"
_HEX_DIGITS = frozenset("0123456789abcdef")
_LINK_MARKER = "LINK:"
_SPECIAL_MARKER = "SPECIAL"


def _validate_digest(digest: str) -> str:
    """Return ``digest`` unchanged, or raise ``ValueError`` if it is not a sha256 hex string."""
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"not a sha256 hex digest: {digest!r}")
    if not set(digest) <= _HEX_DIGITS:
        raise ValueError(f"not a sha256 hex digest (lowercase hex only): {digest!r}")
    return digest


def _canonical_json_bytes(obj: Any) -> bytes:
    """Canonical JSON: sorted keys, compact separators, UTF-8, finite numbers only.

    Canonical form is what makes the digest stable for equal objects regardless
    of key insertion order or of the machine that serialized them.
    """
    text = json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return text.encode("utf-8")


def _unlink_quietly(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink()


def _iter_file_chunks(path: Path) -> Iterator[bytes]:
    """Stream a file in bounded chunks; the whole file is never held in memory."""
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_CHUNK_SIZE)
            if not chunk:
                break
            yield chunk


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    for chunk in _iter_file_chunks(path):
        hasher.update(chunk)
    return hasher.hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    """Case-insensitive containment test (Windows paths differ only by case)."""
    candidate = os.path.normcase(str(path))
    base = os.path.normcase(str(root))
    return candidate == base or candidate.startswith(base + os.sep)


def _is_link(entry: os.DirEntry[str]) -> bool:
    """True for symlinks and NTFS directory junctions (reparse points)."""
    if entry.is_symlink():
        return True
    is_junction = getattr(entry, "is_junction", None)
    if is_junction is None:
        return False
    try:
        return bool(is_junction())
    except OSError:
        return False


def _excluded(rel_posix: str, name: str, patterns: Sequence[str]) -> bool:
    """Match a bare name (any depth) or a POSIX-style relative path.

    Matching is case-sensitive on purpose: the tree digest must be identical on
    every platform and filesystem for the same tree contents.
    """
    for pattern in patterns:
        if fnmatch.fnmatchcase(name, pattern) or fnmatch.fnmatchcase(rel_posix, pattern):
            return True
    return False


def _relative_posix(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _collect_tree_entries(
    root: Path, current: Path, patterns: Sequence[str], out: list[tuple[str, str]]
) -> None:
    """Collect ``(relative_posix_path, marker)`` for files and links below ``current``.

    Only names and per-file digests are kept in memory; file contents are
    streamed. Links are recorded, never traversed, so a junction/symlink that
    points out of the tree cannot leak outside data into the digest.
    """
    with os.scandir(current) as iterator:
        entries = list(iterator)
    for entry in entries:
        rel_posix = _relative_posix(root, Path(entry.path))
        if _excluded(rel_posix, entry.name, patterns):
            continue
        if _is_link(entry):
            target = os.readlink(entry.path)
            resolved = Path(entry.path).resolve()
            if not _is_within(resolved, root):
                raise IntegrityError(
                    f"refusing to traverse {rel_posix!r}: link target escapes the tree root"
                )
            out.append((rel_posix, f"{_LINK_MARKER}{target}"))
        elif entry.is_dir(follow_symlinks=False):
            _collect_tree_entries(root, Path(entry.path), patterns, out)
        elif entry.is_file(follow_symlinks=False):
            out.append((rel_posix, _hash_file(Path(entry.path))))
        else:
            # Sockets/pipes/devices: never read them, but keep them visible to
            # the digest so their appearance or removal is still detected.
            out.append((rel_posix, _SPECIAL_MARKER))


class ContentAddressedStore:
    """Immutable sha256-addressed blob store rooted at ``root``.

    A blob is written atomically: a temporary file in the destination shard is
    ``fsync``-ed and then ``os.replace``-d, so readers only ever observe a
    complete blob. Reads verify the digest, and a missing or mismatching blob
    raises :class:`IntegrityError` -- the digest is the only contract, and it
    has to hold on the way out just as it did on the way in.
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    # ------------------------------------------------------------------ paths
    def path_for(self, digest: str) -> Path:
        """Sharded path for ``digest``; the blob is not required to exist."""
        checked = _validate_digest(digest)
        return self._root / checked[:2] / checked[2:4] / checked

    def exists(self, digest: str) -> bool:
        return self.path_for(digest).is_file()

    def verify(self, digest: str) -> bool:
        """True only when the stored bytes hash back to ``digest``."""
        path = self.path_for(digest)
        if not path.is_file():
            return False
        try:
            return _hash_file(path) == digest
        except OSError:
            return False

    # ------------------------------------------------------------------ writes
    def put_bytes(self, data: bytes) -> str:
        """Store ``data`` and return its sha256 hex digest."""
        digest = hashlib.sha256(data).hexdigest()
        if self.verify(digest):
            return digest
        self._write_blob_atomic(self.path_for(digest), (data,))
        return digest

    def put_json(self, obj: Any) -> str:
        """Store the canonical JSON encoding of ``obj`` and return its digest."""
        return self.put_bytes(_canonical_json_bytes(obj))

    def put_file(self, path: Path) -> str:
        """Store the contents of ``path`` and return their sha256 hex digest."""
        source = Path(path)
        if not source.is_file():
            raise ValueError(f"not a file: {source}")
        digest, staged = self._stage_stream(_iter_file_chunks(source))
        final = self.path_for(digest)
        if self.verify(digest):
            _unlink_quietly(staged)
            return digest
        final.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(staged, final)
        except BaseException:
            _unlink_quietly(staged)
            raise
        return digest

    def put_tree(self, directory: Path, *, exclude: Sequence[str] = ()) -> str:
        """Hash a directory deterministically and return the tree digest.

        The digest is the sha256 of the manifest text built by walking files
        sorted by relative POSIX path and appending ``relative/path\\0<sha256
        of contents>\\n`` for every file (the exact rule documented in
        ``contracts/SCHEMAS.md``'s snapshot section). Symlinks and junctions are
        recorded as ``relative/path\\0LINK:<target>\\n`` and never traversed; a
        link whose target resolves outside the tree raises
        :class:`IntegrityError`. Entries matching ``exclude`` (a bare name at
        any depth, or a POSIX path glob) are skipped entirely, which is how
        secrets stay out of snapshots.

        The manifest itself is stored as a blob, so ``get_bytes(tree_digest)``
        returns the exact listing that produced the digest.
        """
        root = Path(directory)
        if not root.is_dir():
            raise ValueError(f"not a directory: {root}")
        resolved_root = root.resolve()
        patterns = tuple(str(pattern) for pattern in exclude)
        records: list[tuple[str, str]] = []
        _collect_tree_entries(resolved_root, resolved_root, patterns, records)
        records.sort(key=lambda record: record[0])
        hasher = hashlib.sha256()
        manifest = bytearray()
        for rel_posix, marker in records:
            line = f"{rel_posix}\0{marker}\n".encode("utf-8", "surrogateescape")
            hasher.update(line)
            manifest.extend(line)
        tree_digest = hasher.hexdigest()
        self.put_bytes(bytes(manifest))
        return tree_digest

    # ------------------------------------------------------------------- reads
    def get_bytes(self, digest: str) -> bytes:
        """Return the stored bytes, re-hashed against ``digest``."""
        path = self.path_for(digest)
        try:
            data = path.read_bytes()
        except FileNotFoundError as exc:
            raise IntegrityError(f"cas blob is missing: {digest}") from exc
        actual = hashlib.sha256(data).hexdigest()
        if actual != digest:
            raise IntegrityError(
                f"cas blob {digest} is tampered: stored bytes hash to {actual}"
            )
        return data

    def get_json(self, digest: str) -> Any:
        """Return the parsed JSON payload of a blob."""
        raw = self.get_bytes(digest)
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise IntegrityError(f"cas blob {digest} is not valid UTF-8 JSON") from exc

    # ---------------------------------------------------------------- internal
    def _write_blob_atomic(self, final: Path, chunks: Iterable[bytes]) -> None:
        """Write in a temp file next to ``final``, fsync, then atomically replace."""
        final.parent.mkdir(parents=True, exist_ok=True)
        temp = final.parent / f".{final.name}.{uuid.uuid4().hex}.tmp"
        try:
            with open(temp, "wb") as handle:
                for chunk in chunks:
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, final)
        except BaseException:
            _unlink_quietly(temp)
            raise

    def _stage_stream(self, chunks: Iterable[bytes]) -> tuple[str, Path]:
        """Stream to the staging area while hashing; the shard is unknown until the end."""
        staging = self._root / _STAGING_DIR
        staging.mkdir(parents=True, exist_ok=True)
        temp = staging / f"tmp-{uuid.uuid4().hex}"
        hasher = hashlib.sha256()
        try:
            with open(temp, "wb") as handle:
                for chunk in chunks:
                    hasher.update(chunk)
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            _unlink_quietly(temp)
            raise
        return hasher.hexdigest(), temp
