"""SQLite (WAL) registry: the durable, single-writer record of a run.

One table per frozen DTO of :mod:`dream_rsi.domain.models`. Each table keeps a
handful of indexed scalar columns for filtering plus a ``payload`` column that
holds the canonical JSON of ``to_dict()``; rows are rebuilt through the
contract's ``from_dict`` classmethods. There is deliberately no second,
hand-written column mapper that could drift away from the frozen contract, and
unknown keys are simply ignored by ``from_dict``.

Commit ordering (see ``docs/SECURITY.md``): CAS temp write ->
hash verify -> atomic rename -> *this* SQLite transaction. Wrap a whole batch
(nodes + attempt + event) in a single :meth:`Registry.transaction` so a crash
or a kill can never commit half a batch; a world that ends up half-written is
quarantined instead of being scored.

Parent rows are inserted before children: run -> world -> {node, attempt} ->
policy -> {replay, score} -> revision -> event. Foreign keys are enforced, so a
child row for a missing parent is rejected as an integrity failure rather than
silently creating an orphan.

Credentials are never stored here; payloads are JSON, never pickle.
"""
from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import Any, TypeVar

from ..domain.errors import IntegrityError
from ..domain.models import (
    SCHEMA_VERSION,
    Attempt,
    AttemptState,
    Event,
    Node,
    PolicyState,
    PolicyVersion,
    ReplayEpisode,
    Revision,
    Run,
    RunState,
    ScoreRecord,
    Serializable,
    World,
    WorldState,
    utcnow_iso,
)

__all__ = ["SEALED_WORLD_STATES", "Registry"]

_T = TypeVar("_T")

_BUSY_TIMEOUT_MS = 10_000

_TABLES: tuple[str, ...] = (
    "runs",
    "worlds",
    "nodes",
    "attempts",
    "policies",
    "replays",
    "scores",
    "revisions",
    "events",
)

#: World states that a sealed world may legitimately be in.
SEALED_WORLD_STATES: tuple[str, str] = (
    WorldState.SEALED_SCOREABLE.value,
    WorldState.SEALED_UNSCOREABLE.value,
)

_SCHEMA: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS runs (
        run_id     TEXT PRIMARY KEY,
        state      TEXT NOT NULL,
        mode       TEXT NOT NULL,
        created_at TEXT NOT NULL,
        payload    TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_runs_state ON runs(state)",
    """
    CREATE TABLE IF NOT EXISTS worlds (
        world_id    TEXT PRIMARY KEY,
        run_id      TEXT NOT NULL REFERENCES runs(run_id),
        outer_t     INTEGER NOT NULL,
        state       TEXT NOT NULL,
        created_at  TEXT NOT NULL,
        payload     TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_worlds_run ON worlds(run_id, outer_t)",
    "CREATE INDEX IF NOT EXISTS idx_worlds_state ON worlds(state)",
    """
    CREATE TABLE IF NOT EXISTS nodes (
        node_id        TEXT PRIMARY KEY,
        world_id       TEXT NOT NULL REFERENCES worlds(world_id),
        parent_id      TEXT,
        creation_index INTEGER NOT NULL,
        round_index    INTEGER NOT NULL,
        depth          INTEGER NOT NULL,
        payload        TEXT NOT NULL,
        UNIQUE (world_id, creation_index)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_nodes_world_creation ON nodes(world_id, creation_index)",
    "CREATE INDEX IF NOT EXISTS idx_nodes_parent ON nodes(parent_id)",
    """
    CREATE TABLE IF NOT EXISTS attempts (
        attempt_id     TEXT PRIMARY KEY,
        world_id       TEXT NOT NULL REFERENCES worlds(world_id),
        parent_id      TEXT NOT NULL,
        round_index    INTEGER NOT NULL,
        state          TEXT NOT NULL,
        invocation_id  TEXT NOT NULL,
        created_at     TEXT NOT NULL,
        updated_at     TEXT NOT NULL,
        payload        TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_attempts_world ON attempts(world_id, round_index)",
    # One invocation_id is one real call: the unique index is what makes a
    # duplicated response, or a resumed batch, impossible to double-account.
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_attempts_invocation ON attempts(invocation_id)",
    """
    CREATE TABLE IF NOT EXISTS policies (
        version_id  TEXT PRIMARY KEY,
        run_id      TEXT NOT NULL REFERENCES runs(run_id),
        iteration_t INTEGER NOT NULL,
        m           INTEGER NOT NULL,
        state       TEXT NOT NULL,
        origin      TEXT NOT NULL,
        created_at  TEXT NOT NULL,
        payload     TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_policies_run_iteration ON policies(run_id, iteration_t, m)",
    """
    CREATE TABLE IF NOT EXISTS replays (
        episode_id        TEXT PRIMARY KEY,
        world_id          TEXT NOT NULL REFERENCES worlds(world_id),
        policy_version_id TEXT NOT NULL REFERENCES policies(version_id),
        rho               REAL NOT NULL,
        payload           TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_replays_world ON replays(world_id)",
    "CREATE INDEX IF NOT EXISTS idx_replays_policy ON replays(policy_version_id)",
    "CREATE INDEX IF NOT EXISTS idx_replays_triple"
    " ON replays(world_id, policy_version_id, rho)",
    """
    CREATE TABLE IF NOT EXISTS scores (
        policy_version_id TEXT NOT NULL REFERENCES policies(version_id),
        world_id          TEXT NOT NULL REFERENCES worlds(world_id),
        b_count           INTEGER NOT NULL,
        s_best            REAL NOT NULL,
        value             REAL NOT NULL,
        created_at        TEXT NOT NULL,
        payload           TEXT NOT NULL,
        PRIMARY KEY (policy_version_id, world_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_scores_world ON scores(world_id)",
    """
    CREATE TABLE IF NOT EXISTS revisions (
        run_id          TEXT NOT NULL,
        iteration_t     INTEGER NOT NULL,
        m               INTEGER NOT NULL,
        input_policy_id TEXT NOT NULL,
        status          TEXT NOT NULL,
        created_at      TEXT NOT NULL,
        payload         TEXT NOT NULL,
        PRIMARY KEY (run_id, iteration_t, m)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS events (
        seq     INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id  TEXT NOT NULL REFERENCES runs(run_id),
        kind    TEXT NOT NULL,
        at      TEXT NOT NULL,
        payload TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, seq)",
)


def _canonical_json(dto: Serializable) -> str:
    """Canonical JSON text used for every stored payload."""
    return json.dumps(
        dto.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _decode(
    payload: str,
    loader: Callable[[Mapping[str, Any]], _T],
    *,
    context: str,
    expected: Mapping[str, Any] | None = None,
) -> _T:
    """Rebuild a DTO, or raise ``IntegrityError``; never returns a half-built object."""
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise IntegrityError(f"{context}: stored payload is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise IntegrityError(f"{context}: stored payload is not a JSON object")
    version = data.get("schema_version")
    if version is not None and version != SCHEMA_VERSION:
        raise IntegrityError(
            f"{context}: stored schema_version {version!r} is not supported "
            f"(this build understands {SCHEMA_VERSION})"
        )
    try:
        dto = loader(data)
    except (TypeError, ValueError, KeyError) as exc:
        raise IntegrityError(f"{context}: stored payload is not a valid DTO: {exc}") from exc
    for name, column_value in (expected or {}).items():
        if not hasattr(dto, name):
            continue
        actual: Any = getattr(dto, name)
        if isinstance(actual, Enum):
            actual = actual.value
        if actual != column_value:
            raise IntegrityError(
                f"{context}: column {name}={column_value!r} disagrees with the stored payload "
                f"({actual!r}); the row was edited outside the registry"
            )
    return dto


class Registry:
    """Durable record of runs, worlds, nodes, attempts, policies and events.

    The connection runs in WAL mode with ``foreign_keys=ON``, a busy timeout and
    ``synchronous=NORMAL``: a concurrent reader (another process, or another
    instance in this one) never blocks the writer and never sees a torn commit.
    Writes are serialised through :meth:`transaction`, which commits on clean
    exit and rolls back on any exception.
    """

    def __init__(self, db_path: Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._depth = 0
        self._closed = False
        self._conn = sqlite3.connect(
            str(self._path),
            timeout=_BUSY_TIMEOUT_MS / 1000.0,
            isolation_level=None,  # BEGIN/COMMIT are explicit below
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        found = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
        if found > SCHEMA_VERSION:
            self._conn.close()
            raise IntegrityError(
                f"{self._path}: database schema version {found} is newer than this build "
                f"({SCHEMA_VERSION})"
            )
        # An intact database is opened without taking the write lock, so a
        # reader can open the registry while another connection is mid-write.
        if found < SCHEMA_VERSION or not self._tables_present():
            with self.transaction():
                for statement in _SCHEMA:
                    self._conn.execute(statement)
            self._conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    # ------------------------------------------------------------- lifecycle
    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._depth > 0:
                with contextlib.suppress(sqlite3.Error):
                    self._conn.execute("ROLLBACK")
                self._depth = 0
            self._conn.close()

    def __enter__(self) -> Registry:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Single-writer commit boundary; re-entrant within one thread."""
        self._require_open()
        with self._lock:
            if self._depth == 0:
                self._conn.execute("BEGIN IMMEDIATE")
            self._depth += 1
            try:
                yield
            except BaseException:
                self._depth -= 1
                if self._depth == 0:
                    with contextlib.suppress(sqlite3.Error):
                        self._conn.execute("ROLLBACK")
                raise
            self._depth -= 1
            if self._depth == 0:
                try:
                    self._conn.execute("COMMIT")
                except sqlite3.Error:
                    # A failed COMMIT leaves the transaction open; release it so
                    # the next BEGIN on this connection can still start.
                    with contextlib.suppress(sqlite3.Error):
                        self._conn.execute("ROLLBACK")
                    raise

    # ------------------------------------------------------------------ runs
    def create_run(self, run: Run) -> None:
        self._insert(
            "runs",
            ("run_id", "state", "mode", "created_at"),
            (run.run_id, run.state.value, run.mode.value, run.created_at),
            _canonical_json(run),
        )

    def update_run_state(
        self,
        run_id: str,
        state: RunState,
        *,
        note: str | None = None,
        finished: bool = False,
    ) -> None:
        """Record a run transition; ``finished=True`` stamps ``finished_at``.

        The first transition to ``ONLINE_RUNNING`` stamps ``started_at`` when it
        is still unset, so the run carries a real start time. Legal transition
        order is the Controller's business, not the registry's.
        """
        with self.transaction():
            run = self._require_run(run_id)
            changes: dict[str, Any] = {"state": state}
            if note is not None:
                changes["status_note"] = note
            if state is RunState.ONLINE_RUNNING and run.started_at is None:
                changes["started_at"] = utcnow_iso()
            if finished:
                changes["finished_at"] = utcnow_iso()
            self._write_update(
                "runs",
                "run_id",
                run_id,
                replace(run, **changes),
                {"state": state.value},
            )

    def get_run(self, run_id: str) -> Run | None:
        row = self._fetch_one("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        if row is None:
            return None
        return _decode(
            str(row["payload"]),
            Run.from_dict,
            context=f"runs[{run_id}]",
            expected={"run_id": row["run_id"], "state": row["state"]},
        )

    def list_runs(self) -> list[Run]:
        return [self._decode_run(row) for row in self._fetch_all("SELECT * FROM runs ORDER BY rowid")]

    # ---------------------------------------------------------------- worlds
    def add_world(self, world: World) -> None:
        self._insert(
            "worlds",
            ("world_id", "run_id", "outer_t", "state", "created_at"),
            (world.world_id, world.run_id, world.outer_t, world.state.value, world.created_at),
            _canonical_json(world),
        )

    def seal_world(
        self,
        world_id: str,
        *,
        state: WorldState,
        world_digest: str,
        b_count: int,
        s_best: float,
        note: str | None = None,
    ) -> None:
        """Seal a world with the digest and the (B_i, S_i) pair it was built with.

        Re-sealing with exactly the same values is a no-op, so a resumed run can
        repeat the write idempotently; re-sealing with different values is an
        integrity failure, because that would be rewriting history.
        """
        if state.value not in SEALED_WORLD_STATES:
            raise ValueError(f"seal_world requires a sealed state, got {state.value!r}")
        with self.transaction():
            world = self._require_world(world_id)
            if world.state is WorldState.QUARANTINED:
                raise IntegrityError(f"world {world_id} is quarantined and cannot be sealed")
            if world.state.value in SEALED_WORLD_STATES:
                same = (
                    world.state == state
                    and world.world_digest == world_digest
                    and world.b_count == b_count
                    and world.s_best == s_best
                )
                if same:
                    return
                raise IntegrityError(f"world {world_id} is already sealed with different values")
            self._write_update(
                "worlds",
                "world_id",
                world_id,
                replace(
                    world,
                    state=state,
                    world_digest=world_digest,
                    b_count=b_count,
                    s_best=s_best,
                    sealed_at=utcnow_iso(),
                    seal_note=note,
                ),
                {"state": state.value},
            )

    def get_world(self, world_id: str) -> World | None:
        row = self._fetch_one("SELECT * FROM worlds WHERE world_id = ?", (world_id,))
        if row is None:
            return None
        return self._decode_world(row)

    def list_worlds(self, run_id: str, *, sealed_only: bool = False) -> list[World]:
        """Worlds of a run in creation order; ``sealed_only`` keeps scoreable states."""
        sql = "SELECT * FROM worlds WHERE run_id = ?"
        params: list[Any] = [run_id]
        if sealed_only:
            sql += " AND state IN (?, ?)"
            params.extend(SEALED_WORLD_STATES)
        sql += " ORDER BY rowid"
        return [self._decode_world(row) for row in self._fetch_all(sql, params)]

    def quarantine_world(self, world_id: str, reason: str) -> None:
        """Mark a world unusable (half-written batch, crash, tamper), keeping the audit trail."""
        with self.transaction():
            world = self._require_world(world_id)
            if world.state is WorldState.QUARANTINED:
                return
            note = f"QUARANTINED: {reason}"
            if world.seal_note:
                note = f"{world.seal_note} | {note}"
            self._write_update(
                "worlds",
                "world_id",
                world_id,
                replace(
                    world,
                    state=WorldState.QUARANTINED,
                    seal_note=note,
                    sealed_at=world.sealed_at or utcnow_iso(),
                ),
                {"state": WorldState.QUARANTINED.value},
            )

    # ----------------------------------------------------------------- nodes
    def add_node(self, node: Node) -> None:
        self._insert_node(node)

    def add_nodes(self, nodes: Sequence[Node]) -> None:
        """Insert a whole batch in one transaction: all of it, or none of it."""
        with self.transaction():
            for node in nodes:
                self._insert_node(node)

    def get_node(self, node_id: str) -> Node | None:
        row = self._fetch_one("SELECT * FROM nodes WHERE node_id = ?", (node_id,))
        if row is None:
            return None
        return self._decode_node(row)

    def list_nodes(self, world_id: str) -> list[Node]:
        """Nodes of a world in global ``creation_index`` order (replay order)."""
        return [
            self._decode_node(row)
            for row in self._fetch_all(
                "SELECT * FROM nodes WHERE world_id = ? ORDER BY creation_index", (world_id,)
            )
        ]

    # -------------------------------------------------------------- attempts
    def add_attempt(self, attempt: Attempt) -> None:
        self._insert(
            "attempts",
            (
                "attempt_id",
                "world_id",
                "parent_id",
                "round_index",
                "state",
                "invocation_id",
                "created_at",
                "updated_at",
            ),
            (
                attempt.attempt_id,
                attempt.world_id,
                attempt.parent_id,
                attempt.round_index,
                attempt.state.value,
                attempt.invocation_id,
                attempt.created_at,
                attempt.updated_at,
            ),
            _canonical_json(attempt),
        )

    def update_attempt_state(
        self,
        attempt_id: str,
        state: AttemptState,
        *,
        failure_reason: str | None = None,
        workspace_snapshot_digest: str | None = None,
    ) -> None:
        with self.transaction():
            row = self._fetch_one("SELECT * FROM attempts WHERE attempt_id = ?", (attempt_id,))
            if row is None:
                raise IntegrityError(f"attempt not found: {attempt_id}")
            attempt = self._decode_attempt(row)
            changes: dict[str, Any] = {"state": state, "updated_at": utcnow_iso()}
            if failure_reason is not None:
                changes["failure_reason"] = failure_reason
            if workspace_snapshot_digest is not None:
                changes["workspace_snapshot_digest"] = workspace_snapshot_digest
            updated = replace(attempt, **changes)
            self._write_update(
                "attempts",
                "attempt_id",
                attempt_id,
                updated,
                {"state": state.value, "updated_at": updated.updated_at},
            )

    def get_attempt(self, attempt_id: str) -> Attempt | None:
        row = self._fetch_one("SELECT * FROM attempts WHERE attempt_id = ?", (attempt_id,))
        if row is None:
            return None
        return self._decode_attempt(row)

    def list_attempts(self, world_id: str) -> list[Attempt]:
        return [
            self._decode_attempt(row)
            for row in self._fetch_all(
                "SELECT * FROM attempts WHERE world_id = ? ORDER BY rowid", (world_id,)
            )
        ]

    # -------------------------------------------------------------- policies
    def add_policy(self, pv: PolicyVersion) -> None:
        self._insert(
            "policies",
            ("version_id", "run_id", "iteration_t", "m", "state", "origin", "created_at"),
            (pv.version_id, pv.run_id, pv.iteration_t, pv.m, pv.state.value, pv.origin, pv.created_at),
            _canonical_json(pv),
        )

    def update_policy_state(
        self,
        version_id: str,
        state: PolicyState,
        *,
        static_validation_status: str | None = None,
        sandbox_validation_status: str | None = None,
        failure_reason: str | None = None,
    ) -> None:
        with self.transaction():
            row = self._fetch_one("SELECT * FROM policies WHERE version_id = ?", (version_id,))
            if row is None:
                raise IntegrityError(f"policy version not found: {version_id}")
            policy = self._decode_policy(row)
            changes: dict[str, Any] = {"state": state}
            if static_validation_status is not None:
                changes["static_validation_status"] = static_validation_status
            if sandbox_validation_status is not None:
                changes["sandbox_validation_status"] = sandbox_validation_status
            if failure_reason is not None:
                changes["failure_reason"] = failure_reason
            self._write_update(
                "policies", "version_id", version_id, replace(policy, **changes),
                {"state": state.value},
            )

    def get_policy(self, version_id: str) -> PolicyVersion | None:
        row = self._fetch_one("SELECT * FROM policies WHERE version_id = ?", (version_id,))
        if row is None:
            return None
        return self._decode_policy(row)

    def list_policies(self, run_id: str) -> list[PolicyVersion]:
        return [
            self._decode_policy(row)
            for row in self._fetch_all(
                "SELECT * FROM policies WHERE run_id = ? ORDER BY rowid", (run_id,)
            )
        ]

    # --------------------------------------------------------------- replays
    def add_replay(self, ep: ReplayEpisode) -> None:
        self._insert(
            "replays",
            ("episode_id", "world_id", "policy_version_id", "rho"),
            (ep.episode_id, ep.world_id, ep.policy_version_id, ep.rho),
            _canonical_json(ep),
        )

    def list_replays(
        self,
        *,
        world_id: str | None = None,
        policy_version_id: str | None = None,
    ) -> list[ReplayEpisode]:
        clauses: list[str] = []
        params: list[Any] = []
        if world_id is not None:
            clauses.append("world_id = ?")
            params.append(world_id)
        if policy_version_id is not None:
            clauses.append("policy_version_id = ?")
            params.append(policy_version_id)
        sql = "SELECT * FROM replays"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY rowid"
        return [
            _decode(
                str(row["payload"]),
                ReplayEpisode.from_dict,
                context=f"replays[{row['episode_id']}]",
                expected={
                    "episode_id": row["episode_id"],
                    "world_id": row["world_id"],
                    "policy_version_id": row["policy_version_id"],
                    "rho": row["rho"],
                },
            )
            for row in self._fetch_all(sql, params)
        ]

    # ---------------------------------------------------------------- scores
    def add_score(self, sr: ScoreRecord) -> None:
        self._insert(
            "scores",
            ("policy_version_id", "world_id", "b_count", "s_best", "value", "created_at"),
            (sr.policy_version_id, sr.world_id, sr.b_count, sr.s_best, sr.value, sr.created_at),
            _canonical_json(sr),
        )

    def list_scores(
        self,
        *,
        policy_version_id: str | None = None,
        world_id: str | None = None,
    ) -> list[ScoreRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if policy_version_id is not None:
            clauses.append("policy_version_id = ?")
            params.append(policy_version_id)
        if world_id is not None:
            clauses.append("world_id = ?")
            params.append(world_id)
        sql = "SELECT * FROM scores"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY rowid"
        return [
            _decode(
                str(row["payload"]),
                ScoreRecord.from_dict,
                context=f"scores[{row['policy_version_id']},{row['world_id']}]",
                expected={
                    "policy_version_id": row["policy_version_id"],
                    "world_id": row["world_id"],
                    "b_count": row["b_count"],
                },
            )
            for row in self._fetch_all(sql, params)
        ]

    # ------------------------------------------------------------- revisions
    def add_revision(self, rev: Revision) -> None:
        """Store one revision.

        ``Revision`` carries no ``run_id``, so it is derived from the input
        policy (falling back to the output policy) at insert time and kept as an
        indexed column; it is the empty string when neither policy is registered
        yet.
        """
        with self.transaction():
            run_id = self._revision_run_id(rev)
            self._insert(
                "revisions",
                ("run_id", "iteration_t", "m", "input_policy_id", "status", "created_at"),
                (run_id, rev.iteration_t, rev.m, rev.input_policy_id, rev.status, rev.created_at),
                _canonical_json(rev),
            )

    def list_revisions(self, run_id: str, iteration_t: int | None = None) -> list[Revision]:
        sql = "SELECT * FROM revisions WHERE run_id = ?"
        params: list[Any] = [run_id]
        if iteration_t is not None:
            sql += " AND iteration_t = ?"
            params.append(iteration_t)
        sql += " ORDER BY iteration_t, m, rowid"
        return [
            _decode(
                str(row["payload"]),
                Revision.from_dict,
                context=f"revisions[{row['run_id']},{row['iteration_t']},{row['m']}]",
                expected={"iteration_t": row["iteration_t"], "m": row["m"]},
            )
            for row in self._fetch_all(sql, params)
        ]

    # ---------------------------------------------------------------- events
    def append_event(self, event: Event) -> None:
        self._insert(
            "events",
            ("run_id", "kind", "at"),
            (event.run_id, event.kind, event.at),
            _canonical_json(event),
        )

    def list_events(self, run_id: str) -> list[Event]:
        """Events of a run in append order (the log is the audit trail)."""
        return [
            _decode(
                str(row["payload"]),
                Event.from_dict,
                context=f"events[{row['seq']}]",
                expected={"run_id": row["run_id"], "kind": row["kind"]},
            )
            for row in self._fetch_all(
                "SELECT * FROM events WHERE run_id = ? ORDER BY seq", (run_id,)
            )
        ]

    # -------------------------------------------------------------- internals
    def _tables_present(self) -> bool:
        rows = self._conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        present = {str(row["name"]) for row in rows}
        return all(name in present for name in _TABLES)

    def _require_open(self) -> sqlite3.Connection:
        if self._closed:
            raise RuntimeError("registry is closed")
        return self._conn

    def _fetch_one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        row: sqlite3.Row | None = self._require_open().execute(sql, params).fetchone()
        return row

    def _fetch_all(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        return list(self._require_open().execute(sql, params).fetchall())

    def _insert(
        self,
        table: str,
        columns: Sequence[str],
        values: Sequence[Any],
        payload: str,
    ) -> None:
        names = ", ".join((*columns, "payload"))
        placeholders = ", ".join("?" for _ in range(len(columns) + 1))
        with self.transaction():
            try:
                self._require_open().execute(
                    f"INSERT INTO {table} ({names}) VALUES ({placeholders})",
                    (*values, payload),
                )
            except sqlite3.IntegrityError as exc:
                raise IntegrityError(f"registry insert into {table} rejected: {exc}") from exc

    def _insert_node(self, node: Node) -> None:
        self._insert(
            "nodes",
            ("node_id", "world_id", "parent_id", "creation_index", "round_index", "depth"),
            (
                node.node_id,
                node.world_id,
                node.parent_id,
                node.creation_index,
                node.round_index,
                node.depth,
            ),
            _canonical_json(node),
        )

    def _write_update(
        self,
        table: str,
        key_column: str,
        key_value: str,
        updated: Serializable,
        columns: Mapping[str, Any],
    ) -> None:
        """Rewrite a row from ``updated`` and keep the indexed columns in sync."""
        assignments = ", ".join(f"{name} = ?" for name in columns)
        params: list[Any] = [*columns.values(), _canonical_json(updated), key_value]
        try:
            self._require_open().execute(
                f"UPDATE {table} SET {assignments}, payload = ? WHERE {key_column} = ?",
                params,
            )
        except sqlite3.IntegrityError as exc:
            raise IntegrityError(f"registry update of {table} rejected: {exc}") from exc

    def _require_run(self, run_id: str) -> Run:
        run = self.get_run(run_id)
        if run is None:
            raise IntegrityError(f"run not found: {run_id}")
        return run

    def _require_world(self, world_id: str) -> World:
        world = self.get_world(world_id)
        if world is None:
            raise IntegrityError(f"world not found: {world_id}")
        return world

    def _decode_run(self, row: sqlite3.Row) -> Run:
        return _decode(
            str(row["payload"]),
            Run.from_dict,
            context=f"runs[{row['run_id']}]",
            expected={"run_id": row["run_id"], "state": row["state"]},
        )

    def _decode_world(self, row: sqlite3.Row) -> World:
        return _decode(
            str(row["payload"]),
            World.from_dict,
            context=f"worlds[{row['world_id']}]",
            expected={
                "world_id": row["world_id"],
                "run_id": row["run_id"],
                "outer_t": row["outer_t"],
                "state": row["state"],
            },
        )

    def _decode_node(self, row: sqlite3.Row) -> Node:
        return _decode(
            str(row["payload"]),
            Node.from_dict,
            context=f"nodes[{row['node_id']}]",
            expected={
                "node_id": row["node_id"],
                "world_id": row["world_id"],
                "parent_id": row["parent_id"],
                "creation_index": row["creation_index"],
            },
        )

    def _decode_attempt(self, row: sqlite3.Row) -> Attempt:
        return _decode(
            str(row["payload"]),
            Attempt.from_dict,
            context=f"attempts[{row['attempt_id']}]",
            expected={
                "attempt_id": row["attempt_id"],
                "world_id": row["world_id"],
                "invocation_id": row["invocation_id"],
                "state": row["state"],
            },
        )

    def _decode_policy(self, row: sqlite3.Row) -> PolicyVersion:
        return _decode(
            str(row["payload"]),
            PolicyVersion.from_dict,
            context=f"policies[{row['version_id']}]",
            expected={
                "version_id": row["version_id"],
                "run_id": row["run_id"],
                "state": row["state"],
            },
        )

    def _revision_run_id(self, rev: Revision) -> str:
        for version_id in (rev.input_policy_id, rev.output_policy_id):
            if not version_id:
                continue
            row = self._fetch_one("SELECT run_id FROM policies WHERE version_id = ?", (version_id,))
            if row is not None:
                return str(row["run_id"])
        return ""
