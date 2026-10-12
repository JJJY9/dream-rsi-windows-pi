"""The packing demo task: fixed harness, fixture ladder, and a tiny offline loop.

Three layers, each real rather than simulated:

* the **fixed evaluator harness** grades actual candidate files (including the
  shipped initial workspace, so the documented "0.2 lattice" claim is pinned);
* the **fixture ladder** climbs from the lattice to the closed-form brick
  through real runnable code that the same harness scores;
* one **offline loop** runs the full composition root on this task, proving the
  new task is wired end to end (registry, sealing, replay, selection) rather
  than merely existing on disk.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:  # pragma: no cover - import bootstrapping
    sys.path.insert(0, str(SRC))

from dream_rsi.adapters.base import TaskSpec  # noqa: E402
from dream_rsi.adapters.fixture import (  # noqa: E402
    FixtureDiscoveryBackend,
    recipe_for,
    render_candidate,
)
from dream_rsi.cli import _build_controller  # noqa: E402
from dream_rsi.config import Config, LoopConfig  # noqa: E402
from dream_rsi.domain.models import RunState, WorldState  # noqa: E402

TASK_DIR = REPO / "tasks" / "packing"
GRID_SCORE = 0.2
BRICK_SCORE = (8.0 - (1.6) ** 0.5) / 30.0
SHIFT_SCORE = 0.21


def _load_harness() -> object:
    path = TASK_DIR / "evaluator_harness.py"
    spec = importlib.util.spec_from_file_location("packing_evaluator_harness", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HARNESS = _load_harness()


def _grade(tmp_path: Path, source: str, name: str = "cand") -> dict:
    candidate = tmp_path / name
    candidate.mkdir()
    (candidate / "candidate.py").write_text(source, encoding="utf-8")
    return HARNESS.run(candidate)  # type: ignore[attr-defined]


class TestHarness:
    def test_the_shipped_initial_workspace_scores_the_documented_lattice(self) -> None:
        result = HARNESS.run(TASK_DIR / "initial")  # type: ignore[attr-defined]
        assert result["status"] == "OK"
        assert result["score"] == pytest.approx(GRID_SCORE, abs=1e-9)

    def test_the_brick_construction_scores_its_analytic_value(self, tmp_path: Path) -> None:
        source = render_candidate("brick", recipe_for("packing"))
        result = _grade(tmp_path, source)
        assert result["status"] == "OK"
        assert result["score"] == pytest.approx(BRICK_SCORE, abs=1e-9)

    def test_the_mid_ladder_construction_scores_between_grid_and_brick(
        self, tmp_path: Path
    ) -> None:
        source = render_candidate("wide_shift", recipe_for("packing"))
        result = _grade(tmp_path, source)
        assert result["score"] == pytest.approx(SHIFT_SCORE, abs=1e-9)

    @pytest.mark.parametrize(
        "source",
        [
            # 29 points instead of 30.
            "def construct():\n    return [(0.1, 0.1)] * 29\n",
            # A coordinate outside the unit square.
            "def construct():\n    pts = [(0.5, 0.5)] * 30\n    pts[0] = (1.2, 0.5)\n    return pts\n",
            # A non-finite coordinate.
            "def construct():\n    pts = [(0.5, 0.5)] * 30\n    pts[0] = (float('nan'), 0.5)\n    return pts\n",
            # A non-numeric coordinate.
            "def construct():\n    pts = [(0.5, 0.5)] * 30\n    pts[0] = ('x', 0.5)\n    return pts\n",
            # Not an iterable of pairs at all.
            "def construct():\n    return 42\n",
        ],
    )
    def test_constraint_violations_score_a_measured_zero(
        self, tmp_path: Path, source: str
    ) -> None:
        result = _grade(tmp_path, source)
        assert result["status"] == "OK"
        assert result["score"] == 0.0
        assert result["error_type"] == "CONSTRAINT_VIOLATION"

    def test_duplicate_points_score_zero_without_being_an_error(self, tmp_path: Path) -> None:
        """Duplicates violate nothing; they simply measure a minimum of 0."""
        result = _grade(tmp_path, "def construct():\n    return [(0.5, 0.5)] * 30\n")
        assert result["status"] == "OK"
        assert result["score"] == 0.0
        assert result["error_type"] is None

    def test_a_raising_construct_is_an_evaluator_error_not_a_zero(self, tmp_path: Path) -> None:
        result = _grade(tmp_path, "def construct():\n    raise RuntimeError('boom')\n")
        assert result["status"] == "ERROR"
        assert result["error_type"] == "RuntimeError"

    def test_a_missing_construct_is_an_evaluator_error(self, tmp_path: Path) -> None:
        result = _grade(tmp_path, "def other():\n    return []\n")
        assert result["status"] == "ERROR"

    def test_the_result_payload_is_json_serialisable(self, tmp_path: Path) -> None:
        """main() writes this to disk; it must round-trip through json."""
        result = _grade(tmp_path, render_candidate("brick", recipe_for("packing")))
        assert json.loads(json.dumps(result))["points_digest"] == result["points_digest"]


class TestFixtureLadder:
    def _task(self) -> TaskSpec:
        return TaskSpec(
            task_id="packing",
            entry_filename="candidate.py",
            instructions="",
            seed_workspace=TASK_DIR / "initial",
        )

    def _generate(self, tmp_path: Path, parent: Path, output_name: str) -> tuple[Path, object]:
        backend = FixtureDiscoveryBackend()
        output = tmp_path / output_name
        attempt = backend.generate(
            task=self._task(),
            parent_workspace=parent,
            history=(),
            output_workspace=output,
            invocation_id=f"gen-{output_name}",
        )
        return output, attempt

    def test_the_ladder_advances_lattice_to_shift_to_brick(self, tmp_path: Path) -> None:
        first, attempt1 = self._generate(tmp_path, TASK_DIR / "initial", "gen1")
        assert attempt1.extra["strategy"] == "wide_shift"  # type: ignore[attr-defined]
        second, attempt2 = self._generate(tmp_path, first, "gen2")
        assert attempt2.extra["strategy"] == "brick"  # type: ignore[attr-defined]
        # At the top of the ladder the candidate is reproduced unchanged.
        third, attempt3 = self._generate(tmp_path, second, "gen3")
        assert (third / "candidate.py").read_bytes() == (second / "candidate.py").read_bytes()
        assert "unchanged" in attempt3.summary  # type: ignore[attr-defined]

    def test_the_ladder_scores_are_strictly_increasing_on_the_real_harness(
        self, tmp_path: Path
    ) -> None:
        scores = []
        for name in ("wide_shift", "brick"):
            result = _grade(tmp_path, render_candidate(name, recipe_for("packing")), name=name)
            scores.append(result["score"])
        assert GRID_SCORE < scores[0] < scores[1]


class TestOfflineLoopOnTheNewTask:
    def test_the_full_loop_commits_and_every_world_reaches_the_brick(
        self, tmp_path: Path
    ) -> None:
        run_id = "packing-offline-1"
        config = Config(
            project_root=tmp_path,
            task_dir=TASK_DIR,
            incumbent_policy_path=REPO / "policies" / "incumbent.py",
            backend="fixture",
            loop=LoopConfig(outer_rounds=2),
            run_id=run_id,
        )
        config.ensure_dirs(run_id)
        controller, registry = _build_controller(config, run_id, with_developer=True)
        try:
            outcome = controller.run()
            worlds = registry.list_worlds(run_id)
        finally:
            registry.close()

        assert outcome.state is RunState.COMMITTED
        assert [w.outer_t for w in worlds] == [0, 1]
        for world in worlds:
            assert world.state is WorldState.SEALED_SCOREABLE
            assert world.s_best == pytest.approx(BRICK_SCORE, abs=1e-9)
            assert world.b_count > 0
