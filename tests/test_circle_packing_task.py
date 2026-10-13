"""The circle-packing task: fixed harness, fixture ladder, and a tiny offline loop.

Follows the AlphaEvolve/OpenEvolve n=26 benchmark metric (sum of radii, 1e-6
tolerances). Three layers, each real rather than simulated: harness unit tests
on actual candidate code, a fixture ladder through real graded arrangements,
and one offline loop through the full composition root.
"""
from __future__ import annotations

import importlib.util
import json
import math
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

TASK_DIR = REPO / "tasks" / "circle_packing"
TARGET = 2.635983  # published AlphaEvolve result for n=26 (score divisor)
RINGS_SCORE = 0.9574142241742692 / TARGET
HEX_ROWS_SCORE = 2.34 / TARGET
HEX_MAX_SCORE = (26.0 / (2.0 + 5.0 * math.sqrt(3.0))) / TARGET  # ~= 0.9253


def _load_harness() -> object:
    path = TASK_DIR / "evaluator_harness.py"
    spec = importlib.util.spec_from_file_location("circle_packing_harness", path)
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
    def test_the_shipped_initial_workspace_scores_its_ring_arrangement(self) -> None:
        result = HARNESS.run(TASK_DIR / "initial")  # type: ignore[attr-defined]
        assert result["status"] == "OK"
        assert result["score"] == pytest.approx(RINGS_SCORE, abs=1e-9)

    def test_the_fixture_rungs_score_their_analytic_values(self, tmp_path: Path) -> None:
        rows = _grade(tmp_path, render_candidate("hex_rows", recipe_for("circle_packing")), "rows")
        top = _grade(tmp_path, render_candidate("hex_max", recipe_for("circle_packing")), "max")
        assert rows["score"] == pytest.approx(HEX_ROWS_SCORE, abs=1e-9)
        assert top["score"] == pytest.approx(HEX_MAX_SCORE, abs=1e-9)

    @pytest.mark.parametrize(
        "source",
        [
            # 25 circles instead of 26.
            "def construct():\n    return [(0.5, 0.5, 0.01)] * 25\n",
            # A circle poking out of the unit square.
            "def construct():\n    pts = [(0.5, 0.5, 0.01)] * 26\n    pts[0] = (0.05, 0.5, 0.06)\n    return pts\n",
            # Two circles overlapping.
            "def construct():\n    pts = [(0.5, 0.5, 0.01)] * 26\n    pts[0] = (0.2, 0.5, 0.1)\n    pts[1] = (0.35, 0.5, 0.1)\n    return pts\n",
            # A negative radius.
            "def construct():\n    pts = [(0.5, 0.5, 0.01)] * 26\n    pts[0] = (0.5, 0.5, -0.01)\n    return pts\n",
            # A non-finite coordinate.
            "def construct():\n    pts = [(0.5, 0.5, 0.01)] * 26\n    pts[0] = (float('nan'), 0.5, 0.01)\n    return pts\n",
            # Not an iterable of triples at all.
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

    def test_the_tolerances_match_the_benchmark(self, tmp_path: Path) -> None:
        """A hair of overlap inside 1e-6 slack stays valid, beyond it does not."""
        inside = (
            "def construct():\n"
            "    pts = [(0.5, 0.5, 0.0)] * 26\n"
            "    pts[0] = (0.1, 0.5, 0.1)\n"
            "    pts[1] = (0.1 + 0.2 - 5e-7, 0.5, 0.1)\n"
            "    return pts\n"
        )
        outside = (
            "def construct():\n"
            "    pts = [(0.5, 0.5, 0.0)] * 26\n"
            "    pts[0] = (0.1, 0.5, 0.1)\n"
            "    pts[1] = (0.1 + 0.2 - 5e-6, 0.5, 0.1)\n"
            "    return pts\n"
        )
        assert _grade(tmp_path, inside, "inside")["error_type"] is None
        assert _grade(tmp_path, outside, "outside")["error_type"] == "CONSTRAINT_VIOLATION"

    def test_a_raising_construct_is_an_evaluator_error_not_a_zero(self, tmp_path: Path) -> None:
        result = _grade(tmp_path, "def construct():\n    raise RuntimeError('boom')\n")
        assert result["status"] == "ERROR"
        assert result["error_type"] == "RuntimeError"

    def test_a_missing_construct_is_an_evaluator_error(self, tmp_path: Path) -> None:
        result = _grade(tmp_path, "def other():\n    return []\n")
        assert result["status"] == "ERROR"

    def test_the_result_payload_is_json_serialisable(self, tmp_path: Path) -> None:
        result = _grade(tmp_path, render_candidate("hex_max", recipe_for("circle_packing")))
        assert json.loads(json.dumps(result))["circles_digest"] == result["circles_digest"]


class TestFixtureLadder:
    def _task(self) -> TaskSpec:
        return TaskSpec(
            task_id="circle_packing",
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

    def test_the_ladder_advances_rings_to_hex_to_hex_max(self, tmp_path: Path) -> None:
        first, attempt1 = self._generate(tmp_path, TASK_DIR / "initial", "gen1")
        assert attempt1.extra["strategy"] == "hex_rows"  # type: ignore[attr-defined]
        second, attempt2 = self._generate(tmp_path, first, "gen2")
        assert attempt2.extra["strategy"] == "hex_max"  # type: ignore[attr-defined]
        third, attempt3 = self._generate(tmp_path, second, "gen3")
        assert (third / "candidate.py").read_bytes() == (second / "candidate.py").read_bytes()
        assert "unchanged" in attempt3.summary  # type: ignore[attr-defined]

    def test_the_ladder_scores_are_strictly_increasing_on_the_real_harness(
        self, tmp_path: Path
    ) -> None:
        scores = []
        for name in ("hex_rows", "hex_max"):
            result = _grade(tmp_path, render_candidate(name, recipe_for("circle_packing")), name=name)
            scores.append(result["score"])
        assert RINGS_SCORE < scores[0] < scores[1] < 1.0


class TestOfflineLoopOnTheNewTask:
    def test_the_full_loop_commits_and_every_world_reaches_the_top_rung(
        self, tmp_path: Path
    ) -> None:
        run_id = "circle-packing-offline-1"
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
            assert world.s_best == pytest.approx(HEX_MAX_SCORE, abs=1e-9)
            assert world.b_count > 0
