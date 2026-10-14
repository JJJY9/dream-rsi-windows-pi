"""The textdistance speed task: harness gates, fixture ladder, tiny offline loop.

The workspace is a vendored copy of a real project (life4/textdistance, MIT),
so these tests also pin the vendoring itself: the frozen reference and the
manifest must match the shipped workspace, and only the entry file may change.
"""
from __future__ import annotations

import importlib.util
import shutil
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
from dream_rsi.config import Config, LoopConfig, load_config  # noqa: E402
from dream_rsi.domain.models import RunState, WorldState  # noqa: E402

TASK_DIR = REPO / "tasks" / "textdistance"
ENTRY_REL = "textdistance/algorithms/edit_based.py"


def _load_harness() -> object:
    path = TASK_DIR / "evaluator_harness.py"
    spec = importlib.util.spec_from_file_location("textdistance_harness", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HARNESS = _load_harness()


def _workspace(tmp_path: Path, name: str = "ws") -> Path:
    ws = tmp_path / name
    shutil.copytree(TASK_DIR / "initial", ws)
    return ws


def _grade(ws: Path) -> dict:
    return HARNESS.run(ws)  # type: ignore[attr-defined]


def _install_entry(ws: Path, source: str) -> None:
    (ws / ENTRY_REL).write_text(source, encoding="utf-8", newline="\n")


class TestHarness:
    def test_the_shipped_initial_workspace_scores_near_baseline(self, tmp_path: Path) -> None:
        result = _grade(_workspace(tmp_path))
        assert result["status"] == "OK"
        assert result["error_type"] is None
        assert 0.0 < result["score"] < 0.2  # ~1x speedup vs the 20x target
        assert result["speedup"] == pytest.approx(1.0, rel=0.5)

    def test_the_myers_variant_saturates_the_target(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path)
        _install_entry(ws, render_candidate("myers", recipe_for("textdistance")))
        (ws / "scratch_notes.txt").write_text("scratch is allowed", encoding="utf-8")
        result = _grade(ws)
        assert result["status"] == "OK"
        assert result["error_type"] is None
        assert result["speedup"] > 12.0  # ~26x on the authoring machine
        assert result["score"] == 1.0

    def test_the_mid_ladder_variant_scores_between(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path)
        _install_entry(ws, render_candidate("fastdp", recipe_for("textdistance")))
        result = _grade(ws)
        assert result["status"] == "OK"
        assert 0.0 < result["score"] < 0.3

    def test_behavior_mismatch_scores_a_measured_zero(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path)
        # A subtly fast "optimization" that lies: Myers with a wrong score step.
        lying = render_candidate("myers", recipe_for("textdistance")).replace(
            "score += 1", "score += 2"
        )
        _install_entry(ws, lying)
        result = _grade(ws)
        assert result["status"] == "OK"
        assert result["score"] == 0.0
        assert result["error_type"] == "CONSTRAINT_VIOLATION"
        assert "behavior differs" in result["error"]

    def test_editing_another_file_scores_zero(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path)
        base = ws / "textdistance/algorithms/base.py"
        base.write_text(base.read_text(encoding="utf-8") + "\n# sneaky\n", encoding="utf-8")
        result = _grade(ws)
        assert result["status"] == "OK"
        assert result["score"] == 0.0
        assert result["error_type"] == "CONSTRAINT_VIOLATION"
        assert "may be edited" in result["error"]

    def test_a_broken_entry_is_an_evaluator_error(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path)
        _install_entry(ws, "def broken(:\n")
        result = _grade(ws)
        assert result["status"] == "ERROR"
        assert result["score"] == 0.0

    def test_the_frozen_reference_matches_the_shipped_workspace(self) -> None:
        """The reactor copy and manifest must describe the shipped initial tree."""
        import hashlib
        import json

        manifest = json.loads(
            (TASK_DIR / "reactor/manifest.json").read_text(encoding="utf-8")
        )
        for rel, expected in manifest.items():
            actual = hashlib.sha256((TASK_DIR / "initial" / rel).read_bytes()).hexdigest()
            assert actual == expected, rel
        ref_entry = TASK_DIR / "reactor/textdistance_ref/algorithms/edit_based.py"
        init_entry = TASK_DIR / "initial" / ENTRY_REL
        assert ref_entry.read_bytes() == init_entry.read_bytes()


class TestFixtureLadder:
    def _task(self) -> TaskSpec:
        return TaskSpec(
            task_id="textdistance",
            entry_filename=ENTRY_REL,
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

    def test_the_ladder_advances_original_to_fastdp_to_myers(self, tmp_path: Path) -> None:
        first, attempt1 = self._generate(tmp_path, TASK_DIR / "initial", "gen1")
        assert attempt1.extra["strategy"] == "fastdp"  # type: ignore[attr-defined]
        assert (first / ENTRY_REL).read_text(encoding="utf-8").startswith("# STRATEGY: fastdp")
        second, attempt2 = self._generate(tmp_path, first, "gen2")
        assert attempt2.extra["strategy"] == "myers"  # type: ignore[attr-defined]
        _third, attempt3 = self._generate(tmp_path, second, "gen3")
        assert attempt3.extra["strategy"] is None or "unchanged" in attempt3.summary  # type: ignore[attr-defined]

    def test_the_ladder_scores_are_strictly_increasing_on_the_real_harness(
        self, tmp_path: Path
    ) -> None:
        scores = []
        for i, name in enumerate(("fastdp", "myers")):
            ws = _workspace(tmp_path, name=f"ws{i}")
            _install_entry(ws, render_candidate(name, recipe_for("textdistance")))
            scores.append(_grade(ws)["score"])
        assert 0.0 < scores[0] < scores[1]
        assert scores[1] == 1.0


class TestOfflineLoopOnTheNewTask:
    def test_the_full_loop_commits_and_every_world_reaches_the_myers_rung(
        self, tmp_path: Path
    ) -> None:
        run_id = "textdistance-offline-1"
        config = Config(
            project_root=tmp_path,
            task_dir=TASK_DIR,
            incumbent_policy_path=REPO / "policies" / "incumbent.py",
            backend="fixture",
            loop=LoopConfig(outer_rounds=2, K1=2),
            run_id=run_id,
            entry_filename=ENTRY_REL,
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
            # Timing noise could dock the top rung slightly; it must still be
            # far above the mid rung and the baseline.
            assert world.s_best > 0.5

    def test_the_config_names_the_nested_entry_file(self, tmp_path: Path) -> None:
        config = load_config(project_root=REPO, task="textdistance")
        assert config.entry_filename == ENTRY_REL
        assert load_config(project_root=REPO, task="packing").entry_filename == "candidate.py"
