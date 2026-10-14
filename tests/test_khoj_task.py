"""The khoj_orgnode speed task: harness gates, fixture ladder, tiny offline loop.

The workspace vendors a real project (khoj-ai/khoj `src/khoj` at ae229ca,
AGPL-3.0), so the tests also pin the vendoring: the frozen reference and the
manifest must match the shipped workspace, and only the entry file may change.
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

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

TASK_DIR = REPO / "tasks" / "khoj_orgnode"
ENTRY_REL = "processor/content/org_mode/orgnode.py"


def _load_harness() -> object:
    path = TASK_DIR / "evaluator_harness.py"
    spec = importlib.util.spec_from_file_location("khoj_harness", path)
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
        assert 0.25 < result["score"] < 0.6  # ~1x speedup vs the 2.5x target

    def test_the_routed_variant_saturates_the_target(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path)
        _install_entry(ws, render_candidate("routed", recipe_for("khoj_orgnode")))
        (ws / "scratch_notes.txt").write_text("scratch is allowed", encoding="utf-8")
        result = _grade(ws)
        assert result["status"] == "OK"
        assert result["error_type"] is None
        # ~2.6-2.8x on the authoring machine; the margin over the 2.5 target is
        # modest, so the threshold leaves room for machine-load noise (the
        # exact numbers are recorded by the harness in every run).
        assert result["speedup"] > 2.1, result
        assert result["score"] > 0.8, result

    def test_the_mid_ladder_variant_scores_between(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path)
        _install_entry(ws, render_candidate("gated", recipe_for("khoj_orgnode")))
        result = _grade(ws)
        assert result["status"] == "OK"
        # ~2.0x measured; mid-rung ratios compress under machine load, so the
        # assertion only pins "measured, not saturated". The ordering claim
        # lives in the ladder test, with tolerances.
        assert result["speedup"] > 1.3, result
        assert result["score"] < 1.0, result

    def test_behavior_mismatch_scores_a_measured_zero(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path)
        original = (TASK_DIR / "initial" / ENTRY_REL).read_text(encoding="utf-8")
        broken = original.replace(
            "heading = heading_search.group(2)",
            'heading = heading_search.group(2) + "X"',
        )
        assert broken != original
        _install_entry(ws, broken)
        result = _grade(ws)
        assert result["status"] == "OK"
        assert result["score"] == 0.0
        assert result["error_type"] == "CONSTRAINT_VIOLATION"
        assert "behavior differs" in result["error"]

    def test_editing_another_file_scores_zero(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path)
        target = ws / "processor/content/org_mode/org_to_entries.py"
        target.write_text(target.read_text(encoding="utf-8") + "\n# sneaky\n", encoding="utf-8")
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
        import hashlib
        import json

        manifest = json.loads(
            (TASK_DIR / "reactor/manifest.json").read_text(encoding="utf-8")
        )
        for rel, expected in manifest.items():
            actual = hashlib.sha256((TASK_DIR / "initial" / rel).read_bytes()).hexdigest()
            assert actual == expected, rel
        ref_entry = TASK_DIR / "reactor/orgnode_ref.py"
        init_entry = TASK_DIR / "initial" / ENTRY_REL
        assert ref_entry.read_bytes() == init_entry.read_bytes()


class TestFixtureLadder:
    def _task(self) -> TaskSpec:
        return TaskSpec(
            task_id="khoj_orgnode",
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

    def test_the_ladder_advances_original_to_routed(self, tmp_path: Path) -> None:
        first, attempt1 = self._generate(tmp_path, TASK_DIR / "initial", "gen1")
        assert attempt1.extra["strategy"] == "precompiled"  # type: ignore[attr-defined]
        assert (first / ENTRY_REL).read_text(encoding="utf-8").startswith(
            "# STRATEGY: precompiled"
        )
        second, attempt2 = self._generate(tmp_path, first, "gen2")
        assert attempt2.extra["strategy"] == "gated"  # type: ignore[attr-defined]
        third, attempt3 = self._generate(tmp_path, second, "gen3")
        assert attempt3.extra["strategy"] == "routed"  # type: ignore[attr-defined]
        _fourth, attempt4 = self._generate(tmp_path, third, "gen4")
        assert attempt4.extra["strategy"] is None or "unchanged" in attempt4.summary  # type: ignore[attr-defined]

    def test_the_ladder_scores_climb_from_the_baseline(self, tmp_path: Path) -> None:
        baseline = _grade(_workspace(tmp_path, name="wsb"))["score"]
        scores = []
        for i, name in enumerate(("precompiled", "gated", "routed")):
            ws = _workspace(tmp_path, name=f"ws{i}")
            _install_entry(ws, render_candidate(name, recipe_for("khoj_orgnode")))
            scores.append(_grade(ws)["score"])
        # Measured bands on the authoring machine (scores against the 2.5
        # target): baseline ~0.41, precompiled ~0.64-0.70, gated ~0.78-0.83,
        # routed 1.0 (capped). Ratios between adjacent grades compress under
        # machine load, so the tight pair carries a 5% tolerance and the
        # top-rung claims are absolute.
        assert baseline > 0.0, baseline
        assert scores[0] > baseline * 0.95, (baseline, scores)
        assert scores[1] > scores[0], (baseline, scores)
        assert scores[2] > scores[1], (baseline, scores)
        assert scores[2] > 0.8, scores
        assert scores[2] > baseline * 1.4, (baseline, scores)


class TestOfflineLoopOnTheNewTask:
    def test_the_full_loop_commits_and_every_world_reaches_the_top_rung(
        self, tmp_path: Path
    ) -> None:
        run_id = "khoj-offline-1"
        config = Config(
            project_root=tmp_path,
            task_dir=TASK_DIR,
            incumbent_policy_path=REPO / "policies" / "incumbent.py",
            backend="fixture",
            loop=LoopConfig(outer_rounds=2, K1=3),
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
            # World 0 tops out at the routed rung (~2.6-2.8x vs a 2.5x target,
            # score ~1.0); world 1 starts from the gated rung and also reaches
            # routed. The bar leaves room for machine-load noise.
            assert world.s_best > 0.75

    def test_the_config_names_the_nested_entry_file(self) -> None:
        config = load_config(project_root=REPO, task="khoj_orgnode")
        assert config.entry_filename == ENTRY_REL
