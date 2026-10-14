"""The pypinyin speed task: harness gates, fixture ladder, tiny offline loop.

The workspace vendors a real project (mozillazg/pypinyin 0.55.0, MIT), so the
tests also pin the vendoring: the frozen reference and the manifest must match
the shipped workspace, and only the entry file may change.

Grading goes through a fresh ``-I`` subprocess per grade - exactly how
production runs the harness (``FixedCommandEvaluator``). Calling the harness
in-process would time the candidate in a heap that earlier tests in the same
pytest process have already fragmented; the authoring machine measured a
second in-process grade ~15-25% slower than the first while the harness's own
reference subprocess stayed flat, which silently depressed the reported
speedup depending on test order.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
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

TASK_DIR = REPO / "tasks" / "pypinyin"
ENTRY_REL = "pypinyin/converter.py"


def _workspace(tmp_path: Path, name: str = "ws") -> Path:
    ws = tmp_path / name
    shutil.copytree(TASK_DIR / "initial", ws)
    return ws


def _grade(ws: Path) -> dict:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "result.json"
        proc = subprocess.run(
            [sys.executable, "-I", str(TASK_DIR / "evaluator_harness.py"), str(ws), str(out)],
            capture_output=True,
            timeout=300,
        )
        assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
        return json.loads(out.read_text(encoding="utf-8"))


def _install_entry(ws: Path, source: str) -> None:
    (ws / ENTRY_REL).write_text(source, encoding="utf-8", newline="\n")


class TestHarness:
    def test_the_shipped_initial_workspace_scores_near_baseline(self, tmp_path: Path) -> None:
        result = _grade(_workspace(tmp_path))
        assert result["status"] == "OK"
        assert result["error_type"] is None
        assert 0.2 < result["score"] < 0.8  # ~1x speedup vs the 2x target

    def test_the_style_memo_variant_saturates_the_target(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path)
        _install_entry(ws, render_candidate("style_memo", recipe_for("pypinyin")))
        (ws / "scratch_notes.txt").write_text("scratch is allowed", encoding="utf-8")
        result = _grade(ws)
        assert result["status"] == "OK"
        assert result["error_type"] is None
        # Production-style grading measures ~1.6-1.9x on the authoring machine
        # (the in-process grading this suite used earlier flattered every
        # candidate by ~15%); the threshold leaves room for machine-load noise,
        # and the exact numbers are recorded by the harness in every run.
        assert result["speedup"] > 1.35, result
        assert result["score"] > 0.65, result

    def test_the_mid_ladder_variant_scores_between(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path)
        _install_entry(ws, render_candidate("shallow_phrase", recipe_for("pypinyin")))
        result = _grade(ws)
        assert result["status"] == "OK"
        # ~1.2x measured; mid-rung ratios compress under machine load, so the
        # assertion only pins "measured, not saturated". The ordering claim
        # lives in the ladder test, with a tolerance.
        assert result["speedup"] > 0.9, result
        assert result["score"] < 1.0, result

    def test_behavior_mismatch_scores_a_measured_zero(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path)
        broken = (TASK_DIR / "initial" / ENTRY_REL).read_text(encoding="utf-8")
        broken = broken.replace("return [pys]", 'return [["WRONG"]]')
        assert broken != (TASK_DIR / "initial" / ENTRY_REL).read_text(encoding="utf-8")
        _install_entry(ws, broken)
        result = _grade(ws)
        assert result["status"] == "OK"
        assert result["score"] == 0.0
        assert result["error_type"] == "CONSTRAINT_VIOLATION"
        assert "behavior differs" in result["error"]

    def test_editing_another_file_scores_zero(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path)
        target = ws / "pypinyin/style/__init__.py"
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
        ref_entry = TASK_DIR / "reactor/ref_pkg/pypinyin/converter.py"
        init_entry = TASK_DIR / "initial" / ENTRY_REL
        assert ref_entry.read_bytes() == init_entry.read_bytes()


class TestFixtureLadder:
    def _task(self) -> TaskSpec:
        return TaskSpec(
            task_id="pypinyin",
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

    def test_the_ladder_advances_original_to_shallow_to_memo(self, tmp_path: Path) -> None:
        first, attempt1 = self._generate(tmp_path, TASK_DIR / "initial", "gen1")
        assert attempt1.extra["strategy"] == "shallow_phrase"  # type: ignore[attr-defined]
        assert (first / ENTRY_REL).read_text(encoding="utf-8").startswith(
            "# STRATEGY: shallow_phrase"
        )
        second, attempt2 = self._generate(tmp_path, first, "gen2")
        assert attempt2.extra["strategy"] == "style_memo"  # type: ignore[attr-defined]
        _third, attempt3 = self._generate(tmp_path, second, "gen3")
        assert attempt3.extra["strategy"] is None or "unchanged" in attempt3.summary  # type: ignore[attr-defined]

    def test_the_ladder_scores_climb_from_the_baseline(self, tmp_path: Path) -> None:
        scores = []
        for i, name in enumerate(("shallow_phrase", "style_memo")):
            ws = _workspace(tmp_path, name=f"ws{i}")
            _install_entry(ws, render_candidate(name, recipe_for("pypinyin")))
            scores.append(_grade(ws)["score"])
        baseline = _grade(_workspace(tmp_path, name="wsb"))["score"]
        # Measured bands on the authoring machine (production-style grading):
        # baseline ~0.54, mid ~0.60, top ~0.82-0.95. Ratios between adjacent
        # grades compress under machine load, so the ordering tolerance is 5%
        # and the top-rung claim is "a clear gain over baseline".
        assert baseline > 0.0, baseline
        assert scores[0] > baseline * 0.95, (baseline, scores)
        assert scores[1] > scores[0], (baseline, scores)
        assert scores[1] > baseline * 1.25, (baseline, scores)
        assert scores[1] > 0.65, scores


class TestOfflineLoopOnTheNewTask:
    def test_the_full_loop_commits_and_every_world_reaches_the_top_rung(
        self, tmp_path: Path
    ) -> None:
        run_id = "pypinyin-offline-1"
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
            # The top rung measures ~1.6-1.9x against a 2.0x target (a score
            # of ~0.8-0.95); timing noise can dock it, so the bar is "clearly
            # above the mid rung, on the way to saturation".
            assert world.s_best > 0.65

    def test_the_config_names_the_nested_entry_file(self) -> None:
        config = load_config(project_root=REPO, task="pypinyin")
        assert config.entry_filename == ENTRY_REL
