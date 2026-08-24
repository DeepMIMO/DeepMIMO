"""Tests for the dashboard job planner and scene discovery.

Generation and tracing are independent stages: a scene is built once and traced
many times. These tests cover which stages a job plans to run and which existing
scenes it can reuse. Running the stages needs Infinigen and Sionna, so only the
planning is exercised here.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "scenario_dashboard" / "jobs.py"
)
_spec = importlib.util.spec_from_file_location("dashboard_jobs", _MODULE_PATH)
jobs = importlib.util.module_from_spec(_spec)
sys.modules["dashboard_jobs"] = jobs
_spec.loader.exec_module(jobs)


@pytest.fixture
def manager(tmp_path):
    """Build a manager that never launches a subprocess.

    Args:
        tmp_path: Pytest temporary folder.

    Returns:
        A job manager rooted at the temporary folder.

    """
    return jobs.JobManager(
        gen_python="gen",
        rt_python="rt",
        runner=tmp_path / "runner.py",
        work_root=tmp_path / "work",
    )


def _make_scene(root: Path, name: str, *, blend: bool, mitsuba: bool, flat: bool = False) -> Path:
    """Create a scene folder on disk.

    Args:
        root: Folder to create the scene in.
        name: Scene folder name.
        blend: Whether to write a ``scene.blend``.
        mitsuba: Whether to write a ``scene.xml``.
        flat: Write the files directly in the folder rather than in subfolders.

    Returns:
        The scene folder.

    """
    folder = root / name
    if blend:
        target = folder if flat else folder / "blend"
        target.mkdir(parents=True, exist_ok=True)
        (target / "scene.blend").write_bytes(b"")
    if mitsuba:
        target = folder if flat else folder / "mitsuba"
        target.mkdir(parents=True, exist_ok=True)
        (target / "scene.xml").write_text("<scene/>")
    return folder


def test_new_scene_plans_every_stage(manager) -> None:
    """A fresh scene generates, exports and traces."""
    plan = manager.plan_stages(jobs.Job(id="a", params={"name": "s", "source": "new"}))
    assert set(plan) == {"generate", "export", "trace"}
    assert sum(plan.values()) == pytest.approx(1.0)


def test_reused_scene_only_traces(manager) -> None:
    """Reusing a built scene skips generation and export."""
    plan = manager.plan_stages(jobs.Job(id="a", params={"name": "s", "source": "prior"}))
    assert set(plan) == {"trace"}
    assert plan["trace"] == pytest.approx(1.0)


def test_reused_scene_can_re_export(manager) -> None:
    """Re-exporting a built scene adds the export stage back."""
    params = {"name": "s", "source": "prior", "reexport": True}
    plan = manager.plan_stages(jobs.Job(id="a", params=params))
    assert set(plan) == {"export", "trace"}
    assert sum(plan.values()) == pytest.approx(1.0)


def test_trace_only_progress_starts_at_zero(manager) -> None:
    """A trace-only run does not open at the weight of skipped stages."""
    job = jobs.Job(id="a", params={"name": "s", "source": "prior"})
    job.weights = manager.plan_stages(job)
    manager._set_stage(job, "trace")  # noqa: SLF001 - the progress base has no public reader
    assert job.overall == pytest.approx(0.0)


def test_resolve_scene_accepts_both_layouts(tmp_path) -> None:
    """A scene folder may nest its parts or be the exported folder itself."""
    nested = _make_scene(tmp_path, "nested", blend=True, mitsuba=True)
    flat = _make_scene(tmp_path, "flat", blend=False, mitsuba=True, flat=True)

    assert jobs.resolve_scene(nested) == (nested / "blend", nested / "mitsuba")
    assert jobs.resolve_scene(flat)[1] == flat


def test_list_scenes_reports_what_each_scene_can_do(tmp_path) -> None:
    """Scenes are listed with the stages they can still run."""
    work = tmp_path / "work"
    _make_scene(work, "traceable", blend=True, mitsuba=True)
    _make_scene(work, "needs_export", blend=True, mitsuba=False)
    _make_scene(work, "empty", blend=False, mitsuba=False)

    listed = {scene["name"]: scene for scene in jobs.list_scenes(work)}

    assert set(listed) == {"traceable", "needs_export"}
    assert listed["traceable"]["has_geometry"]
    assert not listed["needs_export"]["has_geometry"]
    assert listed["needs_export"]["has_blend"]


def test_scenes_outside_the_work_root_are_named_by_path(tmp_path) -> None:
    """Two roots may hold the same folder name, so extra roots use full paths."""
    work, extra = tmp_path / "work", tmp_path / "extra"
    _make_scene(work, "scene", blend=False, mitsuba=True)
    outside = _make_scene(extra, "scene", blend=False, mitsuba=True, flat=True)

    names = {scene["name"] for scene in jobs.list_scenes(work, extra)}

    assert names == {"scene", str(outside)}
    assert all(scene["label"] == "scene" for scene in jobs.list_scenes(work, extra))
