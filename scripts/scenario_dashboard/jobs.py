"""Background pipeline jobs with progress tracking.

Runs generate -> export -> trace as subprocesses and reports where each one has
got to. Generation is the long stage and the one worth measuring properly:
Infinigen announces how many greedy solver stages a scene needs, then logs each
one as it finishes, so progress is a real fraction and the estimate comes from
how long the finished stages actually took rather than a guess.

Generation and export need Blender; ray tracing needs Sionna, which pins an
incompatible numpy. Each stage therefore names its own interpreter, and the
stages hand off through the scene folder on disk.
"""

from __future__ import annotations

import re
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Infinigen prints its stage count once, as a tqdm total, then logs each stage
# as it either places objects or finds nothing to place.
RE_STAGE_TOTAL = re.compile(r"greedy stages coverage:\s+\d+%\|[^|]*\|\s*\d+/(\d+)")
RE_STAGE_DONE = re.compile(r"Finished solving|No objects to be added")
RE_ANNEAL = re.compile(r"it=(\d+)/(\d+)")
RE_EXPORT_DONE = re.compile(r"exported ([\d,]+) triangles")
RE_TRACE_DONE = re.compile(r"-> scenario '([^']+)'")

#: Marks a source that is an existing DeepMIMO scenario rather than a scene folder.
SCENARIO_PREFIX = "scenario:"

# Ray tracing is only part of the trace stage: the scene has to be loaded first
# and converted to DeepMIMO afterwards, and on a large interior the conversion
# outlasts the tracing. The bar is split so it does not sit at 100% waiting.
TRACE_BATCHES_SHARE = 0.7
TRACE_CONVERT_START = 0.75
TRACE_CONVERT_NAMED = 0.9

# Stage weights, used to turn per-stage progress into one overall bar. Ray
# tracing dominates on CPU, so a naive third-each bar would sit at 66% for most
# of the run.
STAGE_WEIGHTS = {"generate": 0.45, "export": 0.10, "trace": 0.45}


@dataclass
class Job:
    """One pipeline run and everything the dashboard shows about it."""

    id: str
    params: dict[str, Any]
    stage: str = "queued"
    status: str = "running"
    detail: str = ""
    stage_fraction: float = 0.0
    overall: float = 0.0
    eta_seconds: float | None = None
    started: float = field(default_factory=time.time)
    finished: float | None = None
    scenario: str | None = None
    error: str | None = None
    weights: dict[str, float] = field(default_factory=dict)
    log: list[str] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-friendly view of the job.

        Returns:
            Dict describing the job's current state.

        """
        return {
            "id": self.id,
            "stage": self.stage,
            "status": self.status,
            "detail": self.detail,
            "stage_fraction": round(self.stage_fraction, 4),
            "overall": round(self.overall, 4),
            "eta_seconds": None if self.eta_seconds is None else int(self.eta_seconds),
            "elapsed_seconds": int((self.finished or time.time()) - self.started),
            "scenario": self.scenario,
            "error": self.error,
            "log": self.log[-40:],
        }


class JobManager:
    """Runs pipeline jobs one at a time and tracks their progress."""

    def __init__(self, gen_python: str, rt_python: str, runner: Path, work_root: Path) -> None:
        """Initialise the manager.

        Args:
            gen_python: Interpreter with Infinigen and bpy available.
            rt_python: Interpreter with Sionna and DeepMIMO available.
            runner: Path to the pipeline runner script.
            work_root: Folder for generated scenes.

        """
        self.gen_python = gen_python
        self.rt_python = rt_python
        self.runner = runner
        self.work_root = work_root
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, params: dict[str, Any]) -> Job:
        """Start a pipeline run in the background.

        Args:
            params: Generation, export and ray-tracing parameters.

        Returns:
            The newly created job.

        """
        job = Job(id=uuid.uuid4().hex[:8], params=params)
        job.weights = self.plan_stages(job)
        with self._lock:
            self.jobs[job.id] = job
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

    def get(self, job_id: str) -> Job | None:
        """Look up a job.

        Args:
            job_id: Job identifier.

        Returns:
            The job, or None if unknown.

        """
        return self.jobs.get(job_id)

    def latest(self) -> Job | None:
        """Return the most recently submitted job, if any.

        Returns:
            The newest job, or None.

        """
        if not self.jobs:
            return None
        return max(self.jobs.values(), key=lambda j: j.started)

    def _emit(self, job: Job, line: str) -> None:
        """Record a log line on a job.

        Args:
            job: Job to update.
            line: Log line.

        """
        job.log.append(line.rstrip())
        del job.log[:-400]

    def plan_stages(self, job: Job) -> dict[str, float]:
        """Weight the stages this job will actually run.

        A trace-only run must not sit at 55% before it starts, so the weights
        are renormalised over the stages in the plan.

        Args:
            job: Job to plan.

        Returns:
            Mapping of stage name to its share of the progress bar.

        """
        source = job.params.get("source") or "new"
        stages = ["trace"]
        if source == "new" or source.startswith(SCENARIO_PREFIX) or job.params.get("reexport"):
            stages.insert(0, "export")
        if source == "new":
            stages.insert(0, "generate")
        total = sum(STAGE_WEIGHTS[s] for s in stages)
        return {s: STAGE_WEIGHTS[s] / total for s in stages}

    def _set_stage(self, job: Job, stage: str) -> None:
        """Move a job to a new stage.

        Args:
            job: Job to update.
            stage: Stage name.

        """
        job.stage = stage
        job.stage_fraction = 0.0
        job.detail = ""
        job.overall = sum(
            weight
            for name, weight in job.weights.items()
            if _stage_order(name) < _stage_order(stage)
        )

    def _run(self, job: Job) -> None:
        """Execute the requested stages for a job.

        A scene is generated once and usually traced many times, so generation
        and export are skipped when an existing scene is reused. The stages are
        independent: only what is asked for runs.

        Args:
            job: Job to run.

        Raises:
            RuntimeError: If a reused scene has no exported geometry.

        """
        try:
            source = job.params.get("source") or "new"
            from_scenario = source.startswith(SCENARIO_PREFIX)

            if from_scenario:
                # An existing scenario is rebuilt into its own scene folder so the
                # original scenario is never touched.
                origin = source[len(SCENARIO_PREFIX):]
                scene_name = f"{origin}__scene"
                scene_dir = self.work_root / scene_name
                blend_dir, mitsuba_dir = scene_dir / "blend", scene_dir / "mitsuba"
            else:
                scene_name = job.params["name"] if source == "new" else source
                scene_dir = (
                    Path(scene_name)
                    if Path(scene_name).is_absolute()
                    else self.work_root / scene_name
                )
                if source == "new":
                    blend_dir, mitsuba_dir = scene_dir / "blend", scene_dir / "mitsuba"
                else:
                    blend_dir, mitsuba_dir = resolve_scene(scene_dir)

            needs_blender = source == "new" or job.params.get("reexport")
            if needs_blender and not self.gen_python:
                msg = (
                    "this run needs Blender and Infinigen, which this server has no "
                    "interpreter for; trace an existing scene instead"
                )
                raise RuntimeError(msg)  # noqa: TRY301

            if source == "new":
                scene_dir.mkdir(parents=True, exist_ok=True)
                self._set_stage(job, "generate")
                self._generate(job, blend_dir)

            if from_scenario:
                self._set_stage(job, "export")
                self._export_scenario(job, origin, mitsuba_dir)
            elif source == "new" or job.params.get("reexport"):
                if not (blend_dir / "scene.blend").exists():
                    msg = f"{scene_name} has no scene.blend to export from"
                    raise RuntimeError(msg)  # noqa: TRY301
                self._set_stage(job, "export")
                self._export(job, blend_dir, mitsuba_dir)

            if not (mitsuba_dir / "scene.xml").exists():
                msg = (
                    f"{scene_name} has no exported geometry; "
                    "tick 'Re-export geometry' to build it"
                )
                raise RuntimeError(msg)  # noqa: TRY301

            self._set_stage(job, "trace")
            self._trace(job, mitsuba_dir)

            job.stage = "done"
            job.status = "done"
            job.overall = 1.0
            job.eta_seconds = 0
        except Exception as exc:  # noqa: BLE001 - surfaced to the dashboard
            job.status = "failed"
            job.error = str(exc)
        finally:
            job.finished = time.time()

    def _stream(self, job: Job, command: list[str], on_line) -> None:  # noqa: ANN001
        """Run a subprocess, feeding each output line to a progress callback.

        Args:
            job: Job being run.
            command: Command to execute.
            on_line: Callback invoked with each output line.

        Raises:
            RuntimeError: If the subprocess exits non-zero.

        """
        self._emit(job, "$ " + " ".join(command[-6:]))
        process = subprocess.Popen(  # noqa: S603
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for raw in process.stdout:  # type: ignore[union-attr]
            line = raw.rstrip()
            if line.startswith(("PROGRESS", "STATUS")):
                continue
            on_line(line)
        code = process.wait()
        if code != 0:
            tail = " | ".join(job.log[-4:])
            msg = f"{job.stage} failed with exit code {code}: {tail}"
            raise RuntimeError(msg)

    def _generate(self, job: Job, out_dir: Path) -> None:
        """Run Infinigen scene generation with stage-level progress.

        Args:
            job: Job being run.
            out_dir: Folder to write ``scene.blend`` into.

        """
        params = job.params
        command = [
            self.gen_python,
            str(self.runner),
            "generate",
            str(out_dir),
            "--seed",
            str(params.get("seed", 0)),
            "--furniture",
            str(params.get("furniture", 4.0)),
        ]
        if params.get("scene_type"):
            command += ["--scene-type", params["scene_type"]]
        if params.get("wall_height"):
            command += ["--wall-height", str(params["wall_height"])]
        if params.get("stories"):
            command += ["--stories", str(params["stories"])]
        for kind in params.get("add_rooms") or []:
            command += ["--add-room", kind]
        if params.get("room_type"):
            command += ["--room-type", params["room_type"]]
        if params.get("fast", True):
            command += ["--fast"]
        if params.get("single_room", True):
            command += ["--single-room"]

        state = {"total": None, "done": 0, "stage_start": time.time()}

        def on_line(line: str) -> None:
            if match := RE_STAGE_TOTAL.search(line):
                state["total"] = int(match.group(1))
            if RE_STAGE_DONE.search(line):
                state["done"] += 1
            if match := RE_ANNEAL.search(line):
                inner = int(match.group(1)) / max(int(match.group(2)), 1)
            else:
                inner = 0.0

            total = state["total"]
            if total:
                # Whole stages plus however far into the current one we are.
                fraction = min((state["done"] + inner) / total, 1.0)
                job.stage_fraction = fraction
                elapsed = time.time() - state["stage_start"]
                if state["done"] >= 1 and fraction > 0:
                    # Estimate from completed stages, which is far steadier than
                    # extrapolating the annealing counter inside one stage.
                    per_stage = elapsed / state["done"]
                    job.eta_seconds = max(per_stage * (total - state["done"]), 0)
                job.detail = f"solver stage {state['done']}/{total}"
            elif "Loading" in line or "Building" in line:
                job.detail = line[:80]
            job.overall = job.weights.get("generate", 0.0) * job.stage_fraction
            if not line.startswith("it="):
                self._emit(job, line)

        self._stream(job, command, on_line)
        job.stage_fraction = 1.0

    def _export(self, job: Job, blend_dir: Path, out_dir: Path) -> None:
        """Export the generated scene to Mitsuba.

        Args:
            job: Job being run.
            blend_dir: Folder containing ``scene.blend``.
            out_dir: Folder to write the Mitsuba scene into.

        """
        params = job.params
        command = [
            self.gen_python,
            str(self.runner),
            "export",
            str(blend_dir / "scene.blend"),
            str(out_dir),
            "--architecture-budget",
            str(params.get("architecture_budget", 2500)),
            "--furniture-budget",
            str(params.get("furniture_budget", 1500)),
            "--ornament-budget",
            str(params.get("ornament_budget", 120)),
            "--min-size",
            str(params.get("min_size", 0.10)),
        ]
        if not params.get("open_doors", True):
            command += ["--keep-doors"]

        def on_line(line: str) -> None:
            self._emit(job, line)
            job.detail = line[:80]
            if RE_EXPORT_DONE.search(line):
                job.stage_fraction = 1.0
            job.overall = job.weights.get("generate", 0.0) + job.weights.get(
                "export", 0.0,
            ) * job.stage_fraction

        self._stream(job, command, on_line)
        job.stage_fraction = 1.0

    def _export_scenario(self, job: Job, scenario: str, out_dir: Path) -> None:
        """Rebuild a Mitsuba scene from an existing DeepMIMO scenario.

        This runs on the ray-tracing interpreter, not the Infinigen one: it needs
        DeepMIMO to read the scenario, and no Blender at all.

        Args:
            job: Job being run.
            scenario: Name of the scenario to rebuild.
            out_dir: Folder to write the Mitsuba scene into.

        """
        params = job.params
        out_dir.mkdir(parents=True, exist_ok=True)
        command = [
            self.rt_python,
            str(self.runner),
            "from-scenario",
            scenario,
            str(out_dir),
            "--material-mode",
            str(params.get("material_mode", "auto")),
        ]

        def on_line(line: str) -> None:
            self._emit(job, line)
            job.detail = line.strip()[:80]
            job.stage_fraction = min(1.0, job.stage_fraction + 0.15)
            job.overall = job.weights.get("generate", 0.0) + job.weights.get(
                "export", 0.0,
            ) * job.stage_fraction

        self._stream(job, command, on_line)
        job.stage_fraction = 1.0

    def _trace(self, job: Job, scene_dir: Path) -> None:
        """Ray trace the exported scene and convert it.

        Args:
            job: Job being run.
            scene_dir: Folder containing ``scene.xml``.

        """
        params = job.params
        command = [
            self.rt_python,
            str(self.runner),
            "trace",
            str(scene_dir),
            "--scenario",
            params["name"],
            "--spacing",
            str(params.get("spacing", 0.3)),
            "--rx-height",
            str(params.get("rx_height", 1.2)),
            "--n-tx",
            str(params.get("n_tx", 2)),
            "--frequency",
            str(params.get("frequency", 3.5e9)),
            "--max-reflections",
            str(params.get("max_reflections", 4)),
            "--samples",
            str(params.get("samples", 250_000)),
        ]
        if params.get("tx_height"):
            command += ["--tx-height", str(params["tx_height"])]
        if params.get("tx_pos"):
            command += ["--tx-pos", params["tx_pos"]]
        if params.get("rx_bounds"):
            command += ["--rx-bounds", params["rx_bounds"]]
        if not params.get("diffraction", True):
            command += ["--no-diffraction"]

        base = job.weights.get("generate", 0.0) + job.weights.get("export", 0.0)
        state = {"batches": None}

        def on_line(line: str) -> None:
            # The tracer reports batches of receivers; that is the only honest
            # progress signal it gives.
            if match := re.search(r"(\d+)/(\d+) \[", line):
                done, total = int(match.group(1)), int(match.group(2))
                state["batches"] = (done, total)
                job.stage_fraction = TRACE_BATCHES_SHARE * done / max(total, 1)
                job.detail = f"tracing batch {done}/{total}"
            elif match := RE_TRACE_DONE.search(line):
                job.scenario = match.group(1)
                job.stage_fraction = max(job.stage_fraction, TRACE_CONVERT_NAMED)
                job.detail = f"writing scenario {match.group(1)}"
                self._emit(job, line)
            else:
                self._emit(job, line)
                if "converting" in line or "converter" in line:
                    job.detail = "converting to DeepMIMO"
                    job.stage_fraction = max(job.stage_fraction, TRACE_CONVERT_START)
            job.overall = base + job.weights.get("trace", 0.0) * job.stage_fraction

        self._stream(job, command, on_line)


def resolve_scene(folder: Path) -> tuple[Path, Path]:
    """Locate the blend and Mitsuba folders of a scene.

    Scenes built by the dashboard nest them as ``blend/`` and ``mitsuba/``, but a
    folder exported by hand may itself be the Mitsuba or blend folder. Both
    layouts are accepted so any existing scene can be traced.

    Args:
        folder: Scene folder.

    Returns:
        The blend folder and the Mitsuba folder, whether or not they exist.

    """
    blend = folder if (folder / "scene.blend").exists() else folder / "blend"
    mitsuba = folder if (folder / "scene.xml").exists() else folder / "mitsuba"
    return blend, mitsuba


def list_scenes(*roots: Path) -> list[dict[str, Any]]:
    """List previously built scenes that can be traced again without regenerating.

    A folder counts as a scene if it holds ``blend/scene.blend`` (re-exportable)
    or ``mitsuba/scene.xml`` (directly traceable). Folders outside the work root
    are identified by their full path so two roots cannot collide.

    Args:
        *roots: Folders holding scenes.

    Returns:
        One entry per scene, newest first.

    """
    scenes = []
    for index, root in enumerate(roots):
        if not root.is_dir():
            continue
        for folder in sorted(root.iterdir()):
            if not folder.is_dir():
                continue
            blend_dir, mitsuba_dir = resolve_scene(folder)
            blend = blend_dir / "scene.blend"
            mitsuba = mitsuba_dir / "scene.xml"
            if not blend.exists() and not mitsuba.exists():
                continue
            scenes.append(
                {
                    "name": folder.name if index == 0 else str(folder),
                    "label": folder.name,
                    "root": str(root),
                    "has_blend": blend.exists(),
                    "has_geometry": mitsuba.exists(),
                    "modified": (mitsuba if mitsuba.exists() else blend).stat().st_mtime,
                },
            )
    scenes.sort(key=lambda s: -s["modified"])
    return scenes


def _stage_order(stage: str) -> int:
    """Return a stage's position in the pipeline.

    Args:
        stage: Stage name.

    Returns:
        Ordinal position.

    """
    return {"queued": -1, "generate": 0, "export": 1, "trace": 2, "done": 3}.get(stage, 0)
