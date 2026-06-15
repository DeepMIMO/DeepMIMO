# ruff: noqa: EM101, EM102, FBT001, ICN001, N818, PLC0415, S603, TRY003, TRY300, TRY301
"""Run a real MATLAB RT export through the production DeepMIMO converter."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap
import uuid
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


MATLAB_EXPORT_SCRIPT = r"""
json_path = '__JSON_PATH__';
frequency_hz = 3.5e9;
tx_pos = [0; 0; 2];
rx_pos = [10; 0; 1.5];

pm = propagationModel('raytracing', ...
    'CoordinateSystem', 'cartesian', ...
    'Method', 'sbr', ...
    'MaxNumReflections', 0, ...
    'MaxNumDiffractions', 0, ...
    'MaxRelativePathLoss', 40);

tx = txsite('cartesian', ...
    'Name', 'tx1', ...
    'AntennaPosition', tx_pos, ...
    'TransmitterFrequency', frequency_hz);
rx = rxsite('cartesian', ...
    'Name', 'rx1', ...
    'AntennaPosition', rx_pos);

rays_out = raytrace(tx, rx, pm);
if iscell(rays_out)
    ray_list = rays_out{1};
else
    ray_list = rays_out;
end

if isempty(ray_list)
    error('matlab_rt_smoke_export:NoRays', 'raytrace returned no rays.');
end

export = struct();
export.metadata = struct( ...
    'experiment', 'matlab_rt_production_smoke', ...
    'matlab_version', version, ...
    'description', 'One TX, one RX, Cartesian LoS MATLAB ray tracing smoke test');
export.scene = struct( ...
    'coordinate_system', 'cartesian', ...
    'frequency_hz', frequency_hz, ...
    'tx_position_m', row_vector(tx_pos), ...
    'rx_position_m', row_vector(rx_pos), ...
    'geometry', 'empty Cartesian scene; no buildings, terrain mesh, or reflector geometry');
export.transmitter = struct( ...
    'name', char(tx.Name), ...
    'antenna_position_m', row_vector(tx.AntennaPosition), ...
    'transmitter_frequency_hz', tx.TransmitterFrequency);
export.receiver = struct( ...
    'name', char(rx.Name), ...
    'antenna_position_m', row_vector(rx.AntennaPosition));
export.propagation_model = struct( ...
    'class', class(pm), ...
    'coordinate_system', 'cartesian', ...
    'method', 'sbr', ...
    'max_num_reflections', 0, ...
    'max_num_diffractions', 0, ...
    'max_absolute_path_loss_db', [], ...
    'max_relative_path_loss_db', 40);
export.num_rays = numel(ray_list);

ray_records = cell(1, numel(ray_list));
for ray_idx = 1:numel(ray_list)
    ray_records{ray_idx} = export_ray(ray_list(ray_idx), ray_idx);
end
export.rays = ray_records;

try
    json_text = jsonencode(export, PrettyPrint=true);
catch
    json_text = jsonencode(export);
end

fid = fopen(json_path, 'w');
if fid < 0
    error('matlab_rt_smoke_export:OpenFailed', 'Could not open JSON output path.');
end
cleanup = onCleanup(@() fclose(fid));
fwrite(fid, json_text, 'char');

fprintf('MATLAB_RT_SMOKE_EXPORT=%s\n', json_path);
fprintf('MATLAB_RT_SMOKE_RAY_COUNT=%d\n', numel(ray_list));

function record = export_ray(ray, index)
    interactions = get_prop(ray, 'Interactions', struct([]));
    record = struct();
    record.index = index;
    record.class = class(ray);
    record.path_specification = string_value(get_prop(ray, 'PathSpecification', 'Locations'));
    record.coordinate_system = string_value(get_prop(ray, 'CoordinateSystem', 'Cartesian'));
    record.system_scale = scalar_value(get_prop(ray, 'SystemScale', 1));
    record.transmitter_location_m = vector3(get_prop(ray, 'TransmitterLocation', [0; 0; 0]));
    record.receiver_location_m = vector3(get_prop(ray, 'ReceiverLocation', [0; 0; 0]));
    record.line_of_sight = logical(get_prop(ray, 'LineOfSight', false));
    record.frequency_hz = scalar_value(get_prop(ray, 'Frequency', []));
    record.path_loss_source = string_value(get_prop(ray, 'PathLossSource', 'Custom'));
    record.path_loss_db = scalar_value(get_prop(ray, 'PathLoss', []));
    record.phase_shift_rad = scalar_value(get_prop(ray, 'PhaseShift', []));
    record.phase_shift_deg = rad2deg(record.phase_shift_rad);
    record.propagation_delay_s = scalar_value(get_prop(ray, 'PropagationDelay', []));
    record.propagation_distance_m = scalar_value(get_prop(ray, 'PropagationDistance', []));
    record.angle_of_departure_deg = angle_pair(get_prop(ray, 'AngleOfDeparture', []));
    record.angle_of_arrival_deg = angle_pair(get_prop(ray, 'AngleOfArrival', []));
    record.num_interactions = numel(interactions);
    record.interactions = export_interactions(interactions);
    record.path_coordinates_m = path_coordinates_or_sites( ...
        get_prop(ray, 'PathCoordinates', []), ...
        record.transmitter_location_m, ...
        record.receiver_location_m, ...
        record.interactions);
end

function records = export_interactions(interactions)
    records = cell(1, numel(interactions));
    for idx = 1:numel(interactions)
        item = interactions(idx);
        record = struct();
        record.index = idx;
        record.Type = string_value(get_prop(item, 'Type', ''));
        record.Location = vector3(get_prop(item, 'Location', [NaN; NaN; NaN]));
        material_name = string_value(get_prop(item, 'MaterialName', 'unknown'));
        if isempty(material_name)
            material_name = 'unknown';
        end
        record.MaterialName = material_name;
        records{idx} = record;
    end
end

function value = get_prop(object, name, default_value)
    try
        value = object.(name);
    catch
        value = default_value;
    end
end

function out = string_value(value)
    if isempty(value)
        out = '';
    else
        out = char(string(value));
    end
end

function out = scalar_value(value)
    if isempty(value)
        error('matlab_rt_smoke_export:MissingScalar', 'Required ray scalar is empty.');
    end
    out = double(value(1));
end

function out = row_vector(value)
    out = double(value(:).');
end

function out = vector3(value)
    out = row_vector(value);
    if numel(out) ~= 3
        error('matlab_rt_smoke_export:InvalidVector3', 'Expected a 3-element vector.');
    end
end

function out = angle_pair(value)
    out = row_vector(value);
    if numel(out) ~= 2
        error('matlab_rt_smoke_export:InvalidAnglePair', 'Expected a 2-element angle vector.');
    end
end

function out = coordinate_rows(value)
    value = double(value);
    if isempty(value)
        error('matlab_rt_smoke_export:MissingPathCoordinates', 'PathCoordinates is empty.');
    end
    if size(value, 1) == 3
        out = value.';
    elseif size(value, 2) == 3
        out = value;
    else
        error('matlab_rt_smoke_export:InvalidPathCoordinates', ...
            'PathCoordinates must be 3xN or Nx3.');
    end
end

function out = path_coordinates_or_sites(value, tx_location, rx_location, interactions)
    if ~isempty(value)
        out = coordinate_rows(value);
        return;
    end
    out = tx_location;
    for idx = 1:numel(interactions)
        out = [out; interactions{idx}.Location]; %#ok<AGROW>
    end
    out = [out; rx_location];
end
"""


class SmokeFailure(RuntimeError):
    """Raised when the MATLAB RT smoke workflow fails."""


def find_matlab_executable(explicit: str | None = None) -> Path:
    """Return a MATLAB executable path or raise an explicit smoke failure."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("MATLAB_EXE"):
        candidates.append(Path(os.environ["MATLAB_EXE"]))

    discovered = shutil.which("matlab")
    if discovered:
        candidates.append(Path(discovered))

    matlab_root = Path("C:/Program Files/MATLAB")
    if matlab_root.exists():
        candidates.extend(sorted(matlab_root.glob("R*/bin/matlab.exe"), reverse=True))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise SmokeFailure("MATLAB executable not found. Set MATLAB_EXE or pass --matlab-exe.")


def write_matlab_export_script(work_dir: Path, json_path: Path) -> Path:
    """Write the minimal MATLAB ray tracing export script."""
    script_path = work_dir / "export_matlab_rt_smoke.m"
    matlab_json_path = json_path.as_posix().replace("'", "''")
    script_path.write_text(
        MATLAB_EXPORT_SCRIPT.replace("__JSON_PATH__", matlab_json_path),
        encoding="utf-8",
    )
    return script_path


def run_matlab_export(matlab_exe: Path, script_path: Path) -> subprocess.CompletedProcess[str]:
    """Execute MATLAB ray tracing and return the completed process."""
    matlab_script = script_path.as_posix().replace("'", "''")
    command = [str(matlab_exe), "-batch", f"run('{matlab_script}')"]
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def validate_deepmimo_pipeline(
    json_path: Path,
    scenario_name: str,
    keep_output: bool,
) -> dict[str, object]:
    """Convert the MATLAB export and validate DeepMIMO workflows."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import deepmimo as dm
    from deepmimo import consts as c
    from deepmimo.converters.matlab_rt import convert_matlab_rt_json
    from deepmimo.datasets.dataset import Dataset

    scenario_root = REPO_ROOT / c.SCENARIOS_FOLDER
    scenario_path = scenario_root / scenario_name
    scenario_root.mkdir(exist_ok=True)

    try:
        result = convert_matlab_rt_json(
            json_path,
            scenario_root=scenario_root,
            scenario_name=scenario_name,
            overwrite=True,
        )

        dataset = dm.load(scenario_name, max_paths=2)
        if not isinstance(dataset, Dataset):
            raise SmokeFailure(f"Expected a Dataset, got {type(dataset).__name__}.")

        td_params = dm.ChannelParameters(
            freq_domain=False,
            num_paths=2,
            bs_antenna={"shape": [1, 1]},
            ue_antenna={"shape": [1, 1]},
        )
        channel = dataset.compute_channels(td_params)
        if not np.isfinite(channel).all() or np.linalg.norm(channel) <= 0:
            raise SmokeFailure("compute_channels() returned non-finite or zero output.")

        ofdm_params = dm.ChannelParameters(
            freq_domain=True,
            num_paths=2,
            bs_antenna={"shape": [1, 1]},
            ue_antenna={"shape": [1, 1]},
            ofdm={
                "subcarriers": 32,
                "selected_subcarriers": np.arange(8),
                "rx_filter": False,
            },
        )
        ofdm_channel = dataset.compute_channels(ofdm_params)
        if not np.isfinite(ofdm_channel).all() or np.linalg.norm(ofdm_channel) <= 0:
            raise SmokeFailure("OFDM channel generation returned non-finite or zero output.")

        bf_params = dm.ChannelParameters(
            freq_domain=False,
            num_paths=2,
            bs_antenna={"shape": [8, 1]},
            ue_antenna={"shape": [1, 1]},
        )
        bf_channel = dataset.compute_channels(bf_params)
        h = bf_channel[0, 0, :, 0]
        beamformer = h.conj() / np.linalg.norm(h)
        beamforming_gain = float(abs(beamformer.conj() @ h) ** 2)
        if not np.isfinite(beamforming_gain) or beamforming_gain <= 0:
            raise SmokeFailure("Beamforming smoke check returned invalid gain.")

        ax = dataset.plot_rays(0, proj_3D=False)
        ray_plot_line_count = len(ax.lines)
        plt.close(ax.figure)
        if ray_plot_line_count <= 0:
            raise SmokeFailure("Ray plotting returned no plotted lines.")

        return {
            "scenario_path": str(result.scenario_path),
            "written_file_count": len(result.files_written),
            "path_count": dataset.num_paths.tolist(),
            "channel_shape": list(channel.shape),
            "ofdm_shape": list(ofdm_channel.shape),
            "beamforming_gain": beamforming_gain,
            "ray_plot_line_count": ray_plot_line_count,
        }
    finally:
        if not keep_output and scenario_path.exists():
            shutil.rmtree(scenario_path)


def main(argv: list[str] | None = None) -> int:
    """Run the complete smoke workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matlab-exe", help="Optional path to matlab.exe.")
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="Keep generated smoke artifacts.",
    )
    args = parser.parse_args(argv)

    os.chdir(REPO_ROOT)
    run_id = uuid.uuid4().hex[:12]
    work_dir = REPO_ROOT / f".matlab_rt_smoke_tmp_{run_id}"
    work_dir.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MPLCONFIGDIR", str(work_dir / "mplconfig"))

    scenario_name = f"matlab_rt_smoke_{run_id}"
    json_path = work_dir / "matlab_rt_smoke_export.json"

    try:
        matlab_exe = find_matlab_executable(args.matlab_exe)
        script_path = write_matlab_export_script(work_dir, json_path)
        matlab_run = run_matlab_export(matlab_exe, script_path)
        if matlab_run.returncode != 0:
            raise SmokeFailure(
                "MATLAB ray tracing execution failed.\n"
                + textwrap.indent(matlab_run.stdout.strip(), "  ")
            )
        if not json_path.exists():
            raise SmokeFailure("MATLAB completed but did not write the expected JSON export.")

        with json_path.open("r", encoding="utf-8") as file:
            export = json.load(file)
        validation = validate_deepmimo_pipeline(json_path, scenario_name, args.keep_output)

        summary = {
            "matlab_executable": str(matlab_exe),
            "matlab_executed": True,
            "matlab_stdout": matlab_run.stdout.strip().splitlines(),
            "export_path": str(json_path),
            "export_num_rays": export["num_rays"],
            "validation": validation,
        }
        print(json.dumps(summary, indent=2))
        return 0
    except SmokeFailure as error:
        print(f"MATLAB RT smoke test failed: {error}", file=sys.stderr)
        return 2
    finally:
        if not args.keep_output and work_dir.exists():
            shutil.rmtree(work_dir)


if __name__ == "__main__":
    raise SystemExit(main())
