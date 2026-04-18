"""Integration tests for sionna_exporter using real Sionna 2.0 objects.

All tests are skipped automatically when Sionna is not installed so the
standard test suite (without the [sionna] extra) stays green.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

# Skip the entire module if the *real* sionna-rt is not installed.
# We can't rely on pytest.importorskip alone because test_sionna_exporter.py
# may have injected a MagicMock into sys.modules["sionna.rt"] before this
# module is collected, which fools importorskip into not skipping.
# A real module always has __file__ as a str; a MagicMock does not.
try:
    import sionna.rt as _sionna_rt_probe

    if not isinstance(getattr(_sionna_rt_probe, "__file__", None), str):
        pytest.skip("sionna.rt is a mock, not the real package", allow_module_level=True)
    import sionna.rt as sionna_rt
except ImportError:
    pytest.skip("sionna-rt not installed", allow_module_level=True)

from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter

from deepmimo.exporters.sionna_exporter import (
    _get_scene_objects,
    export_paths,
    export_scene_buildings,
    export_scene_materials,
    export_scene_rt_params,
    sionna_exporter,
)
from deepmimo.utils import load_pickle

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def simple_scene():
    """Build a minimal Sionna floor_wall scene with 1 TX and 3 RX."""
    scene = sionna_rt.load_scene(sionna_rt.scene.floor_wall)
    scene.frequency = 3.5e9
    ant = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    scene.tx_array = ant
    scene.rx_array = ant
    scene.add(Transmitter("tx_0", position=[0.0, 0.0, 2.0]))
    for i, x in enumerate([-1.0, 0.0, 1.0]):
        scene.add(Receiver(f"rx_{i}", position=[x, 3.0, 0.5]))
    return scene


@pytest.fixture(scope="module")
def computed_paths(simple_scene):
    """Compute paths on the simple scene with low sample count for speed."""
    p_solver = PathSolver()
    return p_solver(
        scene=simple_scene,
        max_depth=2,
        los=True,
        specular_reflection=True,
        diffuse_reflection=False,
        refraction=True,
        samples_per_src=500_000,
    )


# ---------------------------------------------------------------------------
# export_paths
# ---------------------------------------------------------------------------


class TestExportPaths:
    """export_paths must convert Sionna Paths to well-formed dicts."""

    def test_returns_list(self, computed_paths) -> None:
        """Single Paths input returns a list of length 1."""
        result = export_paths(computed_paths)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_list_input_passthrough(self, computed_paths) -> None:
        """List input with one element returns a list of length 1."""
        result = export_paths([computed_paths])
        assert len(result) == 1

    def test_required_keys_present(self, computed_paths) -> None:
        """All required keys must be present in the exported dict."""
        required = {
            "a",
            "tau",
            "phi_r",
            "phi_t",
            "theta_r",
            "theta_t",
            "vertices",
            "interactions",
            "sources",
            "targets",
        }
        result = export_paths(computed_paths)[0]
        assert required.issubset(result.keys())

    def test_a_is_complex(self, computed_paths) -> None:
        """``a`` must be a complex numpy array (not a tensor tuple)."""
        result = export_paths(computed_paths)[0]
        assert np.iscomplexobj(result["a"])

    def test_sources_shape(self, computed_paths) -> None:
        """Sources must be (n_tx, 3) — transposed from Sionna's (3, n_tx)."""
        result = export_paths(computed_paths)[0]
        assert result["sources"].shape[-1] == 3

    def test_targets_shape(self, computed_paths) -> None:
        """Targets must be (n_rx, 3) — transposed from Sionna's (3, n_rx)."""
        result = export_paths(computed_paths)[0]
        assert result["targets"].shape[-1] == 3

    def test_tau_non_negative(self, computed_paths) -> None:
        """All delays must be >= 0 (time cannot be negative)."""
        result = export_paths(computed_paths)[0]
        # tau may contain zero padding; just check no negative values
        assert np.all(result["tau"] >= 0)


# ---------------------------------------------------------------------------
# export_scene_materials
# ---------------------------------------------------------------------------


class TestExportSceneMaterials:
    """export_scene_materials must return a usable materials list."""

    def test_returns_tuple(self, simple_scene) -> None:
        """Return value must be a (list, ndarray) tuple."""
        mats, indices = export_scene_materials(simple_scene)
        assert isinstance(mats, list)
        assert isinstance(indices, np.ndarray)

    def test_material_has_required_keys(self, simple_scene) -> None:
        """Each material dict must contain all required property keys."""
        required = {
            "name",
            "conductivity",
            "relative_permittivity",
            "scattering_coefficient",
            "xpd_coefficient",
            "scattering_pattern",
        }
        mats, _ = export_scene_materials(simple_scene)
        for mat in mats:
            assert required.issubset(mat.keys()), f"Missing keys in material: {mat}"

    def test_index_count_matches_objects(self, simple_scene) -> None:
        """One index entry per scene object."""
        n_objs = len(_get_scene_objects(simple_scene))
        _, indices = export_scene_materials(simple_scene)
        assert len(indices) == n_objs

    def test_indices_are_valid(self, simple_scene) -> None:
        """All material indices must reference a valid entry in the material list."""
        mats, indices = export_scene_materials(simple_scene)
        assert np.all(indices >= 0)
        assert np.all(indices < len(mats))


# ---------------------------------------------------------------------------
# export_scene_rt_params
# ---------------------------------------------------------------------------


class TestExportSceneRtParams:
    """export_scene_rt_params must capture scene and solver settings."""

    def test_required_keys(self, simple_scene) -> None:
        """All expected parameter keys must be present in the output dict."""
        required = {
            "bandwidth",
            "frequency",
            "rx_array_num_ant",
            "tx_array_num_ant",
            "synthetic_array",
            "raytracer_version",
            "max_depth",
            "los",
            "specular_reflection",
            "diffuse_reflection",
        }
        params = export_scene_rt_params(simple_scene, max_depth=3)
        assert required.issubset(params.keys())

    def test_frequency_matches_scene(self, simple_scene) -> None:
        """Exported frequency must match the scene's configured carrier frequency."""
        params = export_scene_rt_params(simple_scene)
        # Use np.asarray().item() to avoid DeprecationWarning on ndim>0 arrays
        assert float(np.asarray(params["frequency"]).item()) == pytest.approx(3.5e9)

    def test_kwargs_override_defaults(self, simple_scene) -> None:
        """Keyword arguments must override the default parameter values."""
        params = export_scene_rt_params(simple_scene, max_depth=7, los=False)
        assert params["max_depth"] == 7
        assert params["los"] is False

    def test_aliases_present(self, simple_scene) -> None:
        """Converter-compatibility aliases must be present."""
        params = export_scene_rt_params(simple_scene)
        assert "num_samples" in params
        assert "reflection" in params
        assert "scattering" in params
        assert "diffraction" in params

    def test_raytracer_version_is_string_or_none(self, simple_scene) -> None:
        """raytracer_version must be a version string or None."""
        params = export_scene_rt_params(simple_scene)
        assert params["raytracer_version"] is None or isinstance(params["raytracer_version"], str)


# ---------------------------------------------------------------------------
# export_scene_buildings
# ---------------------------------------------------------------------------


class TestExportSceneBuildings:
    """export_scene_buildings must return valid geometry arrays."""

    def test_returns_tuple(self, simple_scene) -> None:
        """Return value must be a (ndarray, dict) tuple."""
        verts, obj_map = export_scene_buildings(simple_scene)
        assert isinstance(verts, np.ndarray)
        assert isinstance(obj_map, dict)

    def test_vertex_matrix_shape(self, simple_scene) -> None:
        """Vertex matrix must be 2-D with 3 columns (x, y, z)."""
        verts, _ = export_scene_buildings(simple_scene)
        assert verts.ndim == 2
        assert verts.shape[1] == 3

    def test_obj_index_map_keys_are_strings(self, simple_scene) -> None:
        """Object map keys must be strings (scene object names)."""
        _, obj_map = export_scene_buildings(simple_scene)
        assert all(isinstance(k, str) for k in obj_map)

    def test_index_ranges_are_contiguous(self, simple_scene) -> None:
        """Start/end index ranges must be non-negative and non-empty."""
        _, obj_map = export_scene_buildings(simple_scene)
        for start, end in obj_map.values():
            assert start >= 0
            assert end > start


# ---------------------------------------------------------------------------
# sionna_exporter (full pipeline)
# ---------------------------------------------------------------------------


class TestSionnaExporter:
    """End-to-end test: scene + paths → pkl files on disk."""

    def test_files_created(self, simple_scene, computed_paths) -> None:
        """All six expected .pkl files must be created in the save folder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rt_params = {
                "max_depth": 2,
                "los": True,
                "specular_reflection": True,
                "diffuse_reflection": False,
                "refraction": True,
                "samples_per_src": 500_000,
            }
            sionna_exporter(simple_scene, computed_paths, rt_params, tmpdir)

            expected = {
                "sionna_paths.pkl",
                "sionna_materials.pkl",
                "sionna_material_indices.pkl",
                "sionna_rt_params.pkl",
                "sionna_vertices.pkl",
                "sionna_objects.pkl",
            }
            created = {p.name for p in Path(tmpdir).iterdir()}
            assert expected == created

    def test_paths_pkl_is_list(self, simple_scene, computed_paths) -> None:
        """sionna_paths.pkl must deserialise to a list of dicts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rt_params = {"max_depth": 2, "los": True, "specular_reflection": True}
            sionna_exporter(simple_scene, computed_paths, rt_params, tmpdir)
            paths_loaded = load_pickle(str(Path(tmpdir) / "sionna_paths.pkl"))
            assert isinstance(paths_loaded, list)
            assert isinstance(paths_loaded[0], dict)

    def test_single_paths_object_accepted(self, simple_scene, computed_paths) -> None:
        """A single (non-list) Paths object must not raise TypeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # This was the bug fixed in the 2.0 port
            sionna_exporter(simple_scene, computed_paths, {}, tmpdir)

    def test_pre_serialised_dicts_accepted(self, simple_scene, computed_paths) -> None:
        """Pre-serialised dict list (cpu_offload=True case) must be written as-is."""
        pre_serialised = export_paths(computed_paths)  # already a list of dicts
        with tempfile.TemporaryDirectory() as tmpdir:
            sionna_exporter(simple_scene, pre_serialised, {}, tmpdir)
            paths_loaded = load_pickle(str(Path(tmpdir) / "sionna_paths.pkl"))
            # Should be the same data, not double-exported
            assert len(paths_loaded) == len(pre_serialised)
