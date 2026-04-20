"""Tests for Sionna Paths module (Sionna 2.0).

Covers:
- transform_interaction_types: LoS, reflections, remapping of all Sionna 2.0
  InteractionType values, edge cases (empty, all-NaN/zero, mixed).
- read_paths: end-to-end mock for the full load → process → save pipeline.
"""

from unittest.mock import patch

import numpy as np
import pytest

from deepmimo import consts as c
from deepmimo.converters.sionna_rt import sionna_paths
from deepmimo.converters.sionna_rt.sionna_paths import (
    _SIONNA_TO_DEEPMIMO,
    SIONNA_INTERACTION_DIFFRACTION,
    SIONNA_INTERACTION_DIFFUSE,
    SIONNA_INTERACTION_NONE,
    SIONNA_INTERACTION_REFRACTION,
    SIONNA_INTERACTION_SPECULAR,
    transform_interaction_types,
)

# ---------------------------------------------------------------------------
# transform_interaction_types
# ---------------------------------------------------------------------------


class TestTransformInteractionTypes:
    """Unit tests for the Sionna 2.0 → DeepMIMO interaction code converter."""

    def test_los_all_zeros(self) -> None:
        """All-zero depth array → INTERACTION_LOS (no bounces)."""
        types = np.array([[0, 0, 0]], dtype=np.float32)
        result = transform_interaction_types(types)
        assert result[0] == c.INTERACTION_LOS

    def test_single_specular_reflection(self) -> None:
        """SPECULAR(1) at depth 0 → DeepMIMO code 1."""
        types = np.array([[SIONNA_INTERACTION_SPECULAR, 0, 0]], dtype=np.float32)
        result = transform_interaction_types(types)
        assert result[0] == 1.0

    def test_two_specular_reflections(self) -> None:
        """Two SPECULAR bounces → code 11."""
        types = np.array([[1, 1, 0]], dtype=np.float32)
        result = transform_interaction_types(types)
        assert result[0] == 11.0

    def test_three_specular_reflections(self) -> None:
        """Three SPECULAR bounces → code 111."""
        types = np.array([[1, 1, 1]], dtype=np.float32)
        result = transform_interaction_types(types)
        assert result[0] == 111.0

    def test_diffuse_remapped_to_scattering(self) -> None:
        """DIFFUSE(2) → DeepMIMO SCATTERING code (3)."""
        types = np.array([[SIONNA_INTERACTION_DIFFUSE, 0, 0]], dtype=np.float32)
        result = transform_interaction_types(types)
        assert result[0] == float(c.INTERACTION_SCATTERING)

    def test_diffraction_remapped(self) -> None:
        """DIFFRACTION(8) → DeepMIMO DIFFRACTION code (2)."""
        types = np.array([[SIONNA_INTERACTION_DIFFRACTION, 0, 0]], dtype=np.float32)
        result = transform_interaction_types(types)
        assert result[0] == float(c.INTERACTION_DIFFRACTION)

    def test_refraction_remapped(self) -> None:
        """REFRACTION(4) → DeepMIMO TRANSMISSION code (4, unchanged value)."""
        types = np.array([[SIONNA_INTERACTION_REFRACTION, 0, 0]], dtype=np.float32)
        result = transform_interaction_types(types)
        assert result[0] == float(c.INTERACTION_TRANSMISSION)

    def test_reflection_then_scattering(self) -> None:
        """SPECULAR then DIFFUSE → code 13 (reflection=1, scattering=3)."""
        types = np.array([[1, 2, 0]], dtype=np.float32)
        result = transform_interaction_types(types)
        assert result[0] == 13.0

    def test_reflection_then_diffraction(self) -> None:
        """SPECULAR then DIFFRACTION → code 12."""
        types = np.array([[1, 8, 0]], dtype=np.float32)
        result = transform_interaction_types(types)
        assert result[0] == 12.0

    def test_diffraction_then_reflection(self) -> None:
        """DIFFRACTION then SPECULAR → code 21 (2 then 1)."""
        types = np.array([[8, 1, 0]], dtype=np.float32)
        result = transform_interaction_types(types)
        assert result[0] == 21.0

    def test_refraction_then_reflection(self) -> None:
        """REFRACTION then SPECULAR → code 41."""
        types = np.array([[4, 1, 0]], dtype=np.float32)
        result = transform_interaction_types(types)
        assert result[0] == 41.0

    def test_mixed_batch(self) -> None:
        """Multiple paths with different sequences in a single batch."""
        types = np.array(
            [
                [0, 0, 0],  # LoS
                [1, 0, 0],  # single reflection
                [1, 1, 0],  # two reflections
                [1, 2, 0],  # reflection + scatter
                [8, 0, 0],  # diffraction
                [1, 8, 0],  # reflection + diffraction
            ],
            dtype=np.float32,
        )
        result = transform_interaction_types(types)
        assert result[0] == float(c.INTERACTION_LOS)
        assert result[1] == 1.0
        assert result[2] == 11.0
        assert result[3] == 13.0
        assert result[4] == 2.0
        assert result[5] == 12.0

    def test_empty_array(self) -> None:
        """Zero-path input → empty output with correct shape."""
        types = np.zeros((0, 5), dtype=np.float32)
        result = transform_interaction_types(types)
        assert result.shape == (0,)

    def test_all_zeros_multiple_paths(self) -> None:
        """Multiple LoS paths → all INTERACTION_LOS."""
        types = np.zeros((4, 3), dtype=np.float32)
        result = transform_interaction_types(types)
        assert np.all(result == c.INTERACTION_LOS)

    def test_output_shape(self) -> None:
        """Output shape is (n_paths,) for any (n_paths, max_depth) input."""
        for n in [1, 5, 100]:
            types = np.zeros((n, 4), dtype=np.float32)
            assert transform_interaction_types(types).shape == (n,)

    def test_none_constant_ignored(self) -> None:
        """NONE(0) slots between non-zero values should not appear in the code.

        E.g. [1, 0, 1] should only encode the outermost non-zero range:
        trailing zeros are stripped, but the 0 between two bounces is kept as
        padding and must be excluded from digit concatenation.
        """
        # The slice goes up to the last non-zero index, then skips 0s inside
        types = np.array([[1, 0, 1]], dtype=np.float32)
        result = transform_interaction_types(types)
        # Valid raw values in range: [1, 0, 1]; non-zero: [1, 1] → code 11
        assert result[0] == 11.0


# ---------------------------------------------------------------------------
# Interaction constants sanity
# ---------------------------------------------------------------------------


def test_sionna_interaction_constants() -> None:
    """Sionna 2.0 enum values match what the remapping table expects."""
    assert SIONNA_INTERACTION_NONE == 0
    assert SIONNA_INTERACTION_SPECULAR == 1
    assert SIONNA_INTERACTION_DIFFUSE == 2
    assert SIONNA_INTERACTION_REFRACTION == 4
    assert SIONNA_INTERACTION_DIFFRACTION == 8


def test_remapping_table_completeness() -> None:
    """Every non-NONE Sionna 2.0 type maps to a known DeepMIMO code."""
    sionna_types = {
        SIONNA_INTERACTION_SPECULAR,
        SIONNA_INTERACTION_DIFFUSE,
        SIONNA_INTERACTION_REFRACTION,
        SIONNA_INTERACTION_DIFFRACTION,
    }
    deepmimo_codes = {
        c.INTERACTION_REFLECTION,
        c.INTERACTION_SCATTERING,
        c.INTERACTION_TRANSMISSION,
        c.INTERACTION_DIFFRACTION,
    }
    assert set(_SIONNA_TO_DEEPMIMO.keys()) == sionna_types
    assert set(_SIONNA_TO_DEEPMIMO.values()) == deepmimo_codes


# ---------------------------------------------------------------------------
# read_paths integration (mocked I/O)
# ---------------------------------------------------------------------------


@patch("deepmimo.converters.sionna_rt.sionna_paths.load_pickle")
def test_read_paths_sionna2(mock_load) -> None:
    """Load Sionna 2.0 paths and verify save_mat is called.

    Sionna 2.0 single-antenna array shapes (no batch dim):
    - a:            (num_rx, 1, num_tx, 1, max_paths)
    - tau/angles:   (num_rx, num_tx, max_paths)
    - interactions: (max_depth, num_rx, num_tx, max_paths)
    - vertices:     (max_depth, num_rx, num_tx, max_paths, 3)
    """
    n_rx, n_tx, max_paths, max_depth = 5, 1, 10, 5

    path_data = {
        "sources": np.zeros((n_tx, 3)),
        "targets": np.zeros((n_rx, 3)),
        "a": np.ones((n_rx, 1, n_tx, 1, max_paths)) * (1 + 1j),
        "tau": np.zeros((n_rx, n_tx, max_paths)),
        "theta_t": np.zeros((n_rx, n_tx, max_paths)),
        "phi_t": np.zeros((n_rx, n_tx, max_paths)),
        "theta_r": np.zeros((n_rx, n_tx, max_paths)),
        "phi_r": np.zeros((n_rx, n_tx, max_paths)),
        "interactions": np.zeros((max_depth, n_rx, n_tx, max_paths)),
        "vertices": np.zeros((max_depth, n_rx, n_tx, max_paths, 3)),
    }
    mock_load.return_value = [path_data]

    txrx_dict = {
        "txrx_set_0": {"is_tx": True, "id": 0, "num_points": n_tx, "num_ant": 1},
        "txrx_set_1": {"is_rx": True, "id": 1, "num_points": n_rx, "num_ant": 1},
    }

    with patch("deepmimo.converters.sionna_rt.sionna_paths.save_mat") as mock_save:
        sionna_paths.read_paths("dummy_folder", "out_folder", txrx_dict)
        assert mock_save.called


@patch("deepmimo.converters.sionna_rt.sionna_paths.load_pickle")
def test_read_paths_interaction_codes_stored(mock_load) -> None:
    """Verify interaction codes from paths are non-trivially stored.

    Uses a single RX with one specular-reflection path (SPECULAR=1 at depth 0)
    and checks the stored interaction code equals 1.0 (DeepMIMO REFLECTION).
    """
    n_rx, n_tx, max_paths, max_depth = 1, 1, 5, 3

    # One active path for the single RX: non-zero amplitude at path index 0
    a = np.zeros((n_rx, 1, n_tx, 1, max_paths), dtype=complex)
    a[0, 0, 0, 0, 0] = 1.0 + 0j

    interactions = np.zeros((max_depth, n_rx, n_tx, max_paths), dtype=np.float32)
    interactions[0, 0, 0, 0] = SIONNA_INTERACTION_SPECULAR  # one reflection at depth 0

    path_data = {
        # Use distinct TX/RX positions so the BS-BS-path guard does not trigger.
        "sources": np.array([[0.0, 0.0, 10.0]]),  # TX at height 10
        "targets": np.array([[5.0, 5.0, 1.5]]),  # RX at street level
        "a": a,
        "tau": np.zeros((n_rx, n_tx, max_paths)),
        "theta_t": np.zeros((n_rx, n_tx, max_paths)),
        "phi_t": np.zeros((n_rx, n_tx, max_paths)),
        "theta_r": np.zeros((n_rx, n_tx, max_paths)),
        "phi_r": np.zeros((n_rx, n_tx, max_paths)),
        "interactions": interactions,
        "vertices": np.zeros((max_depth, n_rx, n_tx, max_paths, 3)),
    }
    mock_load.return_value = [path_data]

    txrx_dict = {
        "txrx_set_0": {"is_tx": True, "id": 0, "num_points": n_tx, "num_ant": 1},
        "txrx_set_1": {"is_rx": True, "id": 1, "num_points": n_rx, "num_ant": 1},
    }

    saved = {}

    def capture_save(arr, key, _path):
        saved[key] = arr

    with patch("deepmimo.converters.sionna_rt.sionna_paths.save_mat", side_effect=capture_save):
        sionna_paths.read_paths("dummy_folder", "out_folder", txrx_dict)

    inter_key = c.INTERACTIONS_PARAM_NAME
    assert inter_key in saved, "interaction array was not saved"
    # Path 0 of the single RX should have code 1.0 (REFLECTION)
    assert saved[inter_key][0, 0] == pytest.approx(float(c.INTERACTION_REFLECTION))


@patch("deepmimo.converters.sionna_rt.sionna_paths.load_pickle")
def test_read_paths_los_code_stored(mock_load) -> None:
    """A path with all-zero interactions must be stored as INTERACTION_LOS."""
    n_rx, n_tx, max_paths, max_depth = 1, 1, 3, 2

    a = np.zeros((n_rx, 1, n_tx, 1, max_paths), dtype=complex)
    a[0, 0, 0, 0, 0] = 1.0 + 0j  # active LoS path

    path_data = {
        # Use distinct TX/RX positions so the BS-BS-path guard does not trigger.
        "sources": np.array([[0.0, 0.0, 10.0]]),  # TX at height 10
        "targets": np.array([[5.0, 5.0, 1.5]]),  # RX at street level
        "a": a,
        "tau": np.zeros((n_rx, n_tx, max_paths)),
        "theta_t": np.zeros((n_rx, n_tx, max_paths)),
        "phi_t": np.zeros((n_rx, n_tx, max_paths)),
        "theta_r": np.zeros((n_rx, n_tx, max_paths)),
        "phi_r": np.zeros((n_rx, n_tx, max_paths)),
        "interactions": np.zeros((max_depth, n_rx, n_tx, max_paths)),  # all zero → LoS
        "vertices": np.zeros((max_depth, n_rx, n_tx, max_paths, 3)),
    }
    mock_load.return_value = [path_data]

    txrx_dict = {
        "txrx_set_0": {"is_tx": True, "id": 0, "num_points": n_tx, "num_ant": 1},
        "txrx_set_1": {"is_rx": True, "id": 1, "num_points": n_rx, "num_ant": 1},
    }

    saved = {}

    def capture_save(arr, key, _path):
        saved[key] = arr

    with patch("deepmimo.converters.sionna_rt.sionna_paths.save_mat", side_effect=capture_save):
        sionna_paths.read_paths("dummy_folder", "out_folder", txrx_dict)

    inter_key = c.INTERACTIONS_PARAM_NAME
    assert inter_key in saved
    assert saved[inter_key][0, 0] == pytest.approx(float(c.INTERACTION_LOS))
