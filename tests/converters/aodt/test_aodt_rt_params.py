"""Tests for AODT RT Parameters."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from deepmimo.converters.aodt import aodt_rt_params


@patch("deepmimo.converters.aodt.aodt_rt_params.pd")
def test_read_rt_params(mock_pd) -> None:
    # Mock db_info DataFrame
    mock_df = MagicMock()
    mock_df.__len__.return_value = 1  # Fix empty check
    mock_df.iloc.__getitem__.return_value = 28e9  # frequency
    mock_pd.read_parquet.return_value = mock_df

    # Mock run_info
    run_df = MagicMock()
    run_df.__len__.return_value = 1  # Fix empty check
    run_df.iloc.__getitem__.return_value = 1.0  # version?

    def read_parquet_side_effect(path):
        if "scenario" in str(path):
            df = MagicMock()
            df.__len__.return_value = 1
            # Mock Series return for iloc[0]
            mock_series = MagicMock()
            mock_series.__getitem__.side_effect = lambda k: {
                "num_scene_interactions_per_ray": 5,
                "num_emitted_rays_in_thousands": 1,
            }[k]
            mock_series.to_dict.return_value = {}

            df.iloc.__getitem__.return_value = mock_series
            return df
        # Default mock with length 1
        d = MagicMock()
        d.__len__.return_value = 1
        return d

    mock_pd.read_parquet.side_effect = read_parquet_side_effect

    with patch.object(Path, "exists", return_value=True):
        params = aodt_rt_params.read_rt_params("dummy_folder")
        assert params["raytracer_name"] == "Aerial Omniverse Digital Twin"
        # Frequency is currently 0 in AODT params
        assert params["frequency"] == 0
