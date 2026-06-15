"""End-to-end tests for the public MATLAB RT converter API."""

from __future__ import annotations

import shutil
import sys
import unittest
import uuid
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

import deepmimo as dm
from deepmimo import consts as c
from deepmimo.converters.matlab_rt import convert_matlab_rt_json
from deepmimo.converters.matlab_rt.errors import MatlabRTWriterError
from deepmimo.converters.matlab_rt.parser import parse_matlab_rt_json
from deepmimo.datasets.dataset import Dataset, MacroDataset
from tests.converters.matlab_rt import expect_raises

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
MULTILINK_FIXTURE = FIXTURE_DIR / "matlab_rt_multilink.json"


class TestMatlabRTConverterE2E(unittest.TestCase):
    """Validate the public converter against real DeepMIMO workflows."""

    def setUp(self) -> None:
        """Prepare an isolated scenario name under DeepMIMO's scenario root."""
        self.scenario_root = Path.cwd() / c.SCENARIOS_FOLDER
        self.scenario_root.mkdir(exist_ok=True)
        self.scenario_name = f"matlab_rt_e2e_{uuid.uuid4().hex[:12]}"
        self.scenario_path = self.scenario_root / self.scenario_name

    def tearDown(self) -> None:
        """Remove generated scenario output."""
        if self.scenario_path.exists():
            shutil.rmtree(self.scenario_path)

    def test_public_converter_end_to_end_workflow(self) -> None:
        """Converted MATLAB RT JSON loads, generates channels, beamforms, and plots rays."""
        sys.modules.pop("matlab.engine", None)

        result = convert_matlab_rt_json(
            MULTILINK_FIXTURE,
            scenario_root=self.scenario_root,
            scenario_name=self.scenario_name,
        )

        assert result.scenario_name == self.scenario_name
        assert (result.scenario_path / "params.json").exists()
        assert "matlab.engine" not in sys.modules

        macro = dm.load(self.scenario_name, max_paths=2)
        assert isinstance(macro, MacroDataset)
        assert len(macro) == 2
        np.testing.assert_array_equal(macro[0].num_paths, np.array([2, 0]))
        np.testing.assert_array_equal(macro[1].num_paths, np.array([0, 2]))

        dataset = dm.load(
            self.scenario_name,
            max_paths=2,
            tx_sets={0: [0]},
            rx_sets=[1],
        )
        assert isinstance(dataset, Dataset)
        assert dataset.power.shape == (2, 2)
        assert dataset.inter_pos.shape == (2, 2, 1, 3)
        assert "MATLAB Ray Tracing" in dm.summary(self.scenario_name, print_summary=False)

        td_params = dm.ChannelParameters(
            freq_domain=False,
            num_paths=2,
            bs_antenna={"shape": [1, 1]},
            ue_antenna={"shape": [1, 1]},
        )
        channel = dataset.compute_channels(td_params)
        assert channel.shape == (2, 1, 1, 2)
        assert np.isfinite(channel).all()
        assert np.linalg.norm(channel[0]) > 0.0
        assert np.linalg.norm(channel[1]) == 0.0

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
        assert ofdm_channel.shape == (2, 1, 1, 8)
        assert np.isfinite(ofdm_channel).all()

        bf_params = dm.ChannelParameters(
            freq_domain=False,
            num_paths=2,
            bs_antenna={"shape": [8, 1]},
            ue_antenna={"shape": [1, 1]},
        )
        bf_channel = dataset.compute_channels(bf_params)
        h = bf_channel[0, 0, :, 0]
        beamformer = h.conj() / np.linalg.norm(h)
        gain = abs(beamformer.conj() @ h) ** 2
        assert bf_channel.shape == (2, 1, 8, 2)
        assert np.isfinite(gain)
        assert gain > 0.0

        ax = dataset.plot_rays(0, proj_3D=False)
        assert len(ax.lines) > 0
        plt.close(ax.figure)

    def test_parsed_export_input_and_overwrite_policy(self) -> None:
        """The public API accepts parsed exports and refuses accidental overwrite."""
        export = parse_matlab_rt_json(MULTILINK_FIXTURE)

        convert_matlab_rt_json(
            export,
            scenario_root=self.scenario_root,
            scenario_name=self.scenario_name,
        )
        with expect_raises(MatlabRTWriterError):
            convert_matlab_rt_json(
                export,
                scenario_root=self.scenario_root,
                scenario_name=self.scenario_name,
            )

        result = convert_matlab_rt_json(
            export,
            scenario_root=self.scenario_root,
            scenario_name=self.scenario_name,
            overwrite=True,
        )
        assert (result.scenario_path / "power_t000_tx000_r001.npz").exists()


if __name__ == "__main__":
    unittest.main()
