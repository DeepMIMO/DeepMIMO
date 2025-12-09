"""Advanced tests for DeepMIMO API."""

import unittest
from unittest.mock import MagicMock, mock_open, patch

import pytest

from deepmimo import api


class TestDeepMIMOAPIAdvanced(unittest.TestCase):
    """Advanced API behaviors for uploads and downloads."""

    @patch("deepmimo.api.zip")
    @patch("deepmimo.api._dm_upload_api_call")
    def test_upload_flow(self, mock_upload_api, mock_zip) -> None:
        """Test the high-level upload function."""
        with (
            patch("deepmimo.api.get_params_path") as mock_get_params,
            patch(
                "deepmimo.api.get_scenario_folder",
            ) as mock_get_folder,
            patch("deepmimo.api.load_dict_from_json") as mock_load_json,
            patch(
                "deepmimo.api.make_submission_on_server",
            ) as mock_make_sub,
        ):
            mock_get_folder.return_value = "scenarios/MyScenario"
            mock_get_params.return_value = "scenarios/MyScenario/params.json"
            mock_load_json.return_value = {"version": "4.0"}
            mock_upload_api.return_value = "MyScenario.zip"
            # DeepMIMO lowercases scenario names
            mock_make_sub.return_value = "myscenario"
            mock_zip.return_value = "scenarios/MyScenario.zip"

            # Test normal upload
            # res comes from _upload_to_db -> _dm_upload_api_call mock
            # -> "MyScenario.zip" -> "MyScenario"
            res = api.upload("MyScenario", "key")
            assert res == "MyScenario"
            mock_upload_api.assert_called()
            mock_make_sub.assert_called()

            # Test submission only
            mock_upload_api.reset_mock()
            # res comes from scenario_name.lower()
            res = api.upload("MyScenario", "key", submission_only=True)
            assert res == "myscenario"
            mock_upload_api.assert_not_called()

            # Test parsing error
            mock_load_json.side_effect = Exception("Parse error")
            with pytest.raises(RuntimeError):
                api.upload("MyScenario", "key")

    @patch("deepmimo.api.plot_summary")
    @patch("deepmimo.api.upload_images")
    def test_make_submission_on_server(self, mock_upl_imgs, mock_plot) -> None:
        """Test making submission on server."""
        with (
            patch("deepmimo.api._process_params_data") as mock_proc_params,
            patch(
                "deepmimo.api._generate_key_components",
            ) as mock_gen_key,
            patch("deepmimo.api.summary") as mock_summary,
            patch(
                "deepmimo.api.requests.post",
            ) as mock_post,
        ):
            mock_proc_params.return_value = {"primaryParameters": {}, "advancedParameters": {}}
            mock_gen_key.return_value = {"sections": []}
            mock_post.return_value.status_code = 200
            mock_plot.return_value = ["img.png"]
            mock_upl_imgs.return_value = [{}]

            # Test success
            # The function returns the input scenario name unchanged
            config = {
                "params_dict": {},
                "details": ["detail"],
                "extra_metadata": {},
                "include_images": True,
            }
            res = api.make_submission_on_server("MyScenario", "key", config)
            assert res == "MyScenario"
            mock_upl_imgs.assert_called()
            mock_summary.assert_called()

            # Test failure
            mock_post.return_value.raise_for_status.side_effect = Exception("Server error")
            mock_post.return_value.text = '{"error": "msg"}'
            config = {
                "params_dict": {},
                "details": [],
                "extra_metadata": {},
                "include_images": False,
            }
            with pytest.raises(RuntimeError):
                api.make_submission_on_server("MyScenario", "key", config)

    @patch("deepmimo.api.get_scenarios_dir", return_value="scenarios")
    @patch("deepmimo.api.get_scenario_folder", return_value="scenarios/myscenario")
    @patch("deepmimo.api.unzip")
    @patch("deepmimo.api.shutil.move")
    @patch("deepmimo.api.Path")
    def test_download_scenario_existing(
        self,
        mock_path,
        mock_move,
        mock_unzip,
        mock_get_folder,
        mock_get_scenarios_dir,
    ) -> None:
        """Scenario already exists: download should no-op."""

        class MockPath:
            def __init__(self, path_str) -> None:
                self.path_str = str(path_str)

            def __str__(self) -> str:
                return self.path_str

            def __truediv__(self, other):
                return MockPath(f"{self.path_str}/{other}")

            def exists(self):
                return self.path_str == "scenarios/myscenario"

            def mkdir(self, **kwargs) -> None:
                pass

            def open(self, mode="r"):
                return mock_open()(self.path_str, mode)

            @property
            def name(self):
                return self.path_str.split("/")[-1]

            @property
            def parent(self):
                return MockPath("/".join(self.path_str.split("/")[:-1]))

        mock_path.side_effect = MockPath

        res = api.download("MyScenario")
        assert res is None
        mock_unzip.assert_not_called()
        mock_move.assert_not_called()
        mock_get_scenarios_dir.assert_called()
        mock_get_folder.assert_called()

    @patch("deepmimo.api.get_scenarios_dir", return_value="scenarios")
    @patch("deepmimo.api.get_scenario_folder", return_value="scenarios/myscenario")
    @patch("deepmimo.api.unzip")
    @patch("deepmimo.api.shutil.move")
    @patch("deepmimo.api.Path")
    def test_download_scenario_downloads(
        self,
        mock_path,
        mock_move,
        mock_unzip,
        mock_get_folder,
        mock_get_scenarios_dir,
    ) -> None:
        """Scenario missing: download and unzip it."""

        class MockPath:
            def __init__(self, path_str) -> None:
                self.path_str = str(path_str)

            def __str__(self) -> str:
                return self.path_str

            def __truediv__(self, other):
                return MockPath(f"{self.path_str}/{other}")

            def exists(self):
                return False

            def mkdir(self, **kwargs) -> None:
                pass

            def open(self, mode="r"):
                return mock_open()(self.path_str, mode)

            @property
            def name(self):
                return self.path_str.split("/")[-1]

            @property
            def parent(self):
                return MockPath("/".join(self.path_str.split("/")[:-1]))

            def stat(self):
                return type("stat", (), {"st_size": 1000})()

        mock_path.side_effect = MockPath

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"redirectUrl": "http://dl.url"}
        mock_resp.headers.get.return_value = "100"
        mock_resp.iter_content.return_value = [b"data"]

        with patch("deepmimo.api.requests.get", return_value=mock_resp):
            mock_unzip.return_value = "myscenario_downloaded"

            with patch("builtins.open", mock_open()):
                res = api.download("MyScenario")

        assert "myscenario_downloaded.zip" in res
        mock_unzip.assert_called()
        mock_move.assert_called()
        mock_get_scenarios_dir.assert_called()
        mock_get_folder.assert_called()

    def test_format_section(self) -> None:
        """Test HTML formatting of summary sections."""
        lines = ["Header", "- Item 1", "- Item 2", "", "Para"]
        res = api.format_section("Test", lines)
        assert "<h4>Header</h4>" in res["description"]
        assert "<li>Item 1</li>" in res["description"]
        assert "<p>Para</p>" in res["description"]
