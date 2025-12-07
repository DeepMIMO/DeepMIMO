"""Advanced tests for DeepMIMO API."""

import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import json
from deepmimo import api

class TestDeepMIMOAPIAdvanced(unittest.TestCase):
    
    @patch('deepmimo.api.zip')
    @patch('deepmimo.api._dm_upload_api_call')
    @patch('deepmimo.api._make_submission_on_server')
    @patch('deepmimo.api.load_dict_from_json')
    @patch('deepmimo.api.get_scenario_folder')
    @patch('deepmimo.api.get_params_path')
    def test_upload_flow(self, mock_get_params, mock_get_folder, mock_load_json, mock_make_sub, mock_upload_api, mock_zip):
        """Test the high-level upload function."""
        mock_get_folder.return_value = "scenarios/MyScenario"
        mock_get_params.return_value = "scenarios/MyScenario/params.json"
        mock_load_json.return_value = {"version": "4.0"}
        mock_upload_api.return_value = "MyScenario.zip"
        # DeepMIMO lowercases scenario names
        mock_make_sub.return_value = "myscenario"
        mock_zip.return_value = "scenarios/MyScenario.zip"
        
        # Test normal upload
        # res comes from _upload_to_db -> _dm_upload_api_call mock -> "MyScenario.zip" -> "MyScenario"
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
        with self.assertRaises(RuntimeError):
            api.upload("MyScenario", "key")

    @patch('deepmimo.api.requests.post')
    @patch('deepmimo.api.summary')
    @patch('deepmimo.api._process_params_data')
    @patch('deepmimo.api._generate_key_components')
    @patch('deepmimo.api.plot_summary')
    @patch('deepmimo.api.upload_images')
    def test_make_submission_on_server(self, mock_upl_imgs, mock_plot, mock_gen_key, mock_proc_params, mock_summary, mock_post):
        """Test making submission on server."""
        mock_proc_params.return_value = {"primaryParameters": {}, "advancedParameters": {}}
        mock_gen_key.return_value = {"sections": []}
        mock_post.return_value.status_code = 200
        mock_plot.return_value = ["img.png"]
        mock_upl_imgs.return_value = [{}]
        
        # Test success
        # The function returns the input scenario name unchanged
        res = api._make_submission_on_server(
            "MyScenario", "key", {}, ["detail"], {}, include_images=True
        )
        assert res == "MyScenario"
        mock_upl_imgs.assert_called()
        
        # Test failure
        mock_post.return_value.raise_for_status.side_effect = Exception("Server error")
        mock_post.return_value.text = '{"error": "msg"}'
        with self.assertRaises(RuntimeError):
            api._make_submission_on_server(
                "MyScenario", "key", {}, [], {}, include_images=False
            )

    @patch('deepmimo.api.requests.get')
    @patch('deepmimo.api.get_scenarios_dir')
    @patch('deepmimo.api.get_scenario_folder')
    @patch('deepmimo.api.unzip')
    @patch('deepmimo.api.shutil.move')
    @patch('deepmimo.api.os.rename')
    @patch('deepmimo.api.os.path.exists')
    @patch('deepmimo.api.os.makedirs')
    def test_download_scenario(self, mock_makedirs, mock_exists, mock_rename, mock_move, mock_unzip, mock_get_folder, mock_get_scenarios_dir, mock_get):
        """Test downloading a scenario."""
        mock_get_scenarios_dir.return_value = "scenarios"
        mock_get_folder.return_value = "scenarios/myscenario"
        
        # Case 1: Scenario already exists
        mock_exists.side_effect = lambda p: p == "scenarios/myscenario"
        res = api.download("MyScenario")
        assert res is None # Should return None if exists
        
        # Case 2: Download
        mock_exists.side_effect = lambda p: False # Not exist
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"redirectUrl": "http://dl.url"}
        mock_resp.headers.get.return_value = "100"
        mock_resp.iter_content.return_value = [b"data"]
        mock_get.return_value = mock_resp
        
        mock_unzip.return_value = "myscenario_downloaded"
        
        with patch("builtins.open", mock_open()):
            res = api.download("MyScenario")
            # The download logic converts name to lowercase
            assert "myscenario_downloaded.zip" in res
            mock_unzip.assert_called()
            mock_move.assert_called()

    def test_format_section(self):
        """Test HTML formatting of summary sections."""
        lines = ["Header", "- Item 1", "- Item 2", "", "Para"]
        res = api._format_section("Test", lines)
        assert "<h4>Header</h4>" in res["description"]
        assert "<li>Item 1</li>" in res["description"]
        assert "<p>Para</p>" in res["description"]
