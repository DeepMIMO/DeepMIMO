"""Tests for DeepMIMO API module."""

import unittest
from unittest.mock import patch, MagicMock
import os
import pytest
import requests

from deepmimo import api

class TestDeepMIMOAPI(unittest.TestCase):
    
    @patch('deepmimo.api.requests.get')
    def test_dm_upload_api_call_auth_fail(self, mock_get):
        """Test upload authorization failure."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Unauthorized")
        mock_get.return_value = mock_response
        
        # Create a dummy file
        with open("test_file.zip", "w") as f:
            f.write("dummy content")
            
        try:
            result = api._dm_upload_api_call("test_file.zip", "fake_key")
            assert result is None
        finally:
            os.remove("test_file.zip")

    @patch('deepmimo.api.requests.get')
    @patch('deepmimo.api.requests.put')
    def test_dm_upload_api_call_success(self, mock_put, mock_get):
        """Test successful upload."""
        # Mock auth response
        auth_response = MagicMock()
        auth_response.status_code = 200
        auth_response.json.return_value = {
            "presignedUrl": "http://upload.url",
            "filename": "test_file.zip",
            "contentType": "application/zip"
        }
        mock_get.return_value = auth_response
        
        # Mock upload response
        upload_response = MagicMock()
        upload_response.status_code = 200
        mock_put.return_value = upload_response
        
        # Create a dummy file
        with open("test_file.zip", "w") as f:
            f.write("dummy content")
            
        try:
            result = api._dm_upload_api_call("test_file.zip", "fake_key")
            assert result == "test_file.zip"
        finally:
            os.remove("test_file.zip")

    def test_process_params_data(self):
        """Test parameter processing."""
        params_dict = {
            "version": "4.0.0",
            "rt_params": {
                "frequency": 28e9,
                "raytracer_name": "Insite",
                "max_reflections": 3
            },
            "txrx_sets": {
                "set1": {"num_active_points": 10, "is_rx": True, "num_ant": 1},
                "set2": {"num_active_points": 1, "is_tx": True, "num_ant": 64}
            }
        }
        
        result = api._process_params_data(params_dict)
        
        primary = result["primaryParameters"]
        assert primary["bands"]["mmW"] is True
        assert primary["numRx"] == 10
        assert primary["maxReflections"] == 3
        assert primary["raytracerName"] == "Insite"

    def test_generate_key_components(self):
        """Test key component generation from summary string."""
        summary_str = """
[Section 1]
Description line 1
Description line 2

[Section 2]
Title line
- Item 1
- Item 2
"""
        result = api._generate_key_components(summary_str)
        assert len(result["sections"]) == 2
        assert result["sections"][0]["name"] == "Section 1"
        assert result["sections"][1]["name"] == "Section 2"

    @patch('deepmimo.api.requests.post')
    def test_search_success(self, mock_post):
        """Test search functionality."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"scenarios": ["scenario1", "scenario2"]}
        mock_post.return_value = mock_response
        
        query = {"bands": ["mmW"]}
        result = api.search(query)
        
        assert result == ["scenario1", "scenario2"]
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs["json"] == query

    @patch('deepmimo.api.requests.post')
    def test_search_failure(self, mock_post):
        """Test search failure."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Server Error")
        mock_post.return_value = mock_response
        
        result = api.search({})
        assert result is None

    def test_download_url(self):
        """Test download URL generation."""
        url = api._download_url("test_scen", rt_source=False)
        assert "filename=test_scen.zip" in url
        assert "rt_source=true" not in url
        
        url_rt = api._download_url("test_scen", rt_source=True)
        assert "rt_source=true" in url_rt

    @patch('deepmimo.api.requests.post')
    def test_upload_images(self, mock_post):
        """Test image upload."""
        # Create dummy image
        with open("test_img.png", "wb") as f:
            f.write(b"fake png content")
            
        try:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"id": "img1"}
            mock_post.return_value = mock_response
            
            results = api.upload_images("test_scen", ["test_img.png"], "key")
            assert len(results) == 1
            assert results[0]["id"] == "img1"
        finally:
            if os.path.exists("test_img.png"):
                os.remove("test_img.png")

    @patch('deepmimo.api.requests.get')
    @patch('deepmimo.api.requests.put')
    def test_upload_rt_source(self, mock_put, mock_get):
        """Test RT source upload."""
        # Create dummy zip
        with open("test_rt.zip", "wb") as f:
            f.write(b"fake zip content")
            
        try:
            # Mock auth
            mock_auth_resp = MagicMock()
            mock_auth_resp.status_code = 200
            mock_auth_resp.json.return_value = {
                "presignedUrl": "http://upload.url",
                "filename": "test_scen.zip",
                "contentType": "application/zip"
            }
            mock_get.return_value = mock_auth_resp
            
            # Mock upload
            mock_upload_resp = MagicMock()
            mock_upload_resp.status_code = 200
            mock_put.return_value = mock_upload_resp
            
            result = api.upload_rt_source("test_scen", "test_rt.zip", "key")
            assert result is True
        finally:
             if os.path.exists("test_rt.zip"):
                os.remove("test_rt.zip")
