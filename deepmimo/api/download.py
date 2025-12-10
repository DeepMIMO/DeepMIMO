"""DeepMIMO API Download Module.

This module provides functionality for downloading scenarios and RT sources
from the DeepMIMO database.

Download flow:
1. Call download() with scenario name and optional output directory
2. Check if scenario already exists locally
3. If not:
   - Get secure download URL using _download_url()
   - Request download token and redirect URL from server
   - Download file using redirect URL with progress bar
   - Unzip downloaded file to scenarios directory
4. Return path to downloaded file
"""

import shutil
import time
from pathlib import Path

import requests
from tqdm import tqdm

from deepmimo.utils import (
    get_rt_source_folder,
    get_rt_sources_dir,
    get_scenario_folder,
    get_scenarios_dir,
    unzip,
)

from ._common import API_BASE_URL, HEADERS, REQUEST_TIMEOUT


def _download_url(scenario_name: str, *, rt_source: bool = False) -> str:
    """Get the secure download endpoint URL for a DeepMIMO scenario.

    Args:
        scenario_name: Name of the scenario ZIP file
        rt_source: Whether to download the raytracing source file

    Returns:
        Secure URL for downloading the scenario through the API endpoint

    Raises:
        ValueError: If scenario name is invalid
        RuntimeError: If server returns error

    """
    if not scenario_name.endswith(".zip"):
        scenario_name += ".zip"

    # Return the secure download endpoint URL with the filename as a parameter
    rt_param = "&rt_source=true" if rt_source else ""
    return f"{API_BASE_URL}/api/download/secure?filename={scenario_name}{rt_param}"


def download_url(scenario_name: str, *, rt_source: bool = False) -> str:
    """Public wrapper around `_download_url`."""
    return _download_url(scenario_name, rt_source=rt_source)


def _resolve_download_paths(
    scenario_name: str, output_dir: str | None, *, rt_source: bool
) -> tuple[str, str, str | None]:
    """Resolve download paths and check if content already exists.

    Args:
        scenario_name: Name of the scenario
        output_dir: Optional output directory
        rt_source: Whether downloading RT source

    Returns:
        Tuple of (download_dir, output_path, None) or (None, None, existing_path)
        If content exists, returns existing path in third element

    """
    if rt_source:
        download_dir = output_dir if output_dir else get_rt_sources_dir()
        output_path = str(Path(download_dir) / f"{scenario_name}_rt_source.zip")
        rt_source_folder = get_rt_source_folder(scenario_name)
        if Path(rt_source_folder).exists():
            print(f'RT source "{scenario_name}" already exists at {rt_source_folder}')
            return (None, None, rt_source_folder)
    else:
        download_dir = output_dir if output_dir else get_scenarios_dir()
        scenario_folder = get_scenario_folder(scenario_name)
        output_path = str(Path(download_dir) / f"{scenario_name}_downloaded.zip")
        if Path(scenario_folder).exists():
            print(f'Scenario "{scenario_name}" already exists in {get_scenarios_dir()}')
            return (None, None, scenario_folder)

    return (download_dir, output_path, None)


def _download_file_from_server(url: str, output_path: str) -> bool:
    """Download file from server with progress bar.

    Args:
        url: Download URL
        output_path: Path to save file

    Returns:
        True if successful, False otherwise

    """
    success = False
    try:
        # Get download token and redirect URL
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        token_data = resp.json()

        if "error" in token_data:
            print(f"Server error: {token_data.get('error')}")
            return success

        # Get and format redirect URL
        redirect_url = token_data.get("redirectUrl")
        if not redirect_url:
            print("Error: Missing redirect URL")
            return success

        if not redirect_url.startswith("http"):
            redirect_url = f"{url.split('/api/')[0]}{redirect_url}"

        # Download the file
        download_resp = requests.get(
            redirect_url, stream=True, headers=HEADERS, timeout=REQUEST_TIMEOUT
        )
        download_resp.raise_for_status()
        total_size = int(download_resp.headers.get("content-length", 0))

        with (
            Path(output_path).open("wb") as file,
            tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="Downloading",
            ) as pbar,
        ):
            for chunk in download_resp.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    pbar.update(len(chunk))

        print(f"✓ Downloaded to {output_path}")
        success = True

    except requests.exceptions.RequestException as e:
        print(f"Download failed: {e!s}")
        output_path_obj = Path(output_path)
        if output_path_obj.exists():
            output_path_obj.unlink()  # Clean up partial download

    return success


def _extract_and_move_rt_source(output_path: str, scenario_name: str) -> None:
    """Extract RT source files and move to RT sources directory.

    Args:
        output_path: Path to downloaded zip file
        scenario_name: Name of the scenario

    """
    rt_sources_dir = get_rt_sources_dir()
    time.sleep(0.5)  # wait for the file to be file lock to be lifted
    unzipped_folder = unzip(output_path)

    # Move extracted folder to RT sources directory
    rt_extracted_path = get_rt_source_folder(scenario_name)
    unzipped_folder_without_suffix = unzipped_folder.replace("_rt_source", "")

    Path(rt_sources_dir).mkdir(parents=True, exist_ok=True)

    # If the target already exists, remove it first
    if Path(rt_extracted_path).exists():
        shutil.rmtree(rt_extracted_path)

    # Rename the unzipped folder and move to RT sources directory
    if unzipped_folder != unzipped_folder_without_suffix:
        Path(unzipped_folder).rename(unzipped_folder_without_suffix)
    shutil.move(unzipped_folder_without_suffix, rt_extracted_path)

    print(f"✓ RT source files extracted to {rt_extracted_path}")
    print(f"✓ RT source '{scenario_name}' downloaded!")


def _extract_and_move_scenario(output_path: str, scenario_name: str) -> None:
    """Extract scenario files and move to scenarios folder.

    Args:
        output_path: Path to downloaded zip file
        scenario_name: Name of the scenario

    """
    scenarios_dir = get_scenarios_dir()
    scenario_folder = get_scenario_folder(scenario_name)
    unzipped_folder = unzip(output_path)

    # Handle nested directory structure
    unzipped_folder_without_suffix = unzipped_folder.replace("_downloaded", "")

    # Check if there's a nested directory with the scenario name
    unzipped_folder_path = Path(unzipped_folder)
    nested_path = unzipped_folder_path / scenario_name
    if nested_path.exists() and nested_path.is_dir():
        tmp_path = unzipped_folder + "_tmp"
        shutil.move(unzipped_folder, tmp_path)
        shutil.move(str(Path(tmp_path) / scenario_name), unzipped_folder)
        shutil.rmtree(tmp_path)
        print(f"✓ Flattened nested directory '{scenario_name}'")

    # Rename to remove suffix if needed
    Path(scenarios_dir).mkdir(parents=True, exist_ok=True)
    if Path(unzipped_folder).exists() and unzipped_folder != unzipped_folder_without_suffix:
        shutil.move(str(Path(unzipped_folder)), unzipped_folder_without_suffix)

    shutil.move(unzipped_folder_without_suffix, scenario_folder)
    print(f"✓ Unzipped and moved to {scenarios_dir}")
    print(f"✓ Scenario '{scenario_name}' ready to use!")


def download(
    scenario_name: str,
    output_dir: str | None = None,
    *,
    rt_source: bool = False,
) -> str | None:
    """Download a DeepMIMO scenario from the database.

    Args:
        scenario_name: Name of the scenario
        output_dir: Directory to save file (defaults to current directory)
        rt_source: Whether to download the raytracing source file instead of the scenario

    Returns:
        Path to downloaded file if successful, None otherwise

    """
    scenario_name = scenario_name.lower()

    # Resolve paths and check if already exists
    download_dir, output_path, existing_path = _resolve_download_paths(
        scenario_name, output_dir, rt_source=rt_source
    )
    if existing_path:
        return None

    # Download file if it doesn't exist
    if not Path(output_path).exists():
        Path(download_dir).mkdir(parents=True, exist_ok=True)
        download_type = "raytracing source" if rt_source else "scenario"
        print(f"Downloading {download_type} '{scenario_name}'")

        url = _download_url(scenario_name, rt_source=rt_source)
        if not _download_file_from_server(url, output_path):
            return None
    else:
        print(f'Scenario zip file "{output_path}" already exists.')

    time.sleep(.5)  # wait for the file lock to be lifted

    # Extract and move to appropriate location
    if rt_source:
        _extract_and_move_rt_source(output_path, scenario_name)
    else:
        _extract_and_move_scenario(output_path, scenario_name)

    return output_path

