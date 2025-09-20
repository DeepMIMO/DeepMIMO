"""
API utility functions for the DeepMIMO dataset generation.

This module provides functions for uploading and downloading DeepMIMO scenarios
from the DeepMIMO server.

Upload flow:
1. Call upload() with scenario name, key, and optional parameters (details, extra_metadata, etc.)
2. If not submission_only:
   - _upload_to_db() is called to upload the scenario zip file, handling:
     * Get presigned URL for upload
     * Calculate file hash
     * Upload file to the database
     * Return authorized filename
3. _make_submission_on_server() creates the submission with:
   - Process parameters using _process_params_data() - used scenario filtering in database
   - Generate key components using _generate_key_components() - used for scenario info on website
   - Create submission on server with processed data
   - If include_images is True:
     * Generate images using plot_summary()
     * Upload images using upload_images()

Download flow:
1. Call download() with scenario name and optional output directory
2. Check if scenario already exists locally
3. If not:
   - Get secure download URL using _download_url()
   - Request download token and redirect URL from server
   - Download file using redirect URL with progress bar
   - Unzip downloaded file to scenarios directory
4. Return path to downloaded file

Search flow:
1. Call search() with query dictionary containing search parameters
2. Send request to /api/search/scenarios endpoint
3. Return list of matching scenario names if successful
4. Use returned scenario names to download and load scenarios:
   for scenario_name in search(query):
       dm.download(scenario_name)
       dataset = dm.load(scenario_name)

"""

import os
import shutil
import requests
import hashlib
from tqdm import tqdm
from typing import Dict, Optional, List
from . import consts as c
from .general_utils import (
    get_scenarios_dir,
    get_scenario_folder,
    get_params_path,
    load_dict_from_json,
    zip,
    unzip
)
from .summary import summary, plot_summary

import json

API_BASE_URL = "https://deepmimo.net"

# Headers for HTTP requests
HEADERS = {
    'User-Agent': 'DeepMIMO-Python/4.0',
    'Accept': '*/*'
}

FILE_SIZE_LIMIT = 1 * 1024 ** 3    # Scenario zip file size limit: 1GB
RT_FILE_SIZE_LIMIT = 5 * 1024 ** 3 # RT source zip file size limit: 5GB
IMAGE_SIZE_LIMIT = 10 * 1024 ** 2  # Image size limit: 10MB

class _ProgressFileReader:
    """Progress file reader for uploading files to the DeepMIMO API."""
    def __init__(self, file_path, progress_bar):
        self.file_path = file_path
        self.progress_bar = progress_bar
        self.file_object = open(file_path, 'rb')
        self.len = os.path.getsize(file_path)
        self.bytes_read = 0

    def read(self, size=-1):
        data = self.file_object.read(size)
        self.bytes_read += len(data)
        self.progress_bar.n = self.bytes_read
        self.progress_bar.refresh()
        return data

    def close(self):
        self.file_object.close()

def _dm_upload_api_call(file: str, key: str) -> Optional[str]:
    """Upload a file to the DeepMIMO API server.
    
    Args:
        file (str): Path to file to upload
        key (str): API authentication key
        
    Returns:
        Optional[str]: Filename if successful,
                       None if upload fails
        
    Notes:
        Uses chunked upload with progress bar for large files.
        Handles file upload only, no longer returns direct download URLs.
    """
    try:
        # Get file info
        filename = os.path.basename(file)
        file_size = os.path.getsize(file)

        if file_size > FILE_SIZE_LIMIT:
            print(f"Error: File size limit of {FILE_SIZE_LIMIT / 1024 ** 3} GB exceeded.")
            return None

        # Get presigned upload URL with filename validation built-in
        auth_response = requests.get(
            f"{API_BASE_URL}/api/b2/authorize-upload",
            params={"filename": filename},  # Use the actual filename from the file
            headers={"Authorization": f"Bearer {key}"},
        )
        auth_response.raise_for_status()
        auth_data = auth_response.json()

        if not auth_data.get("presignedUrl"):
            print("Error: Invalid authorization response")
            return None

        # Verify the authorized filename matches our source filename
        authorized_filename = auth_data.get("filename")
        if authorized_filename and authorized_filename != filename:
            print(f"Error: Filename mismatch. Server authorized '{authorized_filename}' but trying to upload '{filename}'")
            return None

        # Calculate file hash
        sha1 = hashlib.sha1()
        with open(file, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha1.update(chunk)
        file_hash = sha1.hexdigest()

        # Upload file to DB
        print(f"Uploading {authorized_filename} to DB...")
        pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc="Uploading")
        
        try:
            progress_reader = _ProgressFileReader(file, pbar)
            
            # Use the presigned URL for upload
            upload_response = requests.put(
                auth_data["presignedUrl"],
                headers={
                    "Content-Type": auth_data.get("contentType", "application/zip"),
                    "Content-Length": str(file_size),
                    "X-Bz-Content-Sha1": file_hash,
                },
                data=progress_reader
            )
            upload_response.raise_for_status()
        finally:
            progress_reader.close()
            pbar.close()


        # Return the authorized filename (not the local filename)
        # This ensures we're consistent with what was actually uploaded
        if upload_response.status_code == 200:
            return authorized_filename or filename
        else:
            return None

    except requests.exceptions.HTTPError as e: # Catch HTTPError specifically
        print(f"API call failed: {str(e)}") # Print standard HTTP error
        if e.response is not None:
            try:
                # Try to parse the JSON response from the server
                error_data = e.response.json()
                # Extract the specific error message using the 'error' key
                server_message = error_data.get('error', 'No specific error message found in JSON.')
                print(f"Server Error ({e.response.status_code}): {server_message}")
            except ValueError: # Handle cases where response body isn't valid JSON
                print(f"Server Response ({e.response.status_code}): {e.response.text}") # Fallback to raw text
        else:
            print("API call failed without receiving a response from the server.")
        return None
    except requests.exceptions.RequestException as e: # Catch other network/request errors
        print(f"API call failed: {str(e)}")
        if hasattr(e, "response") and e.response:
            print(f"Server response: {json.loads(e.response.text)['error']}")
        return None
    except Exception as e:
        print(f"Upload failed due to an unexpected error: {str(e)}")
        return None

def _process_params_data(params_dict: Dict, extra_metadata: Optional[Dict] = None) -> Dict:
    """Process params.mat data into submission format - used in DeepMIMO database.

    Args:
        params_dict: Dictionary containing parsed params.mat data
        extra_metadata: Optional dictionary with additional metadata fields

    Returns:
        Processed parameters in submission format
    """
    rt_params = params_dict.get(c.RT_PARAMS_PARAM_NAME, {})
    txrx_sets = params_dict.get(c.TXRX_PARAM_NAME, {})
    scene_params = params_dict.get(c.SCENE_PARAM_NAME, {})

    # Convert frequency from Hz to GHz
    frequency = float(rt_params.get("frequency", 3.5e9)) / 1e9

    # Count total Tx and Rx
    num_tx = sum(set_info.get("num_active_points", 0)
                 for set_info in txrx_sets.values()
                 if set_info.get("is_tx")) or 1
    
    num_rx = sum(set_info.get("num_active_points", 0)
                 for set_info in txrx_sets.values()
                 if set_info.get("is_rx")) or 1

    raytracer_map = {
        c.RAYTRACER_NAME_WIRELESS_INSITE: "Insite",
        c.RAYTRACER_NAME_SIONNA: "Sionna",
        c.RAYTRACER_NAME_AODT: "AODT",
    }

    # Create base parameter dictionaries
    primary_params = {
        "bands": {
            "sub6": frequency >= 0 and frequency < 6,
            "mmW": frequency >= 6 and frequency <= 100,
            "subTHz": frequency > 100,
        },
        "numRx": num_rx,
        "maxReflections": rt_params.get("max_reflections", 1),
        "raytracerName": raytracer_map.get(rt_params.get("raytracer_name"), "Insite"),
        "environment": "outdoor",
    }

    advanced_params = {
        "dmVersion": params_dict.get("version", "4.0.0a"),
        "numTx": num_tx,
        "multiRxAnt": any(set_info.get("num_ant", 0) > 1 for set_info in txrx_sets.values()
                           if set_info.get("is_rx")),
        "multiTxAnt": any(set_info.get("num_ant", 0) > 1 for set_info in txrx_sets.values()
                           if set_info.get("is_tx")),
        "dualPolarization": any(set_info.get("dual_pol", False)
                                for set_info in txrx_sets.values()),
        "BS2BS": any(set_info.get("is_tx") and set_info.get("is_rx")
                     for set_info in txrx_sets.values()) or None,
        "pathDepth": rt_params.get("max_path_depth", None),
        "diffraction": bool(rt_params.get("max_diffractions", 0)),
        "scattering": bool(rt_params.get("max_scattering", 0)),
        "transmission": bool(rt_params.get("max_transmissions", 0)),
        "numRays": rt_params.get("num_rays", 1000000),
        "city": None,
        "digitalTwin": False,
        "dynamic": scene_params.get("num_scenes", 1) > 1,
        "bbCoords": None
    }

    # Override with extra metadata if provided
    if extra_metadata:
        for param in extra_metadata:
            if param in primary_params:
                primary_params[param] = extra_metadata[param]
            elif param in advanced_params:
                advanced_params[param] = extra_metadata[param]

    return {
        "primaryParameters": primary_params,
        "advancedParameters": advanced_params
    }

def _generate_key_components(summary_str: str) -> Dict:
    """Generate key components sections from summary string.

    Args:
        summary_str: Summary string from scenario containing sections in [Section Name] format
                    followed by their descriptions

    Returns:
        Dictionary containing sections with their names and HTML-formatted descriptions
    """
    html_dict = {"sections": []}
    current_section = None
    current_lines = []
    
    for line in summary_str.split('\n'):
        line = line.strip()
        if not line or line.startswith('='):  # Skip empty lines and separator lines
            continue
            
        if line.startswith('[') and line.endswith(']'):
            # Process previous section if it exists
            if current_section:
                html_dict["sections"].append(_format_section(current_section, current_lines))
            
            # Start new section
            current_section = line[1:-1]
            current_lines = []
        elif current_section:
            current_lines.append(line)
    
    # Add the final section
    if current_section:
        html_dict["sections"].append(_format_section(current_section, current_lines))

    return html_dict

def _format_section(name: str, lines: list) -> dict:
    """Format a section's content into proper HTML with consistent styling.

    Args:
        name: Section name
        lines: List of content lines for the section

    Returns:
        Formatted section dictionary with name and HTML description
    """
    # Group content by subsections (lines starting with newline)
    subsections = []
    current_subsection = []
    
    for line in lines:
        if line and not line.startswith('-'):  # New subsection header
            if current_subsection:
                subsections.append(current_subsection)
            current_subsection = [line]
        elif line:  # Content line
            current_subsection.append(line)
    
    if current_subsection:
        subsections.append(current_subsection)

    # Build HTML content
    html_parts = []
    for subsection in subsections:
        if len(subsection) == 1:  # Single line - use paragraph
            html_parts.append(f"<p>{subsection[0]}</p>")
        else:  # Multiple lines - use header and list
            header = subsection[0]
            items = [line[2:] for line in subsection[1:]]  # Remove "- " prefix
            
            html_parts.append(f"<h4>{header}</h4>")
            html_parts.append("<ul>")
            html_parts.extend(f"<li>{item}</li>" for item in items)
            html_parts.append("</ul>")

    return {
        "name": name,
        "description": f"""
            <div class="section-content">
                {''.join(html_parts)}
            </div>
        """
    }

def upload_images(scenario_name: str, img_paths: list[str], key: str) -> list[dict]:
    """Upload images and attach them to an existing scenario.
    
    Args:
        scenario_name: Name of the scenario to attach images to
        img_paths: List of paths to image files
        key: API authentication key
    
    Returns:
        List of image objects that were successfully uploaded and attached
    """
    
    if not img_paths:
        print("No images provided for upload")
        return []

    if (len(img_paths) > 5):
        print("Warning: You cannot upload more than 5 images to a submission.")
        return [] 

    uploaded_image_objects = []
    # Endpoint URL structure
    upload_url_template = f"{API_BASE_URL}/api/submissions/{scenario_name}/images"
    
    # Image type mapping for default titles/descriptions
    image_types = {
        # 'los.png': {
        #     'heading': 'Line of Sight',
        #     'description': 'Line of sight coverage for the scenario'
        # },
        # 'power.png': {
        #     'heading': 'Power Distribution',
        #     'description': 'Signal power distribution across the scenario'
        # },
        'scene.png': {
            'heading': 'Scenario Layout',
            'description': 'Physical layout of the scenario'
        }
    }

    print(f"Attempting to upload {len(img_paths)} images for scenario '{scenario_name}'...")

    # Initialize tqdm manually before the loop
    pbar = tqdm(total=len(img_paths), desc="Uploading images", unit="image")
    
    # Iterate directly over img_paths
    for i, img_path in enumerate(img_paths):
        filename = os.path.basename(img_path)
        filesize = os.path.getsize(img_path)

        if filesize > IMAGE_SIZE_LIMIT:
            print(f"Warning: Image {filename} is too large to upload. Skipping...")
            continue

        try:
            # Get default metadata or create generic ones
            default_info = image_types.get(filename, {
                'heading': f"Image {i + 1}",
                'description': f"Visualization {i + 1} for {scenario_name}"
            })

            # Prepare form data
            with open(img_path, 'rb') as img_file:
                files = {'image': (filename, img_file, 'image/png')} # Key is 'image' now
                data = {
                    'heading': default_info['heading'],
                    'description': default_info['description']
                }

                # Make the POST request to the new endpoint for each image
                response = requests.post(
                    upload_url_template,
                    headers={"Authorization": f"Bearer {key}"},
                    files=files,
                    data=data # Send heading/description in form data
                )

            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

            # If successful, server returns the metadata of the uploaded image
            result = response.json()
            uploaded_image_objects.append(result)
            print(f"✓ Successfully uploaded and attached: {filename}")
            
            # Update the progress bar ONLY after successful upload
            pbar.update(1)
        except Exception as e:
            if e.response is not None: 
                server_message = json.loads(e.response.text)["error"]
                print(f"✗ Failed to upload {filename}: {server_message} "
                      f"(Server Response Code: {e.response.status_code})")
            else:
                # Handle cases where the error didn't have a response object
                 print(f"✗ Failed to upload {filename}: {e}")

    # Close the progress bar after the loop finishes or breaks
    pbar.close()

    if uploaded_image_objects:
         print(f"✓ Finished image upload process. Successfully attached {len(uploaded_image_objects)} images.")
    else:
         print("No images were successfully attached.")

    return uploaded_image_objects

def _upload_to_db(scen_folder: str, key: str, skip_zip: bool = False) -> str:
    """Upload a zip file to the database."""

    # Zip scenario
    zip_path = scen_folder + ".zip" if skip_zip else zip(scen_folder)

    try:
        print("Uploading to the database...")
        upload_result = _dm_upload_api_call(zip_path, key)
    except Exception as e:
        print(f"Error: Failed to upload to the database - {str(e)}")

    if not upload_result:
        print(f"Error: Failed to upload to the database")
        raise RuntimeError("Failed to upload to the database")
    print("✓ Upload successful")

    submission_scenario_name = upload_result.split(".")[0].split("/")[-1].split("\\")[-1]
    return submission_scenario_name
    
def _make_submission_on_server(submission_scenario_name: str, key: str, 
                               params_dict: dict, details: list[str], extra_metadata: dict, 
                               include_images: bool = True) -> str:
    """Make a submission on the server."""

    try:
        # Process parameters and generate submission data
        processed_params = _process_params_data(params_dict, extra_metadata)
        key_components = _generate_key_components(summary(submission_scenario_name, print_summary=False))
    except Exception as e:
        print(f"Error: Failed to process parameters and generate key components")
        raise RuntimeError(f"Failed to process parameters and generate key components - {str(e)}")

    submission_data = {
        "title": submission_scenario_name,
        "details": details,
        "keyComponents": key_components["sections"],
        "features": processed_params["primaryParameters"],
        "advancedParameters": processed_params["advancedParameters"],
    }

    print("Creating website submission...")
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/submissions",
            json={"type": "scenario", "content": submission_data},
            headers={"Authorization": f"Bearer {key}"},
        )
        response.raise_for_status()
        print("✓ Submission created successfully")
  
        print('Thank you for your submission!')
        print('Head over to deepmimo.net/dashboard?tab=submissions to monitor it.')
        print('The admins have been notified and will get to it ASAP.')
        print('\n >> Please upload the ray tracing source as well by calling:')
        print(f"upload_rt_source('{submission_scenario_name}', dm.zip(<rt_folder>), <key>)")
        print('where <rt_folder> is the path to the ray tracing source folder as in dm.convert(<rt_folder>)')
    
    except Exception as e:
        print(f"Error: Failed to create submission for {submission_scenario_name}")
        print(json.loads(response.text)["error"])
        raise RuntimeError(f"Failed to create submission - {str(e)}")
    
    # Generate and upload images if requested
    if include_images:
        print("Generating scenario visualizations...")
        try:
            img_paths = plot_summary(submission_scenario_name, save_imgs=True)
            if img_paths:
                uploaded_images_meta = upload_images(submission_scenario_name, img_paths, key)
                print(f"Image upload process completed. {len(uploaded_images_meta)} images attached.")
        except Exception as e:
            print("Warning: Failed during image generation or upload phase")
            print(f"Error: {str(e)}")
        finally:
            # Clean up locally generated temporary image files
            if img_paths:
                print("Cleaning up local image files...")
                cleaned_count = 0
                for img_path in img_paths:
                    if os.path.exists(img_path):
                        os.remove(img_path)
                        cleaned_count += 1
                print(f"Cleaned up {cleaned_count} local image files.")
                
                # Clean up the figure's directory if it's empty
                temp_dir = os.path.dirname(img_paths[0])
                if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                    os.rmdir(temp_dir)
                    print(f"Removed empty directory: {temp_dir}")

    return submission_scenario_name

def upload(scenario_name: str, key: str,
           details: Optional[list[str]] = None, extra_metadata: Optional[dict] = None, 
           skip_zip: bool = False, submission_only: bool = False, include_images: bool = True) -> str:
    """Upload a DeepMIMO scenario to the server.

    Uploads a scenario to the DeepMIMO database by zipping the scenario folder,
    uploading to the database, and creating a submission on the server.

    Args:
        scenario_name (str): Name of the scenario to upload.
        key (str): Authorization key for upload access.
        details (list[str], optional): List of details about the scenario for detail boxes.
        extra_metadata (dict, optional): Additional metadata fields including:
            digitalTwin (bool): Whether scenario is a digital twin
            environment (str): Either 'indoor' or 'outdoor'
            bbCoords (dict): Bounding box coordinates with keys:
            - minLat (float): Minimum latitude
            - minLon (float): Minimum longitude  
            - maxLat (float): Maximum latitude
            - maxLon (float): Maximum longitude
            city (str): City name
        skip_zip (bool, optional): If True, skip zipping scenario folder. Defaults to False.
        include_images (bool, optional): If True, generate and upload visualization images. 
            Defaults to True.
        submission_only (bool, optional): If True, skip database upload and only create server 
            submission. Use when scenario is already uploaded. Defaults to False.

    Returns:
        str: Name of submitted scenario if initial submission succeeds, None otherwise.
            Image upload status does not affect return value.
    """
    scenario_name = scenario_name.lower()
    scen_folder = get_scenario_folder(scenario_name)
    params_path = get_params_path(scenario_name)

    print(f"Processing scenario: {scenario_name}")

    try:
        print("Parsing scenario parameters...")
        params_dict = load_dict_from_json(params_path)
        print("✓ Parameters parsed successfully")
    except Exception as e:
        print("Error: Failed to parse parameters")
        raise RuntimeError(f"Failed to parse parameters - {str(e)}")

    if not submission_only:
        submission_scenario_name = _upload_to_db(scen_folder, key, skip_zip)
    else:
        submission_scenario_name = scenario_name

    _make_submission_on_server(submission_scenario_name, key, params_dict, details, extra_metadata, include_images)

    # Return the scenario name used for submission
    return submission_scenario_name

def upload_rt_source(scenario_name: str, rt_zip_path: str, key: str) -> bool:
    """Upload a Ray Tracing (RT) source file to the database.

    Args:
        scenario_name: The name of the corresponding scenario already uploaded.
                       The RT source will be stored under `<scenario_name>.zip`.
        rt_zip_path: Path to the zipped RT source file to upload.
        key: API authentication key.

    Returns:
        True if the upload was successful, False otherwise.
    """
    print(f"Attempting to upload RT source for scenario: {scenario_name}")
    print(f"Using RT source file: {rt_zip_path}")

    if not os.path.exists(rt_zip_path):
        print(f"Error: RT source file not found at {rt_zip_path}")
        return False

    target_filename = f"{scenario_name}.zip"
    file_size = os.path.getsize(rt_zip_path)

    if file_size > RT_FILE_SIZE_LIMIT:
        print(f"Error: RT source file size limit of {RT_FILE_SIZE_LIMIT / 1024 ** 3} GB exceeded.")
        return False

    try:
        # 1. Get presigned upload URL for the RT database
        print("Requesting RT upload authorization from server...")
        auth_response = requests.get(
            f"{API_BASE_URL}/api/b2/authorize-rt-upload",
            params={"scenario_name": scenario_name}, # Server expects scenario_name
            headers={"Authorization": f"Bearer {key}"},
        )
        auth_response.raise_for_status()
        auth_data = auth_response.json()

        if not auth_data.get("presignedUrl"):
            print("Error: Invalid authorization response from server.")
            return False

        # Server confirms the filename it authorized for the RT bucket
        authorized_filename = auth_data.get("filename")
        if not authorized_filename or authorized_filename != target_filename:
             print("Error: Filename mismatch.")
             print(f"Server authorized RT upload for '{authorized_filename}' but expected '{target_filename}'")
             return False

        print(f"✓ Authorization granted. Uploading to RT database as '{authorized_filename}'...")

        # 2. Calculate file hash (using the local rt_zip_path file)
        sha1 = hashlib.sha1()
        with open(rt_zip_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha1.update(chunk)
        file_hash = sha1.hexdigest()

        # 3. Upload file to the RT database using the presigned URL
        pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc="Uploading RT Source")
        progress_reader = None
        try:
            progress_reader = _ProgressFileReader(rt_zip_path, pbar)

            upload_response = requests.put(
                auth_data["presignedUrl"],
                headers={
                    "Content-Type": auth_data.get("contentType", "application/zip"),
                    "Content-Length": str(file_size),
                    "X-Bz-Content-Sha1": file_hash, # Required by the database
                },
                data=progress_reader
            )
            upload_response.raise_for_status()
        finally:
            if progress_reader:
                progress_reader.close()
            pbar.close()

        print(f"✓ RT source uploaded successfully as {authorized_filename}")

        return True

    except requests.exceptions.HTTPError as e:
        print(f"API call failed: {e.response.status_code}")
        try:
            error_details = e.response.json()
            print(f"Server Error: {error_details.get('error', e.response.text)}")
        except ValueError:
            print(f"Server Response: {e.response.text}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"Network or request error during RT upload: {str(e)}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during RT upload: {str(e)}")
        return False

def _download_url(scenario_name: str, rt_source: bool = False) -> str:
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


def download(scenario_name: str, output_dir: Optional[str] = None, rt_source: bool = False) -> Optional[str]:
    """Download a DeepMIMO scenario from the database.

    Args:
        scenario_name: Name of the scenario
        output_dir: Directory to save file (defaults to current directory)
        rt_source: Whether to download the raytracing source file instead of the scenario

    Returns:
        Path to downloaded file if successful, None otherwise
    """
    scenario_name = scenario_name.lower()
    scenarios_dir = get_scenarios_dir()
    download_dir = output_dir if output_dir else get_scenarios_dir()
    scenario_folder = get_scenario_folder(scenario_name)
    
    # Check if file already exists in scenarios folder
    if os.path.exists(scenario_folder):
        print(f'Scenario "{scenario_name}" already exists in {scenarios_dir}')
        return None

    # Get secure download URL using existing helper
    url = _download_url(scenario_name, rt_source)
    
    file_suffix = "_rt_source" if rt_source else "_downloaded"
    output_path = os.path.join(download_dir, f"{scenario_name}{file_suffix}.zip")

    # Check if file already exists in download folder
    if not os.path.exists(output_path):
        # Create download directory if it doesn't exist
        os.makedirs(download_dir, exist_ok=True)

        download_type = "raytracing source" if rt_source else "scenario"
        print(f"Downloading {download_type} '{scenario_name}'")
        try:
            # Get download token and redirect URL
            resp = requests.get(url, headers=HEADERS)
            resp.raise_for_status()
            token_data = resp.json()
            
            if "error" in token_data:
                print(f"Server error: {token_data.get('error')}")
                return None
            
            # Get and format redirect URL
            redirect_url = token_data.get('redirectUrl')
            if not redirect_url:
                print("Error: Missing redirect URL")
                return None
                
            if not redirect_url.startswith('http'):
                redirect_url = f"{url.split('/api/')[0]}{redirect_url}"
            
            # Download the file
            download_resp = requests.get(redirect_url, stream=True, headers=HEADERS)
            download_resp.raise_for_status()
            total_size = int(download_resp.headers.get("content-length", 0))

            with open(output_path, "wb") as file, \
                 tqdm(total=total_size, unit='B', unit_scale=True, 
                      unit_divisor=1024, desc="Downloading") as pbar:
                for chunk in download_resp.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        pbar.update(len(chunk))
            
            print(f"✓ Downloaded to {output_path}")

        except requests.exceptions.RequestException as e:
            print(f"Download failed: {str(e)}")
            if os.path.exists(output_path):
                os.remove(output_path)  # Clean up partial download
            return None
    else: # Extract the zip if it exists, don't download again
        print(f'Scenario zip file "{output_path}" already exists.')
    
    # Handle file extraction based on type
    if rt_source:
        # For RT source files, just extract and don't move to scenarios folder
        unzipped_folder = unzip(output_path)
        print(f"✓ RT source files extracted to {unzipped_folder}")
        print(f"✓ RT source '{scenario_name}' downloaded!")
    else:
        # For regular scenarios, unzip and move to scenarios folder
        unzipped_folder = unzip(output_path)
        unzipped_folder_without_suffix = unzipped_folder.replace('_downloaded', '')
        os.makedirs(scenarios_dir, exist_ok=True)
        os.rename(unzipped_folder, unzipped_folder_without_suffix)
        shutil.move(unzipped_folder_without_suffix, scenario_folder)
        print(f"✓ Unzipped and moved to {scenarios_dir}")
        print(f"✓ Scenario '{scenario_name}' ready to use!")

    return output_path 

def search(query: Optional[Dict] = None) -> Optional[List[str]]:
    """
    Search for scenarios in the DeepMIMO database.

    Args:
        query: Dictionary containing search parameters from the following list:
        - bands: List[str] - Array of frequency bands ['sub6', 'mmW', 'subTHz']
        - raytracerName: str - Raytracer name or 'all'
        - environment: str - 'indoor', 'outdoor', or 'all'
        - numTx: Dict - Numeric range filter {'min': number, 'max': number}
        - numRx: Dict - Numeric range filter {'min': number, 'max': number}
        - pathDepth: Dict - Numeric range filter {'min': number, 'max': number}
        - maxReflections: Dict - Numeric range filter {'min': number, 'max': number}
        - numRays: Dict - Numeric range filter {'min': number, 'max': number}
        - multiRxAnt: bool - Boolean filter or 'all' to ignore
        - multiTxAnt: bool - Boolean filter or 'all' to ignore
        - dualPolarization: bool - Boolean filter or 'all' to ignore
        - BS2BS: bool - Boolean filter or 'all' to ignore
        - dynamic: bool - Boolean filter or 'all' to ignore
        - diffraction: bool - Boolean filter or 'all' to ignore
        - scattering: bool - Boolean filter or 'all' to ignore
        - transmission: bool - Boolean filter or 'all' to ignore
        - digitalTwin: bool - Boolean filter or 'all' to ignore
        - hasRtSource: bool - Boolean filter or 'all' to ignore
        - city: str - City name text filter
        - bbCoords: Dict - Bounding box coordinates 
            {'minLat': float, 'minLon': float, 'maxLat': float, 'maxLon': float}
    
    Returns:
        Dict containing count and list of matching scenario names if successful, None otherwise
    """
    if query is None:
        query = {}
    try:
        response = requests.post(f"{API_BASE_URL}/api/search/scenarios", json=query)
        response.raise_for_status()
        data = response.json()
        return data['scenarios']
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {str(e)}")
        if hasattr(e.response, 'text'):
            try:
                error_data = e.response.json()
                print(f"Server error details: {error_data.get('error', e.response.text)}")
            except:
                print(f"Server response: {e.response.text}")
        return None
    except requests.exceptions.ConnectionError:
        print("Error: Connection failed. Please check your internet connection and try again.")
        return None
    except requests.exceptions.Timeout:
        print("Error: Request timed out. Please try again later.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request Error: {str(e)}")
        return None
    except ValueError as e:
        print(f"Error parsing response: {str(e)}")
        return None
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return None

