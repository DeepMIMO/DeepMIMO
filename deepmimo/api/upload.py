"""DeepMIMO API Upload Module.

This module provides functionality for uploading scenarios, RT sources,
and images to the DeepMIMO database.

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
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from deepmimo import consts as c
from deepmimo.datasets.summary import plot_summary, summary
from deepmimo.utils import (
    get_params_path,
    get_scenario_folder,
    load_dict_from_json,
    zip,  # noqa: A004
)

from ._common import (
    API_BASE_URL,
    FILE_SIZE_LIMIT,
    HTTP_OK,
    IMAGE_SIZE_LIMIT,
    MAX_IMAGES_PER_UPLOAD,
    MMW_UPPER_GHZ,
    REQUEST_TIMEOUT,
    RT_FILE_SIZE_LIMIT,
    SUB6_UPPER_GHZ,
    SubmissionConfig,
    _ProgressFileReader,
)


def _compute_sha1(file_path: Path) -> str:
    """Compute SHA1 hash of a file."""
    sha1 = hashlib.sha1()  # noqa: S324 (SHA1 is required by the server)
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha1.update(chunk)
    return sha1.hexdigest()


def _get_upload_authorization(
    file_path: Path, key: str
) -> tuple[dict[str, Any], str] | tuple[None, None]:
    """Get upload authorization from server.

    Args:
        file_path: Path to file to upload
        key: API authentication key

    Returns:
        Tuple of (auth_data, authorized_filename) if successful, (None, None) otherwise

    """
    filename = file_path.name
    file_size = file_path.stat().st_size

    if file_size > FILE_SIZE_LIMIT:
        print(f"Error: File size limit of {FILE_SIZE_LIMIT / 1024**3} GB exceeded.")
        return None, None

    # Get presigned upload URL
    auth_response = requests.get(
        f"{API_BASE_URL}/api/b2/authorize-upload",
        params={"filename": filename},
        headers={"Authorization": f"Bearer {key}"},
        timeout=REQUEST_TIMEOUT,
    )
    auth_response.raise_for_status()
    auth_data = auth_response.json()

    if not auth_data.get("presignedUrl"):
        print("Error: Invalid authorization response")
        return None, None

    # Verify filename match
    authorized_filename = auth_data.get("filename")
    if authorized_filename and authorized_filename != filename:
        msg = (
            "Error: Filename mismatch. "
            f"Server authorized '{authorized_filename}' but trying to upload '{filename}'"
        )
        print(msg)
        return None, None

    return auth_data, authorized_filename or filename


def _perform_file_upload(
    file: str, file_size: int, auth_data: dict[str, Any], file_hash: str
) -> bool:
    """Perform the actual file upload with progress bar.

    Args:
        file: Path to file
        file_size: Size of file in bytes
        auth_data: Authorization data from server
        file_hash: SHA1 hash of file

    Returns:
        True if upload successful, False otherwise

    """
    authorized_filename = auth_data.get("filename", Path(file).name)
    print(f"Uploading {authorized_filename} to DB...")
    pbar = tqdm(total=file_size, unit="B", unit_scale=True, desc="Uploading")

    try:
        progress_reader = _ProgressFileReader(file, pbar)
        upload_response = requests.put(
            auth_data["presignedUrl"],
            headers={
                "Content-Type": auth_data.get("contentType", "application/zip"),
                "Content-Length": str(file_size),
                "X-Bz-Content-sha1": file_hash,
            },
            data=progress_reader,
            timeout=REQUEST_TIMEOUT,
        )
        upload_response.raise_for_status()
        return upload_response.status_code == HTTP_OK
    finally:
        progress_reader.close()
        pbar.close()


def _dm_upload_api_call(file: str, key: str) -> str | None:
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
    result = None
    try:
        file_path = Path(file)

        # Get authorization
        auth_data, authorized_filename = _get_upload_authorization(file_path, key)
        if not auth_data:
            return result

        # Calculate file hash and perform upload
        file_hash = _compute_sha1(file_path)
        file_size = file_path.stat().st_size

        if _perform_file_upload(file, file_size, auth_data, file_hash):
            result = authorized_filename

    except requests.exceptions.HTTPError as e:
        print(f"API call failed: {e!s}")
        if e.response is not None:
            try:
                error_data = e.response.json()
                server_message = error_data.get("error", "No specific error message found in JSON.")
                print(f"Server Error ({e.response.status_code}): {server_message}")
            except ValueError:
                print(f"Server Response ({e.response.status_code}): {e.response.text}")
        else:
            print("API call failed without receiving a response from the server.")
    except requests.exceptions.RequestException as e:
        print(f"API call failed: {e!s}")
        if hasattr(e, "response") and e.response:
            print(f"Server response: {json.loads(e.response.text)['error']}")
    except Exception as e:  # noqa: BLE001
        print(f"Upload failed due to an unexpected error: {e!s}")

    return result


def dm_upload_api_call(file: str, key: str) -> str | None:
    """Public wrapper around `_dm_upload_api_call`."""
    return _dm_upload_api_call(file, key)


def _process_params_data(params_dict: dict, extra_metadata: dict | None = None) -> dict:
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
    num_tx = (
        sum(
            set_info.get("num_active_points", 0)
            for set_info in txrx_sets.values()
            if set_info.get("is_tx")
        )
        or 1
    )

    num_rx = (
        sum(
            set_info.get("num_active_points", 0)
            for set_info in txrx_sets.values()
            if set_info.get("is_rx")
        )
        or 1
    )

    raytracer_map = {
        c.RAYTRACER_NAME_WIRELESS_INSITE: "Insite",
        c.RAYTRACER_NAME_SIONNA: "Sionna",
        c.RAYTRACER_NAME_AODT: "AODT",
    }

    # Create base parameter dictionaries
    primary_params = {
        "bands": {
            "sub6": 0 <= frequency < SUB6_UPPER_GHZ,
            "mmW": SUB6_UPPER_GHZ <= frequency <= MMW_UPPER_GHZ,
            "subTHz": frequency > MMW_UPPER_GHZ,
        },
        "numRx": num_rx,
        "maxReflections": rt_params.get("max_reflections", 1),
        "raytracerName": raytracer_map.get(rt_params.get("raytracer_name"), "Insite"),
        "environment": "outdoor",
    }

    advanced_params = {
        "dmVersion": params_dict.get("version", "4.0.0a"),
        "numTx": num_tx,
        "multiRxAnt": any(
            set_info.get("num_ant", 0) > 1
            for set_info in txrx_sets.values()
            if set_info.get("is_rx")
        ),
        "multiTxAnt": any(
            set_info.get("num_ant", 0) > 1
            for set_info in txrx_sets.values()
            if set_info.get("is_tx")
        ),
        "dualPolarization": any(set_info.get("dual_pol", False) for set_info in txrx_sets.values()),
        "BS2BS": any(
            set_info.get("is_tx") and set_info.get("is_rx") for set_info in txrx_sets.values()
        )
        or None,
        "pathDepth": rt_params.get("max_path_depth", None),
        "diffraction": bool(rt_params.get("max_diffractions", 0)),
        "scattering": bool(rt_params.get("max_scattering", 0)),
        "transmission": bool(rt_params.get("max_transmissions", 0)),
        "numRays": rt_params.get("num_rays", 1000000),
        "city": None,
        "digitalTwin": False,
        "dynamic": scene_params.get("num_scenes", 1) > 1,
        "bbCoords": None,
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
        "advancedParameters": advanced_params,
    }


def process_params_data(params_dict: dict, extra_metadata: dict | None = None) -> dict:
    """Public wrapper around `_process_params_data`."""
    return _process_params_data(params_dict, extra_metadata)


def _generate_key_components(summary_str: str) -> dict:
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

    for line_str in summary_str.split("\n"):
        line = line_str.strip()
        if not line or line.startswith("="):  # Skip empty lines and separator lines
            continue

        if line.startswith("[") and line.endswith("]"):
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


def generate_key_components(summary_str: str) -> dict:
    """Public wrapper around `_generate_key_components`."""
    return _generate_key_components(summary_str)


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
        if line and not line.startswith("-"):  # New subsection header
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
                {"".join(html_parts)}
            </div>
        """,
    }


def format_section(name: str, lines: list) -> dict:
    """Public wrapper around `_format_section` for external callers."""
    return _format_section(name, lines)


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

    if len(img_paths) > MAX_IMAGES_PER_UPLOAD:
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
        "scene.png": {
            "heading": "Scenario Layout",
            "description": "Physical layout of the scenario",
        },
    }

    print(f"Attempting to upload {len(img_paths)} images for scenario '{scenario_name}'...")

    # Initialize tqdm manually before the loop
    pbar = tqdm(total=len(img_paths), desc="Uploading images", unit="image")

    # Iterate directly over img_paths
    for i, img_path in enumerate(img_paths):
        img_path_obj = Path(img_path)
        filename = img_path_obj.name
        filesize = img_path_obj.stat().st_size

        if filesize > IMAGE_SIZE_LIMIT:
            print(f"Warning: Image {filename} is too large to upload. Skipping...")
            continue

        try:
            # Get default metadata or create generic ones
            default_info = image_types.get(
                filename,
                {
                    "heading": f"Image {i + 1}",
                    "description": f"Visualization {i + 1} for {scenario_name}",
                },
            )

            # Prepare form data
            with img_path_obj.open("rb") as img_file:
                files = {"image": (filename, img_file, "image/png")}  # Key is 'image' now
                data = {
                    "heading": default_info["heading"],
                    "description": default_info["description"],
                }

                # Make the POST request to the new endpoint for each image
                response = requests.post(
                    upload_url_template,
                    headers={"Authorization": f"Bearer {key}"},
                    files=files,
                    data=data,  # Send heading/description in form data
                    timeout=REQUEST_TIMEOUT,
                )

            response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)

            # If successful, server returns the metadata of the uploaded image
            result = response.json()
            uploaded_image_objects.append(result)
            print(f"✓ Successfully uploaded and attached: {filename}")

            # Update the progress bar ONLY after successful upload
            pbar.update(1)
        except requests.exceptions.RequestException as err:
            if err.response is not None:
                server_message = json.loads(err.response.text)["error"]
                print(
                    f"✗ Failed to upload {filename}: {server_message} "
                    f"(Server Response Code: {err.response.status_code})",
                )
            else:
                # Handle cases where the error didn't have a response object
                print(f"✗ Failed to upload {filename}: {err}")

    # Close the progress bar after the loop finishes or breaks
    pbar.close()

    if uploaded_image_objects:
        msg = (
            "✓ Finished image upload process. "
            f"Successfully attached {len(uploaded_image_objects)} images."
        )
        print(msg)
    else:
        print("No images were successfully attached.")

    return uploaded_image_objects


def _upload_to_db(scen_folder: str, key: str, *, skip_zip: bool = False) -> str:
    """Upload a zip file to the database."""
    # Zip scenario
    zip_path = scen_folder + ".zip" if skip_zip else zip(scen_folder)

    try:
        print("Uploading to the database...")
        upload_result = _dm_upload_api_call(zip_path, key)
    except Exception as e:  # noqa: BLE001
        print(f"Error: Failed to upload to the database - {e!s}")

    if not upload_result:
        print("Error: Failed to upload to the database")
        msg = "Failed to upload to the database"
        raise RuntimeError(msg)
    print("✓ Upload successful")

    return upload_result.split(".")[0].split("/")[-1].split("\\")[-1]


def _handle_submission_images(
    submission_scenario_name: str,
    key: str,
) -> None:
    """Generate and upload images for a submission."""
    print("Generating scenario visualizations...")
    img_paths = []
    try:
        img_paths = plot_summary(submission_scenario_name, save_imgs=True)
        if img_paths:
            uploaded_images_meta = upload_images(submission_scenario_name, img_paths, key)
            print(
                f"Image upload process completed. {len(uploaded_images_meta)} images attached.",
            )
    except Exception as e:  # noqa: BLE001
        print("Warning: Failed during image generation or upload phase")
        print(f"Error: {e!s}")
    finally:
        # Clean up locally generated temporary image files
        if img_paths:
            print("Cleaning up local image files...")
            cleaned_count = 0
            for img_path in img_paths:
                if Path(img_path).exists():
                    Path(img_path).unlink()
                    cleaned_count += 1
            print(f"Cleaned up {cleaned_count} local image files.")

            # Clean up the figure's directory if it's empty
            temp_dir = Path(img_paths[0]).parent
            if temp_dir.exists() and not list(temp_dir.iterdir()):
                temp_dir.rmdir()
                print(f"Removed empty directory: {temp_dir}")


def _make_submission_on_server(
    submission_scenario_name: str,
    key: str,
    config: SubmissionConfig,
) -> str:
    """Make a submission on the server."""
    params_dict = config["params_dict"]
    details = config["details"]
    extra_metadata = config["extra_metadata"]
    include_images = config["include_images"]

    try:
        # Process parameters and generate submission data
        processed_params = _process_params_data(params_dict, extra_metadata)
        key_components = _generate_key_components(
            summary(submission_scenario_name, print_summary=False),
        )
    except Exception as e:
        print("Error: Failed to process parameters and generate key components")
        msg = f"Failed to process parameters and generate key components - {e!s}"
        raise RuntimeError(msg) from e

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
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        print("✓ Submission created successfully")

        print("Thank you for your submission!")
        print("Head over to deepmimo.net/dashboard?tab=submissions to monitor it.")
        print("The admins have been notified and will get to it ASAP.")
        print("\n >> Please upload the ray tracing source as well by calling:")
        print(f"upload_rt_source('{submission_scenario_name}', dm.zip(<rt_folder>), <key>)")
        print(
            "where <rt_folder> is the path to the ray tracing source folder "
            "as in dm.convert(<rt_folder>)",
        )

    except requests.exceptions.RequestException as err:
        print(f"Error: Failed to create submission for {submission_scenario_name}")
        print(json.loads(response.text)["error"])
        msg = f"Failed to create submission - {err!s}"
        raise RuntimeError(msg) from err
    except Exception as err:
        print(f"Unexpected error creating submission for {submission_scenario_name}: {err!s}")
        msg = f"Failed to create submission - {err!s}"
        raise RuntimeError(msg) from err

    # Generate and upload images if requested
    if include_images:
        _handle_submission_images(submission_scenario_name, key)

    return submission_scenario_name


def make_submission_on_server(
    submission_scenario_name: str,
    key: str,
    config: SubmissionConfig,
) -> str:
    """Public wrapper for `_make_submission_on_server`."""
    return _make_submission_on_server(submission_scenario_name, key, config)


def upload(  # noqa: PLR0913 - comprehensive upload requires all configuration parameters
    scenario_name: str,
    key: str,
    details: list[str] | None = None,
    extra_metadata: dict | None = None,
    *,
    skip_zip: bool = False,
    submission_only: bool = False,
    include_images: bool = True,
) -> str:
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
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print("Error: Failed to parse parameters")
        msg = f"Failed to parse parameters - {e!s}"
        raise RuntimeError(msg) from e
    except Exception as e:
        print("Error: Failed to parse parameters")
        msg = f"Failed to parse parameters - {e!s}"
        raise RuntimeError(msg) from e

    if not submission_only:
        submission_scenario_name = _upload_to_db(scen_folder, key, skip_zip=skip_zip)
    else:
        submission_scenario_name = scenario_name

    config: SubmissionConfig = {
        "params_dict": params_dict,
        "details": details or [],
        "extra_metadata": extra_metadata or {},
        "include_images": include_images,
    }

    make_submission_on_server(
        submission_scenario_name,
        key,
        config,
    )

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

    rt_zip_path_obj = Path(rt_zip_path)
    if not rt_zip_path_obj.exists():
        print(f"Error: RT source file not found at {rt_zip_path}")
        return False

    target_filename = f"{scenario_name}.zip"
    file_size = rt_zip_path_obj.stat().st_size

    if file_size > RT_FILE_SIZE_LIMIT:
        print(f"Error: RT source file size limit of {RT_FILE_SIZE_LIMIT / 1024**3} GB exceeded.")
        return False

    result = False
    try:
        # 1. Get presigned upload URL for the RT database
        print("Requesting RT upload authorization from server...")
        auth_response = requests.get(
            f"{API_BASE_URL}/api/b2/authorize-rt-upload",
            params={"scenario_name": scenario_name},  # Server expects scenario_name
            headers={"Authorization": f"Bearer {key}"},
            timeout=REQUEST_TIMEOUT,
        )
        auth_response.raise_for_status()
        auth_data = auth_response.json()

        if not auth_data.get("presignedUrl"):
            print("Error: Invalid authorization response from server.")
            return result

        # Server confirms the filename it authorized for the RT bucket
        authorized_filename = auth_data.get("filename")
        if not authorized_filename or authorized_filename != target_filename:
            print("Error: Filename mismatch.")
            print(
                "Server authorized RT upload for "
                f"'{authorized_filename}' but expected '{target_filename}'",
            )
            return result

        print(f"✓ Authorization granted. Uploading to RT database as '{authorized_filename}'...")

        # 2. Calculate file hash (using the local rt_zip_path file)
        file_hash = _compute_sha1(rt_zip_path_obj)

        # 3. Upload file to the RT database using the presigned URL
        pbar = tqdm(total=file_size, unit="B", unit_scale=True, desc="Uploading RT Source")
        progress_reader = None
        try:
            progress_reader = _ProgressFileReader(rt_zip_path, pbar)

            upload_response = requests.put(
                auth_data["presignedUrl"],
                headers={
                    "Content-Type": auth_data.get("contentType", "application/zip"),
                    "Content-Length": str(file_size),
                    "X-Bz-Content-sha1": file_hash,  # Required by the database
                },
                data=progress_reader,
                timeout=REQUEST_TIMEOUT,
            )
            upload_response.raise_for_status()
            print(f"✓ RT source uploaded successfully as {authorized_filename}")
            result = True
        finally:
            if progress_reader:
                progress_reader.close()
            pbar.close()

    except requests.exceptions.HTTPError as e:
        print(f"API call failed: {e.response.status_code}")
        try:
            error_details = e.response.json()
            print(f"Server Error: {error_details.get('error', e.response.text)}")
        except ValueError:
            print(f"Server Response: {e.response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Network or request error during RT upload: {e!s}")
    except Exception as e:  # noqa: BLE001
        print(f"An unexpected error occurred during RT upload: {e!s}")

    return result


