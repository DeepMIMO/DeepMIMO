"""This file contains utility functions for Blender.
Many of them will only work inside Blender.
"""

import logging
import math
import os
import subprocess
import sys
from typing import Any

# Blender imports
import bpy  # type: ignore
import mathutils  # type: ignore (comes with blender)
import requests

ADDONS = {
    "blosm": "blosm_2.7.11.zip",
    "mitsuba-blender": "mitsuba-blender.zip",
}

ADDON_URLS = {
    "blosm": "https://www.dropbox.com/scl/fi/cka3yriyrjppnfy2ztjq9/blosm_2.7.11.zip?rlkey=9ak7vnf4h13beqd4hpwt9e3ws&st=znk7icsq&dl=1",
    # blosm link is self-hosted on dropbox because it is not properly hosted anywhere else.
    # The original link is: https://github.com/vvoovv/blosm (which forwards to gumroad)
    "mitsuba-blender": "https://www.dropbox.com/scl/fi/lslog12ehjl7n6f8vjaaj/mitsuba-blender.zip?rlkey=vve9h217m42ksros47x40sl45&st=oltvhszv&dl=1",
    # mitsuba-blender link is self-hosted on dropbox because it is a slightly changed
    # version that fixes a bug to work solely with bpy in linux.
}

# Material names for scene objects
FLOOR_MATERIAL = "itu_wet_ground"
PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))

# Blender version
BLENDER_MAJOR_VERSION = bpy.app.version[0]

###############################################################################
# LOGGER SETUP
###############################################################################

LOGGER: Any | None = None


def log_local_setup(log_file_path: str) -> logging.Logger:
    """Set up local logging configuration for both console and file output.

    Args:
        log_file_path (str): Full path to the log file

    """
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),  # Console handler
            logging.FileHandler(log_file_path, mode="w"),  # File handler
        ],
    )
    return logging.getLogger(os.path.basename(log_file_path))


def set_LOGGER(logger: Any) -> None:
    """Set the logger for the BlenderUtils class."""
    global LOGGER
    LOGGER = logger


###############################################################################
# ADD-ON INSTALLATION UTILITIES
###############################################################################


def download_addon(addon_name: str) -> str:
    """Download a file from a URL and save it to a local path."""
    output_path = os.path.join(PROJ_ROOT, "blender_addons", ADDONS[addon_name])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    url = ADDON_URLS[addon_name]
    LOGGER.info(f"📥 Downloading file from {url} to {output_path}")
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(response.content)
    except Exception as e:
        error_msg = f"❌ Failed to download file from {url}: {e!s}"
        LOGGER.exception(error_msg)
        raise Exception(error_msg)

    return output_path


def install_python_package(pckg_name: str) -> None:
    """Install a Python package using Blender's Python executable."""
    LOGGER.info(f"📦 Installing Python package: {pckg_name}")
    python_exe = sys.executable
    LOGGER.debug(f"Using Python executable: {python_exe}")

    try:
        subprocess.call([python_exe, "-m", "ensurepip"])
        subprocess.call([python_exe, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.call([python_exe, "-m", "pip", "install", pckg_name])
        LOGGER.info(f"✅ Successfully installed {pckg_name}")
    except Exception as e:
        error_msg = f"❌ Failed to install {pckg_name}: {e!s}"
        LOGGER.exception(error_msg)
        raise Exception(error_msg)


def install_blender_addon(addon_name: str) -> None:
    """Install and enable a Blender add-on from a zip file if not already installed."""
    LOGGER.info(f"🔧 Processing Blender add-on: {addon_name}")
    zip_name = ADDONS.get(addon_name)
    if not zip_name:
        LOGGER.error(f"❌ No zip file defined for add-on '{addon_name}'")
        return

    if addon_name in bpy.context.preferences.addons:
        LOGGER.info(f"📌 Add-on '{addon_name}' is already installed")
        if not bpy.context.preferences.addons[addon_name].module:
            LOGGER.info(f"  Enabling add-on '{addon_name}'")
            bpy.ops.preferences.addon_enable(module=addon_name)
            bpy.ops.wm.save_userpref()
    else:
        addon_zip_path = os.path.join(PROJ_ROOT, "blender_addons", zip_name)
        if not os.path.exists(addon_zip_path):
            LOGGER.warning(f"⚠ Add-on zip file not found: {addon_zip_path}")
            LOGGER.info(f"Attempting to download {addon_zip_path}")
            addon_zip_path = download_addon(addon_name)

        try:
            bpy.ops.preferences.addon_install(filepath=addon_zip_path)
            bpy.ops.preferences.addon_enable(module=addon_name)
            bpy.ops.wm.save_userpref()
            LOGGER.info(f"✅ Add-on '{addon_name}' installed and enabled")
        except Exception as e:
            LOGGER.exception(f"❌ Failed to install/enable add-on '{addon_name}': {e!s}")
            raise

    # Special handling for Mitsuba
    if addon_name == "mitsuba-blender":
        try:
            import mitsuba

            LOGGER.info("✅ Mitsuba import successful")
        except ImportError:
            LOGGER.info("📦 Mitsuba not found, installing mitsuba package")
            install_python_package("mitsuba==3.5.0")  # sionna 0.19
            # install_python_package('mitsuba==3.6.2') # sionna 1.0
            LOGGER.warning("🔄 Packages installed! Restarting Blender to update imports")
            bpy.ops.wm.quit_blender()


###############################################################################
# BLOSM (OpenStreetMap) UTILITIES
###############################################################################


def configure_osm_import(
    output_folder: str,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
) -> None:
    """Configure blosm add-on for OSM data import."""
    LOGGER.info(
        f"🗺️ Configuring OSM import for region: [{min_lat}, {min_lon}] to [{max_lat}, {max_lon}]",
    )
    try:
        prefs = bpy.context.preferences.addons["blosm"].preferences
        prefs.dataDir = output_folder

        scene = bpy.context.scene.blosm
        scene.mode = "3Dsimple"
        scene.minLat, scene.maxLat = min_lat, max_lat
        scene.minLon, scene.maxLon = min_lon, max_lon
        scene.buildings, scene.highways = True, True
        scene.water, scene.forests, scene.vegetation, scene.railways = False, False, False, False
        scene.singleObject, scene.ignoreGeoreferencing = True, True
        LOGGER.info("✅ OSM import configuration complete")
    except Exception as e:
        error_msg = f"❌ Failed to configure OSM import: {e!s}"
        LOGGER.exception(error_msg)
        raise Exception(error_msg)


def save_osm_origin(scene_folder: str) -> None:
    """Save OSM origin coordinates to a text file."""
    origin_lat = bpy.data.scenes["Scene"]["lat"]
    origin_lon = bpy.data.scenes["Scene"]["lon"]
    LOGGER.info(f"📍 Saving OSM origin coordinates: [{origin_lat}, {origin_lon}]")
    try:
        output_path = os.path.join(scene_folder, "osm_gps_origin.txt")
        with open(output_path, "w") as f:
            f.write(f"{origin_lat}\n{origin_lon}\n")
        LOGGER.info("✅ OSM origin saved")
    except Exception as e:
        error_msg = f"❌ Failed to save OSM origin: {e!s}"
        LOGGER.exception(error_msg)
        raise Exception(error_msg)


###############################################################################
# CORE BLENDER UTILITIES
###############################################################################


def clear_blender() -> None:
    """Remove all datablocks from Blender to start with a clean slate."""
    block_lists: list[Any] = [
        bpy.data.collections,
        bpy.data.objects,
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.textures,
        bpy.data.curves,
        bpy.data.cameras,
    ]

    # First: clear all non-critical blocks
    for block_list in block_lists:
        for block in list(block_list):
            block_list.remove(block, do_unlink=True)

    # Special handling for images (some blender likes to manager itself)
    for img in list(bpy.data.images):
        if img.name not in {"Render Result", "Viewer Node"}:
            bpy.data.images.remove(img, do_unlink=True)


def get_xy_bounds_from_latlon(
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
    pad: float = 0,
) -> tuple[float, float, float, float]:
    """Convert lat/lon bounds to XY bounds centered at 0,0.

    Args:
        min_lat: Minimum latitude
        min_lon: Minimum longitude
        max_lat: Maximum latitude
        max_lon: Maximum longitude

    Returns:
        tuple[float, float, float, float]: (min_x, max_x, min_y, max_y) in meters

    """
    LOGGER.info(f"🌐 Converting lat/lon bounds: [{min_lat}, {min_lon}] to [{max_lat}, {max_lon}]")

    # Get center point
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    LOGGER.debug(f"📍 Center point: [{center_lat}, {center_lon}]")

    # Constants for conversion (meters per degree at equator)
    METER_PER_DEGREE_LAT = 111320  # Approximately constant
    meter_per_degree_lon = 111320 * math.cos(math.radians(center_lat))  # Varies with latitude

    # Convert lat/lon differences to meters
    min_y = (min_lat - center_lat) * METER_PER_DEGREE_LAT - pad
    max_y = (max_lat - center_lat) * METER_PER_DEGREE_LAT + pad
    min_x = (min_lon - center_lon) * meter_per_degree_lon - pad
    max_x = (max_lon - center_lon) * meter_per_degree_lon + pad

    LOGGER.info(
        f"📐 Converted bounds (meters): x=[{min_x:.2f}, {max_x:.2f}], y=[{min_y:.2f}, {max_y:.2f}]",
    )
    if pad > 0:
        LOGGER.debug(f"\t (with padding of {pad} meters to all sides)")

    return min_x, max_x, min_y, max_y


def compute_distance(coord1: tuple[float, float], coord2: tuple[float, float]) -> float:
    """Compute Haversine distance between two coordinates in meters."""
    R = 6371.0  # Earth radius in kilometers
    lat1, lon1 = map(math.radians, coord1)
    lat2, lon2 = map(math.radians, coord2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c * 1000  # Convert to meters


def setup_world_lighting() -> None:
    """Configure world lighting with a basic emitter."""
    LOGGER.info("💡 Setting up world lighting")
    try:
        world = bpy.context.scene.world
        world.use_nodes = True
        nodes = world.node_tree.nodes
        links = world.node_tree.links
        nodes.clear()

        background_node = nodes.new("ShaderNodeBackground")
        output_node = nodes.new("ShaderNodeOutputWorld")
        background_node.inputs["Color"].default_value = (0.517334, 0.517334, 0.517334, 1.0)
        background_node.inputs["Strength"].default_value = 1.0
        links.new(background_node.outputs["Background"], output_node.inputs["Surface"])
        LOGGER.info("✅ World lighting configured")
    except Exception as e:
        error_msg = f"❌ Failed to setup world lighting: {e!s}"
        LOGGER.exception(error_msg)
        raise Exception(error_msg)


def create_camera_and_render(
    output_path: str,
    location: tuple[float, float, float] = (0, 0, 1000),
    rotation: tuple[float, float, float] = (0, 0, 0),
) -> None:
    """Add a camera, render the scene, and delete the camera."""
    LOGGER.info(f"📸 Setting up camera for render at {output_path}")
    scene = bpy.context.scene
    output_folder = os.path.dirname(output_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
        LOGGER.debug(f"📸 Created output folder = {output_folder}")

    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.camera_add(location=location, rotation=rotation)
    camera = bpy.context.active_object
    scene.camera = camera
    LOGGER.debug(f"📸 Camera = {camera}")
    scene.render.filepath = output_path
    LOGGER.debug(f"📸 Path = {scene.render.filepath}")

    try:
        bpy.ops.render.render(write_still=True)
        LOGGER.debug("📸 Camera Rendered -> deleting cam!")
        bpy.data.objects.remove(camera, do_unlink=True)
        LOGGER.debug("📸 Camera deleted!")
    except Exception as e:
        error_msg = f"❌ Failed to render scene: {e!s}"
        LOGGER.exception(error_msg)
        raise Exception(error_msg)


###############################################################################
# SCENE PROCESSING UTILITIES
###############################################################################

REJECTED_ROAD_KEYWORDS = ["profile_", "paths_steps"]

TIERS = {
    1: [
        "map.osm_roads_primary",
        "map.osm_roads_residential",
        "map.osm_roads_tertiary",
        "map.osm_roads_secondary",
        "map.osm_roads_unclassified",
        "map.osm_roads_service",
    ],
    2: ["map.osm_paths_footway"],
}

# Reject all roads because of sionna 1.1 material bug
REJECTED_ROAD_KEYWORDS += TIERS[1] + TIERS[2]


def create_ground_plane(
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
) -> bpy.types.Object:
    """Create and size a ground plane with FLOOR_MATERIAL."""
    LOGGER.info("🌍 Creating ground plane")
    try:
        bpy.ops.mesh.primitive_plane_add(size=1)
        x_size = compute_distance([min_lat, min_lon], [min_lat, max_lon]) * 1.2
        y_size = compute_distance([min_lat, min_lon], [max_lat, min_lon]) * 1.2

        plane = bpy.data.objects.get("Plane")
        if plane is None:
            msg = "Failed to create ground plane"
            raise ValueError(msg)
        plane.scale = (x_size, y_size, 1)
        plane.name = "terrain"

        floor_material = bpy.data.materials.new(name=FLOOR_MATERIAL)
        plane.data.materials.append(floor_material)
    except Exception as e:
        error_msg = f"❌ Failed to create ground plane: {e!s}"
        LOGGER.exception(error_msg)
        raise Exception(error_msg)

    return plane


def add_materials_to_objs(
    name_pattern: str,
    material: bpy.types.Material,
) -> bpy.types.Object | None:
    """Join objects matching a name pattern and apply a material."""
    LOGGER.info(f"🔄 Processing objects matching pattern: {name_pattern}")
    bpy.ops.object.select_all(action="DESELECT")

    # Find mesh objects
    mesh_objs = [o for o in bpy.data.objects if name_pattern in o.name.lower() and o.type == "MESH"]

    if not mesh_objs:
        LOGGER.warning(f"⚠️ No objects found matching pattern: {name_pattern}")
        return None

    try:
        for obj in mesh_objs:
            obj.data.materials.clear()
            obj.data.materials.append(material)
        return obj
    except Exception as e:
        error_msg = f"❌ Failed to process objects with pattern '{name_pattern}': {e!s}"
        LOGGER.exception(error_msg)
        raise Exception(error_msg)


def trim_faces_outside_bounds(
    obj: bpy.types.Object,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
) -> None:
    """Trim faces of an object at the boundary lines and remove parts outside the bounds using boolean intersection."""
    LOGGER.info(f"✂️ Trimming faces at bounds for object: {obj.name}")
    try:
        # First check if object is completely outside bounds
        bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
        obj_min_x = min(corner.x for corner in bbox_corners)
        obj_max_x = max(corner.x for corner in bbox_corners)
        obj_min_y = min(corner.y for corner in bbox_corners)
        obj_max_y = max(corner.y for corner in bbox_corners)

        LOGGER.debug(
            f"Object bounds: x=[{obj_min_x:.2f}, {obj_max_x:.2f}], y=[{obj_min_y:.2f}, {obj_max_y:.2f}]",
        )
        LOGGER.debug(f"Target bounds: x=[{min_x:.2f}, {max_x:.2f}], y=[{min_y:.2f}, {max_y:.2f}]")

        # Expand the bounds by a factor to keep more of the roads
        expansion_factor = 2.0  # Double the bounds to better match road sizes
        expanded_min_x = min_x * expansion_factor
        expanded_max_x = max_x * expansion_factor
        expanded_min_y = min_y * expansion_factor
        expanded_max_y = max_y * expansion_factor

        LOGGER.debug(
            f"Expanded bounds: x=[{expanded_min_x:.2f}, {expanded_max_x:.2f}], y=[{expanded_min_y:.2f}, {expanded_max_y:.2f}]",
        )

        # If object is completely outside expanded bounds, delete it
        if (
            obj_max_x < expanded_min_x
            or obj_min_x > expanded_max_x
            or obj_max_y < expanded_min_y
            or obj_min_y > expanded_max_y
        ):
            LOGGER.warning(f"Object {obj.name} is completely outside expanded bounds - skipping")
            return

        # If object is completely inside original bounds, keep it
        if obj_min_x >= min_x and obj_max_x <= max_x and obj_min_y >= min_y and obj_max_y <= max_y:
            LOGGER.info(f"Object {obj.name} is completely inside bounds - keeping as is")
            return

        LOGGER.info(f"Initial face count for {obj.name}: {len(obj.data.polygons)}")

        # Create a cube that will be our bounding box
        padding = 0.1  # Small padding to avoid precision issues
        bpy.ops.mesh.primitive_cube_add(size=1)
        bound_box = bpy.context.active_object

        # Scale and position the bounding box using expanded bounds
        width = (expanded_max_x - expanded_min_x) + 2 * padding
        height = (expanded_max_y - expanded_min_y) + 2 * padding
        depth = 1000  # Make it very tall to ensure it intersects the full height

        bound_box.scale = (width / 2, height / 2, depth / 2)
        bound_box.location = (
            (expanded_max_x + expanded_min_x) / 2,
            (expanded_max_y + expanded_min_y) / 2,
            0,
        )

        # Add boolean modifier to the original object
        bool_mod = obj.modifiers.new(name="Boolean", type="BOOLEAN")
        bool_mod.object = bound_box
        bool_mod.operation = "INTERSECT"

        # Apply the boolean modifier
        LOGGER.debug("Applying boolean intersection")
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier=bool_mod.name)

        # Delete the bounding box
        bpy.data.objects.remove(bound_box, do_unlink=True)

        LOGGER.info(f"Final face count for {obj.name}: {len(obj.data.polygons)}")

    except Exception as e:
        error_msg = f"❌ Failed to trim faces for {obj.name}: {e!s}"
        LOGGER.exception(error_msg)
        raise Exception(error_msg)


def convert_objects_to_mesh() -> None:
    """Convert all selected objects to mesh type."""
    LOGGER.info("🔄 Converting objects to mesh")
    bpy.ops.object.select_all(action="SELECT")
    try:
        if len(bpy.context.selected_objects):
            bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
            bpy.ops.object.convert(target="MESH", keep_original=False)
            LOGGER.info("✅ All objects successfully converted to mesh.")
        else:
            LOGGER.warning("⚠ No objects found for conversion. Skipping.")
    except Exception as e:
        error_msg = f"❌ Failed to convert objects to mesh: {e!s}"
        LOGGER.exception(error_msg)
        raise Exception(error_msg)


def process_roads(
    terrain_bounds: tuple[float, float, float, float],
    road_material: bpy.types.Material,
) -> None:
    """Process roads using tiered priority and material assignment.

    Args:
        terrain_bounds: (min_x, max_x, min_y, max_y) in meters
        road_material: Material to apply to selected roads

    """
    LOGGER.info("🛣️ Starting road processing")

    # Step 1: Delete rejected roads early
    for obj in list(bpy.data.objects):
        if any(k in obj.name.lower() for k in REJECTED_ROAD_KEYWORDS):
            LOGGER.debug(f"❌ Rejecting road: {obj.name}")
            bpy.data.objects.remove(obj, do_unlink=True)

    # Step 2: Tiered selection
    selected_roads = []
    for tier, names in TIERS.items():
        objs = [obj for name in names if (obj := bpy.data.objects.get(name))]
        if objs:
            selected_roads = objs
            selected_tier = tier
            LOGGER.info(f"✅ Using Tier {tier} roads")
            break

    if not selected_roads:
        LOGGER.warning("⚠️ No valid road objects found in any tier")
        return

    # Step 3: Remove roads from lower tiers
    for tier, names in TIERS.items():
        if tier <= selected_tier:
            continue
        for name in names:
            obj = bpy.data.objects.get(name)
            if obj:
                LOGGER.debug(f"🗑️ Removing tier {tier} road: {obj.name}")
                bpy.data.objects.remove(obj, do_unlink=True)

    # Step 4: Process selected roads
    for obj in selected_roads:
        LOGGER.info(f"🔄 Processing road: {obj.name}")
        trim_faces_outside_bounds(obj, *terrain_bounds)
        obj.data.materials.clear()
        obj.data.materials.append(road_material)


###############################################################################
# SIONNA PIPELINE SPECIFIC
###############################################################################


def export_mitsuba_scene(scene_folder: str) -> None:
    """Export scene to Mitsuba and save .blend file."""
    LOGGER.info("📤 Exporting Sionna Scene")

    try:
        mitsuba_path = os.path.join(scene_folder, "scene.xml")
        blend_path = os.path.join(scene_folder, "scene.blend")

        bpy.ops.export_scene.mitsuba(
            filepath=mitsuba_path,
            export_ids=True,
            axis_forward="Y",
            axis_up="Z",
        )

        bpy.ops.wm.save_as_mainfile(filepath=blend_path)
        LOGGER.info("✅ Mitsuba scene export complete")
    except Exception as e:
        error_msg = f"❌ Failed to export scene: {e!s}"
        LOGGER.exception(error_msg)


###############################################################################
# WIRELESS INSITE PIPELINE SPECIFIC
###############################################################################


def export_mesh_obj_to_ply(object_type: str, output_folder: str) -> None:
    """Export mesh objects to PLY format."""
    # First deselect everything
    bpy.ops.object.select_all(action="DESELECT")

    # Find and select matching objects
    objects = [o for o in bpy.data.objects if object_type in o.name.lower()]

    # Log all selected object names
    LOGGER.debug(f"🔍 Found objects matching '{object_type}':")
    for obj in objects:
        LOGGER.debug(f"  - {obj.name}")
        obj.select_set(True)

    if objects:
        emoji = "🏗" if "building" in object_type else "🛣"
        LOGGER.info(f"{emoji} Exporting {len(objects)} {object_type}s to .ply")
        ply_path = os.path.join(output_folder, f"{object_type}s.ply")
        if BLENDER_MAJOR_VERSION >= 4:
            bpy.ops.wm.ply_export(
                filepath=ply_path,
                ascii_format=True,
                export_selected_objects=True,
            )
        else:
            bpy.ops.export_mesh.ply(filepath=ply_path, use_ascii=True, use_selection=True)
    else:
        LOGGER.warning(f"⚠ No {object_type}s found for export.")


###############################################################################
# MISC UTILITIES
###############################################################################


def save_bbox_metadata(
    output_folder: str,
    minlat: float,
    minlon: float,
    maxlat: float,
    maxlon: float,
) -> None:
    """Save scenario properties to a metadata file."""
    LOGGER.info("📝 Saving scenario metadata")
    try:
        metadata_path = os.path.join(output_folder, "scenario_info.txt")
        with open(metadata_path, "w") as meta_file:
            meta_file.write(f"Bounding Box: [{minlat}, {minlon}] to [{maxlat}, {maxlon}]\n")
        LOGGER.info("✅ Scenario metadata saved.")
    except Exception as e:
        error_msg = f"❌ Failed to save scenario metadata: {e!s}"
        LOGGER.exception(error_msg)
        raise Exception(error_msg)
