"""Blender Script: Convert OSM Data to PLY Files (Folder Naming Based on Bounding Box)
Each scenario's output is stored in a folder named after its bounding box.
"""

import os

import bpy  # type: ignore

from .utils.blender_utils import (
    add_materials_to_objs,
    clear_blender,
    configure_osm_import,
    convert_objects_to_mesh,
    create_camera_and_render,  # sometimes incompatible in headless servers
    create_ground_plane,
    export_mesh_obj_to_ply,
    export_mitsuba_scene,
    get_xy_bounds_from_latlon,
    install_blender_addon,
    log_local_setup,
    process_roads,
    save_bbox_metadata,
    save_osm_origin,
    set_LOGGER,
    setup_world_lighting,
)


def fetch_osm_scene(
    minlat: float,
    minlon: float,
    maxlat: float,
    maxlon: float,
    output_folder: str,
    output_formats: list[str] | None = None,
) -> None:
    """Process an OpenStreetMap scene and export it in the specified formats.

    Args:
            minlat (float): Minimum latitude of the bounding box
            minlon (float): Minimum longitude of the bounding box
            maxlat (float): Maximum latitude of the bounding box
            maxlon (float): Maximum longitude of the bounding box
            output_folder (str): Path to the output folder
            output_formats (list[str], optional): List of output formats. Defaults to ["insite"].
                                                                                      Possible values: "insite", "sionna"

    """
    # Check if the folder already exists
    if output_formats is None:
        output_formats = ["insite"]
    if os.path.exists(output_folder):
        print(f"⏩ Folder '{output_folder}' already exists. Skipping OSM extraction.")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Setup logging to both console and file (great for debugging)
    log_file = os.path.join(output_folder, "logging_blender_osm.txt")
    logger = log_local_setup(log_file)
    set_LOGGER(logger)  # So the inner functions can log

    logger.info(f"📍 Processing Scenario: [{minlat}, {minlon}] to [{maxlat}, {maxlon}]")
    logger.info(f"📂 Saving output to: {output_folder}")
    logger.info(f"📊 Output formats: {output_formats}")

    # Clear existing objects in Blender
    clear_blender()

    # Automatically install all addons
    install_blender_addon("blosm")
    if "sionna" in output_formats:
        install_blender_addon("mitsuba-blender")

    # Configure & Fetch OSM data
    configure_osm_import(output_folder, minlat, maxlat, minlon, maxlon)
    bpy.ops.blosm.import_data()
    logger.info("✅ OSM data imported successfully.")

    # Save OSM GPS origin (needed for pipeline!)
    save_osm_origin(output_folder)

    # Save bbox (lats and lons) to a file (just for reference)
    save_bbox_metadata(output_folder, minlat, minlon, maxlat, maxlon)

    # Initialize scene
    setup_world_lighting()

    BUILDING_MATERIAL = "itu_concrete"
    ROAD_MATERIAL = "itu_brick"

    # Create materials (for lighting/coloring and downstream processing)
    building_material = bpy.data.materials.new(name=BUILDING_MATERIAL)
    building_material.diffuse_color = (0.75, 0.40, 0.16, 1)  # Beige
    road_material = bpy.data.materials.new(name=ROAD_MATERIAL)
    road_material.diffuse_color = (0.29, 0.25, 0.21, 1)  # Dark grey

    # Convert all to meshes
    convert_objects_to_mesh()

    # Render original scene (no processing)
    im_path = os.path.join(output_folder, "figs", "cam_org.png")
    create_camera_and_render(im_path)

    # Process buildings
    add_materials_to_objs("building", building_material)

    # Process roads
    terrain_bounds = get_xy_bounds_from_latlon(minlat, minlon, maxlat, maxlon, pad=40)
    process_roads(terrain_bounds, road_material)  # Filter, trim to bounds and add material

    # Render processed scene
    create_camera_and_render(im_path.replace(".png", "_processed.png"))

    # Export based on the selected format
    if "insite" in output_formats:
        logger.info("🔄 Outputting InSite scene...")

        # Export buildings and roads
        export_mesh_obj_to_ply("building", output_folder)
        export_mesh_obj_to_ply("road", output_folder)

        logger.info("✅ InSite scene exported.")

    if "sionna" in output_formats:
        logger.info("🔄 Outputting Sionna scene...")

        # Create ground plane
        create_ground_plane(minlat, maxlat, minlon, maxlon)

        # Create scene
        export_mitsuba_scene(output_folder)

        logger.info("✅ Sionna scene exported.")

    # Quit Blender
    bpy.ops.wm.quit_blender()


if __name__ == "__main__":
    # Example usage
    minlat = 40.68503298
    minlon = -73.84682129
    maxlat = 40.68597435
    maxlon = -73.84336302
    output_folder = "C:/Users/jmora/Downloads/osm_root/city_0_newyork_3p5_119"
    output_formats = ["insite"]  # or ["insite", "sionna"] for both formats

    fetch_osm_scene(minlat, minlon, maxlat, maxlon, output_folder, output_formats)
