"""Blender Script: Convert OSM Data to PLY Files (Folder Naming Based on Bounding Box).

Each scenario's output is stored in a folder named after its bounding box.
"""

from pathlib import Path

import bpy  # type: ignore[import-not-found]

from .utils.blender_utils import (
    add_materials_to_objs,
    clear_blender,
    configure_osm_import,
    convert_objects_to_mesh,
    create_camera_and_render,  # sometimes incompatible in headless servers
    create_ground_plane,
    export_mesh_obj_to_ply,
    export_mitsuba_scene,
    install_blender_addon,
    log_local_setup,
    save_bbox_metadata,
    save_osm_origin,
    set_logger,
    setup_world_lighting,
)


def fetch_osm_scene(  # noqa: PLR0913 - signature follows Blender/OSM requirements
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
            output_formats (list[str], optional): List of output formats.
                Defaults to ["insite"]. Possible values: "insite", "sionna"

    """
    # Check if the folder already exists
    if output_formats is None:
        output_formats = ["insite"]
    if Path(output_folder).exists():
        print(f"⏩ Folder '{output_folder}' already exists. Skipping OSM extraction.")
        return

    # Create output directory if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Setup logging to both console and file (great for debugging)
    log_file = str(Path(output_folder) / "logging_blender_osm.txt")
    logger = log_local_setup(log_file)
    set_logger(logger)  # So the inner functions can log

    logger.info("📍 Processing Scenario: [%s, %s] to [%s, %s]", minlat, minlon, maxlat, maxlon)
    logger.info("📂 Saving output to: %s", output_folder)
    logger.info("📊 Output formats: %s", output_formats)

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

    building_material_name = "itu_concrete"

    # Create materials (for lighting/coloring and downstream processing)
    building_material = bpy.data.materials.new(name=building_material_name)
    building_material.diffuse_color = (0.75, 0.40, 0.16, 1)  # Beige

    # Convert all to meshes
    convert_objects_to_mesh()

    # Render original scene (no processing)
    im_path = str(Path(output_folder) / "figs", "cam_org.png")
    create_camera_and_render(im_path)

    # Process buildings
    add_materials_to_objs("building", building_material)

    # Render processed scene
    create_camera_and_render(im_path.replace(".png", "_processed.png"))

    # Export based on the selected format
    if "insite" in output_formats:
        logger.info("🔄 Outputting InSite scene...")

        # Export buildings
        export_mesh_obj_to_ply("building", output_folder)

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
