"""
Blender Script: Convert OSM Data to PLY Files (Folder Naming Based on Bounding Box)
Each scenario's output is stored in a folder named after its bounding box.
"""

import bpy # type: ignore
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.blender_utils import (
    log_local_setup,
    install_all_addons,
    set_LOGGER,
    export_mitsuba_scene,
    save_osm_origin,
    save_bbox_metadata,
    convert_objects_to_mesh,
    export_mesh_obj_to_ply,
    clear_blender,
    configure_osm_import,
    create_ground_plane,
    setup_world_lighting,
    create_camera_and_render,
    join_and_materialize_objects,
    trim_faces_outside_bounds,
    get_xy_bounds_from_latlon
)

# Setup logging to both console and file (great for debugging)
# root_dir = os.path.dirname(os.path.abspath(__file__))  # TODO: Check if this is NEEDED
logger = log_local_setup('logging_blender_osm.txt')
set_LOGGER(logger)  # So the inner functions can log

# Parse command-line arguments
try:
    minlat = float(sys.argv[sys.argv.index("--minlat") + 1])
    minlon = float(sys.argv[sys.argv.index("--minlon") + 1])
    maxlat = float(sys.argv[sys.argv.index("--maxlat") + 1])
    maxlon = float(sys.argv[sys.argv.index("--maxlon") + 1])
    output_folder = sys.argv[sys.argv.index("--output") + 1]
    
    output_formats = ["insite"] # Default format
    if "--format" in sys.argv:
        output_fmt = sys.argv[sys.argv.index("--format") + 1]

        valid_formats = ["insite", "sionna", "both"]
        if output_fmt not in valid_formats:
            err = f"❌ Invalid format: {output_fmt}. Valid formats are: {', '.join(valid_formats)}"
            logger.error(err)
            raise Exception(err)
        
        output_formats = ["insite", "sionna"] if output_fmt == "both" else [output_fmt]
    
except (ValueError, IndexError):
    err = "❌ Invalid input format. Provide explicit coordinates " \
          "(--minlat --minlon --maxlat --maxlon) and --output for custom output directory."
    logger.error(err)
    raise Exception(err)

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

logger.info(f"📍 Processing Scenario: [{minlat}, {minlon}] to [{maxlat}, {maxlon}]")
logger.info(f"📂 Saving output to: {output_folder}")
logger.info(f"📊 Output formats: {output_formats}")

# Clear existing objects in Blender
clear_blender()

# Automatically install all addons
install_all_addons()

# Configure & Fetch OSM data
configure_osm_import(output_folder, minlat, maxlat, minlon, maxlon)
bpy.ops.blosm.import_data()
logger.info("✅ OSM data imported successfully.")

# Save OSM GPS origin
save_origin = True
if save_origin:
    save_osm_origin(output_folder)

# Save bbox (lats and lons) to a file
save_bbox = False
if save_bbox:
    save_bbox_metadata(output_folder, minlat, minlon, maxlat, maxlon)

# Initialize scene
setup_world_lighting()

BUILDING_MATERIAL = 'itu_concrete'
ROAD_MATERIAL = 'itu_brick'  # TODO: CHECK if Manually changed to asphalt in Sionna

# Create materials (for lighting/coloring and downstream processing)
building_material = bpy.data.materials.new(name=BUILDING_MATERIAL)
building_material.diffuse_color = (0.75, 0.40, 0.16, 1)  # Beige
road_material = bpy.data.materials.new(name=ROAD_MATERIAL)
road_material.diffuse_color = (0.29, 0.25, 0.21, 1)  # Dark grey

# Convert all to meshes
convert_objects_to_mesh()

# Render original scene (no processing)
im_path = os.path.join(output_folder, 'figs', 'cam_org.png')
create_camera_and_render(im_path)

# Process buildings
buildings = join_and_materialize_objects('building', 'buildings', building_material)
if buildings and buildings.type == 'MESH':
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.separate(type='LOOSE')
    bpy.ops.object.mode_set(mode='OBJECT')

# Process roads
terrain_bounds = get_xy_bounds_from_latlon(minlat, minlon, maxlat, maxlon, pad=10)
bpy.ops.object.select_all(action='DESELECT')
for obj in bpy.data.objects:
    if 'terrain' in obj.name.lower() or 'buildings' in obj.name.lower():
        obj.select_set(True)
bpy.ops.object.select_all(action='INVERT')
road_objs = bpy.context.selected_objects
if road_objs:
    for obj in road_objs:
        trim_faces_outside_bounds(obj, *terrain_bounds)
        bpy.context.view_layer.objects.active = obj
        road = bpy.context.active_object
        road.data.materials.clear()
        road.data.materials.append(road_material)

# TODO: CHECK if some known roads are "better" than others - maybe we don't want ALL of them..
# known_roads = ['map.osm_roads_residential', 'map.osm_roads_unclassified', 
#                'map.osm_roads_primary', 'map.osm_paths_footway']

# TODO: MAKE FUNCTIONS to reduce code in main

# Render processed scene
create_camera_and_render(im_path.replace('.png', '_processed.png'))

# TODO: CHECK THE BLENDER ADDONS - DELETE THE MAIN FILES (+ test installation!)

# Export based on the selected format
if "insite" in output_formats:
    logger.info("🔄 Outputting InSite scene...")
    
    # Export buildings and roads
    export_mesh_obj_to_ply("building", output_folder, logger)
    export_mesh_obj_to_ply("road", output_folder, logger)
    
    # TODO: SOMETHING WRONG HERE!!

    logger.info("✅ InSite scene exported.")

if "sionna" in output_formats:
    logger.info("🔄 Outputting Sionna scene...")

    # Create ground plane
    create_ground_plane(minlat, maxlat, minlon, maxlon)

    # Create scene
    export_mitsuba_scene(output_folder)

    # TODO: SOMETHING WRONG HERE!!

    logger.info("✅ Sionna scene exported.")

# Quit Blender   
bpy.ops.wm.quit_blender()
