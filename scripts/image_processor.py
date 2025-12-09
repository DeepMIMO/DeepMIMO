# %%
"""Utilities for zooming and saving ray-tracing images."""

from pathlib import Path

from PIL import Image


def process_image(input_path: str, output_path: str, zoom_factor: float = 1.5) -> None:
    """Resize, zoom, and crop an image before saving to a new path."""
    # Open the image
    img = Image.open(input_path)

    # Ensure the image is 1080p if it isn't already
    if img.size != (1920, 1080):
        img = img.resize((1920, 1080), Image.Resampling.LANCZOS)

    # Calculate dimensions for zooming
    width, height = img.size
    new_width = int(width * zoom_factor)
    new_height = int(height * zoom_factor)

    # Zoom in by resizing to larger dimensions
    zoomed_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Calculate coordinates for trimming back to original size
    left = (new_width - width) // 2
    top = (new_height - height) // 2
    right = left + width
    bottom = top + height

    # Trim the image back to original size
    final_img = zoomed_img.crop((left, top, right, bottom))

    # Save the processed image
    final_img.save(output_path, quality=95)


# %%
if __name__ == "__main__":
    # Example usage
    base_run = Path("M:/AutoRayTracingSionna/all_runs_sionna/run_04-07-2025_18H13M23S")
    folder = base_run / "city_1_losangeles_3p5_s" / "figs"
    fold_name = folder.parent.name
    # Replace with your input/output image paths as needed
    input_image = folder / f"{fold_name}_processed.png"
    output_image = folder / f"{fold_name}_processed_zoomed.png"

    try:
        process_image(input_image, output_image)
        print(f"Image processed successfully! Saved as {output_image}")
    except (OSError, ValueError) as e:
        print(f"An error occurred: {e!s}")

    # %%

    main_folder = base_run
    for folder in [p.name for p in main_folder.iterdir()]:
        if folder.startswith("._") or not folder.startswith("city_"):
            continue
        print(f"running: {folder}")
        input_image = main_folder / folder / "figs" / f"{folder}_processed.png"
        output_image = main_folder / folder / "figs" / f"{folder}_processed_zoomed.png"
        process_image(str(input_image), str(output_image))

# %%
