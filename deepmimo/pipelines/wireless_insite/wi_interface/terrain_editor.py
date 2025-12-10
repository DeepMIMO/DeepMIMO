"""TerrainEditor module for terrain file manipulation.

This module provides functionality to create and edit terrain (.ter) files for
electromagnetic simulations, including setting vertex positions and material properties.
"""

from pathlib import Path

import numpy as np


class TerrainEditor:
    """Class for creating and editing terrain (.ter) files.

    This class provides methods to set vertex positions for a flat rectangular terrain,
    incorporate material properties, and save the resulting terrain file.

    Attributes:
        template_ter_file (str): Path to the template terrain file
        file (list[str]): Contents of the terrain file
        material_file (Optional[list[str]]): Contents of the material file

    """

    def __init__(self, template_ter_file: str | None = None) -> None:
        """Initialize the TerrainEditor with a template terrain file.

        Args:
            template_ter_file (str, optional): Path to the template terrain file.

        """
        self.template_ter_file = template_ter_file
        if template_ter_file is None:
            script_dir = str(Path(str(Path(__file__).resolve()).parent))
            self.template_ter_file = str(
                Path(script_dir) / "..",
                "resources",
                "feature",
                "newTerrain.ter",
            )

        with Path(self.template_ter_file).open() as f:
            self.file = f.readlines()

    def set_vertex(self, xmin: float, ymin: float, xmax: float, ymax: float, z: float = 0) -> None:
        """Set the vertices of a flat rectangular terrain.

        Creates a flat rectangular terrain with the specified dimensions by setting
        the vertices of two triangles that form the rectangle.

        Args:
            xmin (float): Minimum x-coordinate
            ymin (float): Minimum y-coordinate
            xmax (float): Maximum x-coordinate
            ymax (float): Maximum y-coordinate
            z (float, optional): z-coordinate (height). Defaults to 0.

        """
        v1 = np.asarray([xmin, ymin, z])
        v2 = np.asarray([xmax, ymin, z])
        v3 = np.asarray([xmax, ymax, z])
        v4 = np.asarray([xmin, ymax, z])

        # First triangle (v1, v2, v3)
        self.file[40] = f"{v1[0]:.10f} {v1[1]:.10f} {v1[2]:.10f}\n"
        self.file[41] = f"{v2[0]:.10f} {v2[1]:.10f} {v2[2]:.10f}\n"
        self.file[42] = f"{v3[0]:.10f} {v3[1]:.10f} {v3[2]:.10f}\n"

        # Second triangle (v4, v1, v3)
        self.file[47] = f"{v4[0]:.10f} {v4[1]:.10f} {v4[2]:.10f}\n"
        self.file[48] = f"{v1[0]:.10f} {v1[1]:.10f} {v1[2]:.10f}\n"
        self.file[49] = f"{v3[0]:.10f} {v3[1]:.10f} {v3[2]:.10f}\n"

    def set_material(self, material_path: str) -> None:
        """Set the material properties for the terrain.

        Reads a material file and incorporates its properties into the terrain file.

        Args:
            material_path (str): Path to the material file

        """
        with Path(material_path).open() as f:
            self.material_file = f.readlines()

        # Find the material section in the terrain file
        for i in range(len(self.file)):
            if self.file[i].startswith("begin_<Material>"):
                start = i
            if self.file[i].startswith("end_<Material>"):
                end = i

        # Replace the material section with the new material properties
        self.file = self.file[:start] + self.material_file + self.file[end + 1 :]

        # Replace 'none' by 'directive_with_backscatter' in 'diffuse_scattering_model'
        idx_to_replace = next(
            line_idx
            for line_idx, line in enumerate(self.file)
            if line == "diffuse_scattering_model none\n"
        )
        self.file[idx_to_replace] = "diffuse_scattering_model directive_with_backscatter\n"

    def save(self, outfile_path: str) -> None:
        """Save the terrain file.

        Args:
            outfile_path (str): Path to save the terrain file

        """
        # clean the output file before writing
        Path(outfile_path).open("w+").close()

        with Path(outfile_path).open("w") as out:
            out.writelines(self.file)


if __name__ == "__main__":
    material_path = "resources/material/ITU Wet earth 2.4 GHz.mtl"
    outfile_path = "test/newTerrain.ter"
    editor = TerrainEditor()
    editor.set_vertex(-200, -200, 200, 200, 0)
    editor.set_material(material_path)
    editor.save(outfile_path)
    print("done")
