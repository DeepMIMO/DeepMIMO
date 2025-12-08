"""ObjectEditor module for 3D object manipulation.

This module provides functionality to read, transform, and save 3D object files
used in electromagnetic simulations.
"""

from math import cos, radians, sin


class ObjectEditor:
    """Class for editing 3D object files.

    This class provides methods to read, transform (translate and rotate), and save
    3D object files used in electromagnetic simulations.

    Attributes:
        infile_path (str): Path to the input object file
        file (List[str]): Contents of the file as lines of text

    """

    def __init__(self, infile_path: str) -> None:
        """Initialize the ObjectEditor with an input file path.

        Args:
            infile_path (str): Path to the input object file

        """
        self.infile_path = infile_path
        self.read_file()

    def read_file(self) -> None:
        """Read the object file into memory."""
        with open(self.infile_path) as f:
            self.file = f.readlines()
        print()

    def transform(self, translate: list[float] | None = None, rotate_angle: float = 0) -> None:
        """Transform the object by translation and rotation.

        Applies translation and rotation to all vertices in the object file.
        Rotation is performed around the Z-axis (flat earth assumption).

        Args:
            translate (List[float], optional): Translation vector [x, y, z]. Defaults to [0, 0, 0].
            rotate_angle (float, optional): Rotation angle in degrees. Defaults to 0.

        """
        if translate is None:
            translate = [0, 0, 0]
        for i in range(len(self.file)):
            if "nVertices" in self.file[i]:
                num_vertex = int(self.file[i].split()[1])
                for j in range(i + 1, i + 1 + num_vertex):
                    vals = self.file[j].split()
                    x = float(vals[0])
                    y = float(vals[1])
                    z = float(vals[2])

                    # Rotation (Flat earth assumption) - should be extended for slopes
                    cos_z_rot = cos(radians(rotate_angle))
                    sin_z_rot = sin(radians(rotate_angle))
                    x_new = cos_z_rot * x - sin_z_rot * y
                    y_new = sin_z_rot * x + cos_z_rot * y
                    z_new = z

                    # Translation
                    x_new += translate[0]
                    y_new += translate[1]
                    z_new += translate[2]

                    self.file[j] = str.format(
                        "{:12.10f} {:12.10f} {:12.10f} \n",
                        x_new,
                        y_new,
                        z_new,
                    )

    def save(self, outfile_path: str) -> None:
        """Save the transformed object to a file.

        Args:
            outfile_path (str): Path to save the output object file

        """
        # clean the output file before writing
        open(outfile_path, "w+").close()

        with open(outfile_path, "w") as out:
            out.writelines(self.file)


if __name__ == "__main__":
    infile_path = "scenario/car_models/audiA7_lowpoly.object"
    outfile_path = "scenario/car_models/audiA7_lowpoly_test.object"
    editor = ObjectEditor(infile_path)
    editor.transform([30, 40, 50], 90)
    editor.save(outfile_path)
    print("done")
