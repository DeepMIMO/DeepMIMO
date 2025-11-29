import re
import time

import numpy as np


def gen_pl_p2m_file(n_lines=100):
    file_content = """# <Transmitter Set: Tx: 1 BS - Point 1> 
        # <Receiver Set: Rx: 2 ue_grid> 
        # <X(m)> <Y(m)> <Z(m)> <Distance(m)> <PathLoss(dB)>"""
    file_content += "".join(
        [f"\n{i} -90 -60 1.53079 96.0221 108.661" for i in range(1, int(n_lines) + 1)],
    )

    test_file = f"file.pl.n={n_lines}.p2m"
    with open(test_file, "w") as fp:
        fp.writelines(file_content)
    return test_file


test_file = gen_pl_p2m_file(10000)


def read_pl_p2m_file(filename: str):
    """Returns xyz, distance, pl from p2m file."""
    assert filename.endswith(".p2m")  # should be a .p2m file
    assert ".pl." in filename  # should be the pathloss p2m

    # Initialize empty lists for matrices
    xyz_list = []
    dist_list = []
    path_loss_list = []

    # Define (regex) patterns to match numbers (optionally signed floats)
    re_data = r"-?\d+\.?\d*"

    # If we want to preallocate matrices, count lines
    # num_lines = sum(1 for _ in open(filename, 'rb'))

    with open(filename) as fp:
        lines = fp.readlines()

    for line in lines:
        if line[0] != "#":
            data = re.findall(re_data, line)
            xyz_list.append([float(data[1]), float(data[2]), float(data[3])])  # XYZ (m)
            dist_list.append([float(data[4])])  # distance (m)
            path_loss_list.append([float(data[5])])  # path loss (dB)

    # Convert lists to numpy arrays
    xyz_matrix = np.array(xyz_list, dtype=np.float32)
    dist_matrix = np.array(dist_list, dtype=np.float32)
    path_loss_matrix = np.array(path_loss_list, dtype=np.float32)

    return xyz_matrix, dist_matrix, path_loss_matrix


# Line count execution
t1 = time.time()
line_count = sum(1 for line in open(test_file, "rb"))
t2 = time.time()
print("Line count time:", t2 - t1, "seconds")

# Time the execution
t1 = time.time()
xyz_matrix, dist_matrix, path_loss_matrix = read_pl_p2m_file(test_file)
t2 = time.time()

# Print the results
print("Execution Time:", t2 - t1, "seconds")
print("XYZ Matrix Shape:", xyz_matrix.shape)
print("Distance Matrix Shape:", dist_matrix.shape)
print("Path Loss Matrix Shape:", path_loss_matrix.shape)
