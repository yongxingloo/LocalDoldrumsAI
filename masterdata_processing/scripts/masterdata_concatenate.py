import os

import numpy as np


def concatenate_npy_with_variable(folder_path, variable_name):
    """
    Find and concatenate all .npy files in a folder containing a specific variable name,
    in alphabetical order of the filenames.

    Parameters:
        folder_path (str): Path to the folder to search for .npy files.
        variable_name (str): String to search for in the filenames.

    Returns:
        np.ndarray: Concatenated array from the selected .npy files.
    """
    # List all files in the directory and filter for .npy with variable_name
    npy_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".npy") and variable_name in f
    ]

    if not npy_files:
        raise ValueError(f"No .npy files with '{variable_name}' found in {folder_path}")

    # Sort files alphabetically
    npy_files.sort()
    print(npy_files)

    # Load and concatenate arrays
    arrays = [np.load(file) for file in npy_files]
    concatenated_array = np.concatenate(arrays, axis=0)

    return concatenated_array


images_train = concatenate_npy_with_variable("../masterdata_filtered/train", "images")
lat_train = concatenate_npy_with_variable("../masterdata_filtered/train", "lat")
lon_train = concatenate_npy_with_variable("../masterdata_filtered/train", "lon")
time_train = concatenate_npy_with_variable("../masterdata_filtered/train", "time")
target_train = concatenate_npy_with_variable("../masterdata_filtered/train", "target")

# save

np.save("../masterdata_train/masterdata_images_train.npy", images_train)
np.save("../masterdata_train/masterdata_lat_train.npy", lat_train)
np.save("../masterdata_train/masterdata_lon_train.npy", lon_train)
np.save("../masterdata_train/masterdata_time_train.npy", time_train)
np.save("../masterdata_train/masterdata_target_train.npy", target_train)

images_test = concatenate_npy_with_variable("../masterdata_filtered/test", "images")
lat_test = concatenate_npy_with_variable("../masterdata_filtered/test", "lat")
lon_test = concatenate_npy_with_variable("../masterdata_filtered/test", "lon")
time_test = concatenate_npy_with_variable("../masterdata_filtered/test", "time")
target_test = concatenate_npy_with_variable("../masterdata_filtered/test", "target")

np.save("../masterdata_test/masterdata_images_test.npy", images_test)
np.save("../masterdata_test/masterdata_lat_test.npy", lat_test)
np.save("../masterdata_test/masterdata_lon_test.npy", lon_test)
np.save("../masterdata_test/masterdata_time_test.npy", time_test)
np.save("../masterdata_test/masterdata_target_test.npy", target_test)
