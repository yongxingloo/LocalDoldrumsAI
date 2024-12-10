import os
import numpy as np

import os
import numpy as np

def concatenate_npy_with_variable(folder_path, variable_name):
    """
    Find and concatenate all .npy files in folders with 'final_output' in their name,
    where filenames contain a specific variable name, in alphabetical order of the filenames.

    Parameters:
        folder_path (str): Path to the folder to search for .npy files.
        variable_name (str): String to search for in the filenames.

    Returns:
        np.ndarray: Concatenated array from the selected .npy files.
    """
    # List all files in directories with 'final_output' in their name and filter for .npy with variable_name
    npy_files = [
        os.path.join(root, f)
        for root, dirs, files in os.walk(folder_path)
        if any("final_output" in dir_name for dir_name in dirs) or "final_output" in os.path.basename(root)
        for f in files
        if f.endswith(".npy") and variable_name in f
    ]

    if not npy_files:
        raise ValueError(f"No .npy files with '{variable_name}' found in folders containing 'final_output' in {folder_path}")

    # Sort files alphabetically
    npy_files.sort()
    print(npy_files)

    # Load and concatenate arrays
    arrays = [np.load(file) for file in npy_files]
    concatenated_array = np.concatenate(arrays, axis=0)

    return concatenated_array



lat = concatenate_npy_with_variable('../../../output/', 'lat')
lon = concatenate_npy_with_variable('../../../output/', 'lon')
time = concatenate_npy_with_variable('../../../output/', 'time')
label = concatenate_npy_with_variable('../../../output/', 'label')


np.save('../../masterdata_all/unsplit/lat_all',lat)
np.save('../../masterdata_all/unsplit/lon_all',lon)
np.save('../../masterdata_all/unsplit/time_all',time)
np.save('../../masterdata_all/unsplit/label_all',label)




print('done')