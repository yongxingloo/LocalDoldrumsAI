import itertools
import random

import matplotlib.pyplot as plt
import numpy as np

from run_function import *

# Grid Search

parameters = {
    "batch_size": [16, 128, 512],
    "lr": [10**-i for i in range(2, 6)],
    "weight_decay": [10**-i for i in range(2, 6)],
    "optimizer_choice": ["Adam", "SGD", "RMSprop"],
    "features_numerical": [[32, 16, 8, 4], [16, 8, 4], [4, 8, 16]],
    "features_cnn": [[32, 64, 128, 256], [32, 64, 128], [128, 64, 32]],
    "kernel_size": [3],
    "activation_numerical": ["ReLU", "Tanh"],
    "activation_cnn": ["ReLU", "Leaky_ReLU"],
    "activation_final": ["None"],
    "criterion": ["MSE"],
    "stride": [1],
}


combinations = list(itertools.product(*parameters.values()))
param_names = list(parameters.keys())


def random_search(num_iterations, combinations, param_names, num_epochs):
    results = []
    params_list = []
    path_folders = []
    for i in range(num_iterations):
        print(f"Iteration {i+1}/{num_iterations}")
        param_values = random.choice(combinations)
        params = dict(zip(param_names, param_values))
        print(params)
        result, path_folder = MultiInputModelTraining(**params, num_epochs=num_epochs)
        results.append(result)
        params_list.append(params)
        path_folders.append(path_folder)
    return results, params_list, path_folder


results, params_list, path_folder = random_search(
    num_iterations=2, combinations=combinations, param_names=param_names, num_epochs=2
)

df = pd.DataFrame(params_list)
df["test_loss"] = results
df["folder_path"] = path_folder
