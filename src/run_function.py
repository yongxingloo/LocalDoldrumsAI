"""
@author: Yongxing Loo
"""

import datetime
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

from func import *
from impl import *
from MultiInputModel import *

images_dual_channel = np.load("../../Subset + Split/train-validation/train_images.npy")
lat = np.load("../../Subset + Split/train-validation/train_lat.npy")
lon = np.load("../../Subset + Split/train-validation/train_lon.npy")
time = np.load("../../Subset + Split/train-validation/train_time.npy")
target = np.load("../../Subset + Split/train-validation/train_target.npy")

# Load the test data

images_dual_channel_test = np.load("../../Subset + Split/test/test_images.npy")
lat_test = np.load("../../Subset + Split/test/test_lat.npy")
lon_test = np.load("../../Subset + Split/test/test_lon.npy")
time_test = np.load("../../Subset + Split/test/test_time.npy")
target_test = np.load("../../Subset + Split/test/test_target.npy")

Normalization = True
solar_angle_transform = True

if solar_angle_transform:
    saa = solar_azimuth_angle(lat, lon, time)
    sza = solar_zenith_angle(lat, lon, time)

if Normalization:
    images_dual_channel, target, image_mean, image_std, target_mean, target_std = (
        normalization_mean_std(images_dual_channel, target)
    )
    saa, sza, saa_mean, saa_std, sza_mean, sza_std = normalization_solar_angles(
        saa, sza
    )

# Bring the data into the expected format

dataset = MultiInputDataset(
    images_dual_channel,
    saa,
    sza,
    target,
    transform=None,
)


# Train / Test split
print("Start of training !")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)


saa_test = solar_azimuth_angle(lat_test, lon_test, time_test)
sza_test = solar_zenith_angle(lat_test, lon_test, time_test)

images_dual_channel_test_norm, target_test_norm = normalization_mean_std_test(
    images_dual_channel_test,
    target_test,
    image_mean,
    image_std,
    target_mean,
    target_std,
)
saa_test, sza_test = normalization_solar_angles_test(
    saa_test, sza_test, saa_mean, saa_std, sza_mean, sza_std
)

test_dataset = MultiInputDataset(
    images_dual_channel_test_norm,
    saa_test,
    sza_test,
    target_test_norm,
    transform=None,
)


def clear_logging_handlers():
    logger = logging.getLogger()
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)


def MultiInputModelTraining(
    batch_size,
    lr,
    weight_decay,
    optimizer_choice,
    features_numerical,
    features_cnn,
    kernel_size,
    activation_numerical,
    activation_cnn,
    activation_final,
    criterion,
    stride,
    num_epochs,
):
    experiment_name = "GS"
    """Network Parameters"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_height = 27
    image_width = 27
    in_channels = 2  # Number of channels in the input image (C01 and C03)
    num_numerical_inputs = 2  # Number of numerical inputs (lat, lon, hour(sin), hour(cos), day(sin), day(cos))

    activation_map = {
        "ReLU": nn.ReLU(),
        "Tanh": nn.Tanh(),
        "Leaky_ReLU": nn.LeakyReLU(),
        "sigmoid": nn.Sigmoid(),
        "None": nn.Identity(),
    }

    criterion_map = {
        "MSE": nn.MSELoss(),
        "L1Loss": nn.L1Loss(),
    }

    model = MultiInputModel(
        image_height,
        image_width,
        num_numerical_inputs,
        features_cnn,
        features_numerical,
        kernel_size,
        in_channels,
        activation_map[activation_cnn],
        activation_map[activation_numerical],
        activation_map[activation_final],
        stride,
    ).to(device)

    # Early stopping parameters

    patience_epochs = 30
    patience_loss = 0.001

    # Create directory for all saving

    path_folder = create_folder()

    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(path_folder, "./hyperparameters.log"),
        filemode="a",
        format="%(asctime)s   %(levelname)s   %(message)s",
    )
    logging.info("Hyperparameters")
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Learning rate: {lr}")
    logging.info(f"Weight decay: {weight_decay}")
    logging.info(f"Optimizer choice: {optimizer_choice}")
    logging.info(f"Features numerical: {features_numerical}")
    logging.info(f"Features CNN: {features_cnn}")
    logging.info(f"Kernel size: {kernel_size}")
    logging.info(f"Activation numerical: {activation_numerical}")
    logging.info(f"Activation CNN: {activation_cnn}")
    logging.info(f"Activation final: {activation_final}")
    logging.info(f"Criterion: {criterion}")
    logging.info(f"Stride: {stride}")
    logging.info(f"Number of epochs: {num_epochs}")

    clear_logging_handlers()

    # Prepare the data (Expermiental)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=False, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, drop_last=False)

    # Train the model

    best_val_outputs, best_val_labels, model, train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs,
        lr,
        weight_decay,
        criterion_map[criterion],
        device,
        optimizer_choice,
        patience_epochs,
        patience_loss,
        path_folder,
    )

    print("Training done !")

    # Test the model
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    test_output, test_loss = test_model(
        model, test_loader, criterion_map[criterion], device
    )

    # Rescale the output and target for minmax

    test_output = test_output * target_std + target_mean

    np.save(os.path.join(path_folder, "test_output.npy"), test_output)

    msescore = MSE(test_output, target_test)

    return msescore, path_folder
