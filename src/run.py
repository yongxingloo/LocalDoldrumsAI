"""
@author: Yongxing Loo
"""

import datetime
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

experiment_name = "Classification"
"""Network Parameters"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_height = 27
image_width = 27
in_channels = 2  # Number of channels in the input image (C01 and C03)
num_numerical_inputs = (
    2  # Number of numerical inputs (lat, lon, hour(sin), hour(cos), day(sin), day(cos))
)
num_epochs = 10


# IN GRIDSEARCH

criterion = (
    nn.BCEWithLogitsLoss()
)  # "MSELoss", "L1Loss" "BCEWithLogitsLoss" "CrossEntropyLoss"
optimizer_choice = "Adam"  # "Adam", "SGD", "RMSprop"

batch_size = 512
lr = 0.001
weight_decay = 0.001  # L2 regularization

features_numerical = [4, 8, 16]
features_cnn = [32, 64, 128]

kernel_size = 3
stride = 1

activation_numerical = nn.ReLU()
activation_cnn = nn.ReLU()
activation_final = nn.Identity()


model = MultiInputModel(
    image_height,
    image_width,
    num_numerical_inputs,
    features_cnn,
    features_numerical,
    kernel_size,
    in_channels,
    activation_cnn,
    activation_numerical,
    activation_final,
    stride,
).to(device)

# Early stopping parameters

patience_epochs = 50
patience_loss = 0.001

# Create directory for all saving

path_folder = create_folder()

print(device)
print(
    "hyperparameters:",
    "criterion:",
    criterion,
    "lr:",
    lr,
    "num_epochs:",
    num_epochs,
    "batch_size:",
    batch_size,
)

# Load the data

images_dual_channel = np.load(
    "../local_data/classification/classifier_train-validation/train_images_classification.npy"
)
lat = np.load(
    "../local_data/classification/classifier_train-validation/train_lat_classification.npy"
)
lon = np.load(
    "../local_data/classification/classifier_train-validation/train_lon_classification.npy"
)
time = np.load(
    "../local_data/classification/classifier_train-validation/train_time_classification.npy"
)
target = np.load(
    "../local_data/classification/classifier_train-validation/train_target_classification.npy"
)

# Load the test data

images_dual_channel_test = np.load(
    "../local_data/classification/classifier_test/test_images_classification.npy"
)
lat_test = np.load(
    "../local_data/classification/classifier_test/test_lat_classification.npy"
)
lon_test = np.load(
    "../local_data/classification/classifier_test/test_lon_classification.npy"
)
time_test = np.load(
    "../local_data/classification/classifier_test/test_time_classification.npy"
)
target_test = np.load(
    "../local_data/classification/classifier_test/test_target_classification.npy"
)

# Prepare the data (Expermiental)

Normalization = True
solar_angle_transform = True

if solar_angle_transform:
    saa = solar_azimuth_angle(lat, lon, time)
    sza = solar_zenith_angle(lat, lon, time)

if Normalization:
    images_dual_channel, _, image_mean, image_std, target_mean, target_std = (
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
dataloader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)


for images, numerical_input, target in dataloader:
    print(images.shape)
    print(numerical_input.shape)
    print(target.shape)
    print("example target:", target)
    print("example numerical input:", numerical_input)
    break

print("Data loaded !")

# Train / Test split
print("Start of training !")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)

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
    criterion,
    device,
    optimizer_choice,
    patience_epochs,
    patience_loss,
    path_folder,
)

print("Training done !")

# Ploting and saving the results

plot_save_loss(
    best_val_outputs,
    best_val_labels,
    train_losses,
    val_losses,
    path_folder,
    saving=False,
)

# Test the model


saa_test = solar_azimuth_angle(lat_test, lon_test, time_test)
sza_test = solar_zenith_angle(lat_test, lon_test, time_test)

images_dual_channel_test_norm, _ = normalization_mean_std_test(
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
    target_test,
    transform=None,
)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

test_output, test_loss = test_model(model, test_loader, criterion, device)

accuracy = calculate_accuracy(test_output, target_test)

# Rescale the output and target for minmax

# test_output = test_output * target_std + target_mean

np.save(os.path.join(path_folder, "test_output.npy"), test_output)
