import os
import sys
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../src')
import numpy as np
import matplotlib.pyplot as plt

from func import *
from impl import *
from MultiInputModel import *

"""Model Path"""

PATH = '../model_weights/model.pt'

""" Dataset to be used for inference """

images_dual_channel = np.load('./sliced/images_dual_channel.npy')
lat = np.load('./sliced/lat_array.npy')
lon = np.load('./sliced/lon_array.npy')
time = np.load('./sliced/time_array.npy')

"""Network Parameters"""


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_height            = 27
image_width             = 27
in_channels             = 2  # Number of channels in the input image (C01 and C03)
batch_size              = 256
num_numerical_inputs    = 6  # Number of numerical inputs (lat, lon, hour(sin), hour(cos), day(sin), day(cos))
num_epochs              = 400
criterion               = nn.MSELoss()
model                   = MultiInputModel(image_height, image_width,num_numerical_inputs).to(device)
lr                      = 0.001

interpretibility = True


print(device)
print('hyperparameters:',
      'criterion:',criterion,
      'lr:',lr,
      'num_epochs:',num_epochs,
      'batch_size:',batch_size)


# Loading the model 
model = MultiInputModel(image_height, image_width,num_numerical_inputs).to(device)
model.load_state_dict(torch.load(PATH, map_location=device))
model.eval()


# Define your inference 
inference_dataset = MultiInputDataset(images_dual_channel, lat, lon, time, transform=None, time_transform=DateTimeCyclicEncoder())

# Create DataLoader for inference
inference_loader = torch.utils.data.DataLoader(inference_dataset, batch_size, shuffle=False)

# Iterate through the DataLoader for inference
for images, numerical_input in inference_loader:
    print(images.shape)
    print(numerical_input.shape)
    # Perform inference with your model here
    # outputs = model(images, numerical_input)
    # print(outputs)
    break

all_outputs = []

with torch.no_grad():  # Disable gradient calculation for inference
    for images, numerical_input in inference_loader:
        images = images.to(device)
        numerical_input = numerical_input.to(device)
        
        # Perform inference with your model
        outputs = model(images, numerical_input)
        
        # Move outputs to CPU and convert to numpy
        outputs = outputs.cpu().numpy()
        
        # Append outputs to the list
        all_outputs.append(outputs)

# Concatenate all outputs into a single numpy array
all_outputs = np.concatenate(all_outputs, axis=0)

# Print the shape of the final output array
print("Inference completed. Output shape:", all_outputs.shape)

# Optionally, save the outputs to a file
np.save("inference_outputs.npy", all_outputs)