import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#name_dataset = 'metopa_des_janmarch'
#name_dataset = 'metopb_des_janmarch'
#name_dataset = 'metopc_des_janmarch'
#name_dataset = 'metopb_des_mayaug'
name_dataset = 'metopc_des_mayaug'
#name_dataset = 'metopc_des_novdec'

images_C01 = np.load(f'../../output/final_output_{name_dataset}/{name_dataset}_goes_images_C01.npy')
images_C03 = np.load(f'../../output/final_output_{name_dataset}/{name_dataset}_goes_images_C03.npy')
images_dual_channel = np.stack((images_C01, images_C03), axis=1)
images_dual_channel = np.nan_to_num(images_dual_channel, nan=0.0)

lat = np.load(f'../../output/final_output_{name_dataset}/{name_dataset}_lat.npy')
lon = np.load(f'../../output/final_output_{name_dataset}/{name_dataset}_lon.npy') 
time = np.load(f'../../output/final_output_{name_dataset}/{name_dataset}_time.npy')
target = np.load(f'../../output/final_output_{name_dataset}/{name_dataset}_label.npy')

#Create boxes for each range

idx_0 = np.where((target >= 0) & (target < 1))[0]
idx_1 = np.where((target >= 1) & (target < 2))[0]
idx_2 = np.where((target >= 2) & (target < 3))[0]
idx_3 = np.where((target >= 3) & (target < 4))[0]
idx_4 = np.where((target >= 4) & (target < 5))[0]
idx_5 = np.where((target >= 5) & (target < 6))[0]
idx_6 = np.where((target >= 6) & (target < 7))[0]
idx_7 = np.where((target >= 7) & (target < 8))[0]
idx_8 = np.where((target >= 8) & (target < 9))[0]
idx_9 = np.where((target >= 9) & (target < 10))[0]
idx_10 = np.where((target >= 10) & (target < 11))[0]
idx_11 = np.where((target >= 11) & (target < 12))[0]
idx_12 = np.where((target >= 12))[0]

min_length = np.min([len(idx_0), len(idx_1), len(idx_2), len(idx_3), len(idx_4), len(idx_5), len(idx_6), len(idx_7), len(idx_8), len(idx_9), len(idx_10), len(idx_11), len(idx_12)])

samples_idx = []
number_of_samples_per_range = min_length

samples_idx.extend(np.random.choice(idx_0, number_of_samples_per_range, replace=False))
samples_idx.extend(np.random.choice(idx_1, number_of_samples_per_range, replace=False))
samples_idx.extend(np.random.choice(idx_2, number_of_samples_per_range, replace=False))
samples_idx.extend(np.random.choice(idx_3, number_of_samples_per_range, replace=False))
samples_idx.extend(np.random.choice(idx_4, number_of_samples_per_range, replace=False))
samples_idx.extend(np.random.choice(idx_5, number_of_samples_per_range, replace=False))
samples_idx.extend(np.random.choice(idx_6, number_of_samples_per_range, replace=False))
samples_idx.extend(np.random.choice(idx_7, number_of_samples_per_range, replace=False))
samples_idx.extend(np.random.choice(idx_8, number_of_samples_per_range, replace=False))
samples_idx.extend(np.random.choice(idx_9, number_of_samples_per_range, replace=False))
samples_idx.extend(np.random.choice(idx_10, number_of_samples_per_range, replace=False))
samples_idx.extend(np.random.choice(idx_11, number_of_samples_per_range, replace=False))
samples_idx.extend(np.random.choice(idx_12, number_of_samples_per_range, replace=False))

print(np.shape(samples_idx))

images_dual_channel_picked = images_dual_channel[samples_idx]
lat_picked = lat[samples_idx]
lon_picked = lon[samples_idx]
time_picked = time[samples_idx]
target_picked = target[samples_idx]

#Split the data 80 / 20

images_train, images_test, lat_train, lat_test, lon_train, lon_test, time_train, time_test, target_train, target_test = train_test_split(
    images_dual_channel_picked, lat_picked, lon_picked, time_picked, target_picked, test_size=0.2, random_state=42)


np.save(f'../masterdata_filtered/train/{name_dataset}_train_images_scalability.npy', images_train)
np.save(f'../masterdata_filtered/train/{name_dataset}_train_lat_scalability.npy', lat_train)
np.save(f'../masterdata_filtered/train/{name_dataset}_train_lon_scalability.npy', lon_train)
np.save(f'../masterdata_filtered/train/{name_dataset}_train_time_scalability.npy', time_train)
np.save(f'../masterdata_filtered/train/{name_dataset}_train_target_scalability.npy', target_train)

np.save(f'../masterdata_filtered/test/{name_dataset}_test_images.npy', images_test)
np.save(f'../masterdata_filtered/test/{name_dataset}_test_lat.npy', lat_test)
np.save(f'../masterdata_filtered/test/{name_dataset}_test_lon.npy', lon_test)
np.save(f'../masterdata_filtered/test/{name_dataset}_test_time.npy', time_test)
np.save(f'../masterdata_filtered/test/{name_dataset}_test_target.npy', target_test)

print(np.shape(images_train), np.shape(lat_train), np.shape(lon_train), np.shape(time_train), np.shape(target_train))
print(np.shape(images_test), np.shape(lat_test), np.shape(lon_test), np.shape(time_test), np.shape(target_test))