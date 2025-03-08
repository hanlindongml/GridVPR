import os
import pandas as pd
import numpy as np
from tqdm import tqdm


def get_all_filenames(folder_path):
    return [filename for filename in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, filename))]


def read_txt(filename):
    data = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            values = line.strip().split(", ")
            data.append([float(x) for x in values])
    return data


def save_to_txt(data, filename_prefix):
    for i, sublist in enumerate(data):
        np.savetxt(f"{filename_prefix}_{i}.txt", sublist, fmt="%.2f", delimiter=", ")


def read_image_path_txt(filename):
    return pd.read_csv(filename, header=None, names=['path', 'utm_east', 'utm_north', 'info'])


def main(group_idx):
    group_centers = read_txt(f'group_centers_by_lat/group_centers_{group_idx}.txt')
    metadata = read_image_path_txt(f'path_by_lat/lat__{group_idx}.txt')
    metavalue = metadata[['utm_east', 'utm_north', 'info']].values

    group_centers = np.array(group_centers)

    temp = []

    # Calculate distance for each image
    for item in tqdm(metavalue, total=len(metavalue)):
        image_utm_east, image_utm_north, info = item
        dx = np.abs(group_centers[:, 3] - image_utm_east)
        dy = np.abs(group_centers[:, 4] - image_utm_north)
        distance_squared = dx**2 + dy**2
        within_lat_range = dy <= group_centers[:, 5]
        within_lon_range = dx <= group_centers[:, 5]
        within_distance = distance_squared <= group_centers[:, 6]

        valid_centers = (within_lat_range & within_lon_range & within_distance)
        if np.any(valid_centers):
            matched_centers = group_centers[valid_centers]
            temp.extend([[int(center[1]), int(center[2]), info.split(' ')[1]] for center in matched_centers])

    # Saving
    file_path = "group_info/lat37.7"+str(group_idx)+"_group.txt"
    with open(file_path, 'w') as file:
        for sublist in temp:
            line = ' '.join(map(str, sublist)) + '\n'
            file.write(line)

    print("Number "+str(group_idx)+" of data has been saved to:", file_path)


if __name__ == '__main__':

    saving_folder_name = "group_info"
    if not os.path.exists(saving_folder_name):
        os.makedirs(saving_folder_name)
        print(f"Folder '{saving_folder_name}' has been created successfully!")
    else:
        print(f"Folder '{saving_folder_name}' already exists.")

    for i in range(12):
        main(group_idx=i)
