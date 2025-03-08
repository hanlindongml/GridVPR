import numpy as np
import tqdm
import os


def read_large_text_file(file_path):
    lines = []
    utm_east = []
    utm_north = []
    lat = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            string = line.strip()
            lines.append(string)
            temp = string.split('@')
            lat.append(temp[0][:-2])
            utm_east.append(float(temp[1]))
            utm_north.append(float(temp[2]))
    return np.array(lines), np.array(utm_east), np.array(utm_north), np.array(lat)


# Reading and splitting
file_path = 'database_images_paths.txt'  # Using the file you download from SF-XL dataset
data_array, utm_east, utm_north, lat = read_large_text_file(file_path)

data_merge = [[] for _ in range(12)]

for i in range(12):
    for j in tqdm.trange(0, len(data_array), 12):
        if int(lat[j][-2:]) - 70 == i:
            data_merge[i].append([lat[j], utm_east[j], utm_north[j], data_array[j]])


# Saving
def save_to_txt(data, filename_prefix):
    for i, sublist in enumerate(data):
        with open(f"{filename_prefix}_{i}.txt", "w") as f:
            for item in tqdm.tqdm(sublist):
                f.write(f"{item[0]}, {item[1]}, {item[2]}, {item[3]}\n")


saving_folder_name = "path_by_lat"

if not os.path.exists(saving_folder_name):
    os.makedirs(saving_folder_name)
    print(f"Folder '{saving_folder_name}' has been created successfully!")
else:
    print(f"Folder '{saving_folder_name}' already exists.")

save_to_txt(data_merge, "path_by_lat/lat_")