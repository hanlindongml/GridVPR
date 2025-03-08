import os
import torch
import random
import logging
import numpy as np
from PIL import Image
from PIL import ImageFile
from collections import defaultdict
from tqdm import tqdm


ImageFile.LOAD_TRUNCATED_IMAGES = True

min_images_per_class = 20
cachename = 'database_scale[1,2]_orientation[0,15]_phase[0,2]_mips20.torch'


def get_all_filenames(folder_path):
    filenames = []
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            filenames.append(filename)
    return filenames


logging.debug("Group together images belonging to the same class")
images_per_class = defaultdict(list)

group_id__class_id = []
file_names = get_all_filenames('group_info_merge_name')
for file_name in file_names:
    with open('group_info_merge_name/'+file_name, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            values = line.strip().split(' ')
            image_name = values[0]
            if len(values[1:]) % 2 != 0:
                raise IndexError
            for i in range(int(len(values[1:])/2)):
                group_id = values[1+i*2]
                class_id = group_id + '_' + values[2 + i * 2]
                group_id__class_id.append((group_id, class_id))
                images_per_class[class_id].append(image_name)

images_per_class = {k: v for k, v in images_per_class.items() if
                    len(v) >= min_images_per_class}

group_id__class_id = set(group_id__class_id)

logging.debug("Group together classes belonging to the same group")
classes_per_group = defaultdict(set)
for group_id, class_id in group_id__class_id:
    if class_id not in images_per_class:
        continue  # Skip classes with too few images
    classes_per_group[group_id].add(class_id)
classes_per_group = [list(c) for c in classes_per_group.values()]

torch.save((classes_per_group, images_per_class), cachename)