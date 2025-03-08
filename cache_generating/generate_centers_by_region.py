import math
import numpy as np
from tqdm import tqdm
import utm
import os
from time import time


# SF-XL dataset coverage
UTM_EAST_MIN = 542823
UTM_EAST_MAX = 555844
UTM_NORTH_MIN = 4172649
UTM_NORTH_MAX = 4184989
START_CENTER = (int((UTM_EAST_MIN+UTM_EAST_MAX)/2), int((UTM_NORTH_MIN+UTM_NORTH_MAX)/2))  # Start from the center of the area
RADIUS = 5
E = 5  # Tolerable errors


def utm_to_latlong(u, zone_number=10, zone_letter='S'):
    u = np.array(u)
    if len(u.shape) > 1:
        return np.array([utm_to_latlong(u_i, zone_number=zone_number, zone_letter=zone_letter) for u_i in u])

    easting, northing = u
    return utm.to_latlon(easting, northing, zone_number=zone_number, zone_letter=zone_letter)


def rotate_and_translate(Position_A, Position_B, angle_degrees, translation):

    A_X, A_Y = Position_A
    B_X, B_Y = Position_B
    translation_X, translation_Y = translation

    delta_x = B_X - A_X
    delta_y = B_Y - A_Y

    angle_radians = math.radians(-1 * angle_degrees)

    delta_x_prime = delta_x * math.cos(angle_radians) - delta_y * math.sin(angle_radians)
    delta_y_prime = delta_x * math.sin(angle_radians) + delta_y * math.cos(angle_radians)

    B_X_prime = A_X + delta_x_prime
    B_Y_prime = A_Y + delta_y_prime

    translation_matrix = np.array([[1, 0, translation_X], [0, 1, translation_Y], [0, 0, 1]])
    center_transformed = np.dot(translation_matrix[:2, :2], (B_X_prime, B_Y_prime)) + translation_matrix[:2, 2]

    return tuple([int(x) for x in center_transformed])


def generate_hexagon(curr_center, scale, x_range=None, y_range=None, spacing=4):
    if y_range is None:
        y_range = [UTM_NORTH_MIN - scale * RADIUS - E, UTM_NORTH_MAX + scale * RADIUS + E]
    if x_range is None:
        x_range = [UTM_EAST_MIN - scale * RADIUS - E, UTM_EAST_MAX + scale * RADIUS + E]
    points = []
    for i in range(6):
        angle_rad = math.radians(60 * i)
        x = round(curr_center[0] + spacing * math.cos(angle_rad) * RADIUS * scale, 2)
        y = round(curr_center[1] + spacing * math.sin(angle_rad) * RADIUS * scale, 2)
        if x_range[0] <= x <= x_range[1] and y_range[0] <= y <= y_range[1]:
            new_center = (x, y)
            points.append(new_center)
    return points


def generate_centers_for_group(scale, orientation, phase):
    storage = set()
    storage.add(START_CENTER)
    queue = [START_CENTER]

    while queue:
        current_center = queue.pop(0)
        new_centers = generate_hexagon(current_center, scale)
        for new_center in new_centers:
            if new_center not in storage:
                queue.append(new_center)
                storage.add(new_center)

    centers_for_group = []
    for center in storage:
        centers_for_group.append(rotate_and_translate(START_CENTER, center, orientation, phase))

    for center in centers_for_group.copy():
        if center[0] <= UTM_EAST_MIN - scale * RADIUS or center[0] >= UTM_EAST_MAX + scale * RADIUS:
            centers_for_group.remove(center)
    for center in centers_for_group.copy():
        if center[1] <= UTM_NORTH_MIN - scale * RADIUS or center[1] >= UTM_NORTH_MAX + scale * RADIUS:
            centers_for_group.remove(center)

    return centers_for_group


def save_to_txt(data, filename_prefix):
    for i, sublist in enumerate(data):
        with open(f"{filename_prefix}_{i}.txt", "w") as f:
            for item in tqdm(sublist):
                f.write(f"{item[0]}, {item[1]}, {item[2]}, {item[3]}, {item[4]}, {item[5]}, {item[6]}\n")


if __name__ == '__main__':

    # Generating centers
    t0 = time()
    centers_groups = []

    # You can change the feature setting here
    scale_list = [1, 2]
    orientation_list = [0, 15]
    phase_scale = [0, 2]
    phase_list = []
    for x in phase_scale:
        for y in phase_scale:
            phase_list.append([x * RADIUS, y * RADIUS])

    for scale in scale_list:
        for orientation in orientation_list:
            for phase in phase_list:
                revision_phase = [phase[0]*scale, phase[1]*scale]
                centers_groups.append(generate_centers_for_group(scale, orientation, revision_phase))
                now_time = time()
                print('Complete a group of centers generation, taking a total of ' + str(round((now_time - t0), 2)) + 's.')

    # Saving
    for group in centers_groups:
        print(len(group))

    group_radius_square = [25, 25, 25, 25, 25, 25, 25, 25, 100, 100, 100, 100, 100, 100, 100, 100]
    group_radius = [5, 5, 5, 5, 5, 5, 5, 5, 10, 10, 10, 10, 10, 10, 10, 10]

    flattened_list = [[] for _ in range(13)]

    for group_idx, group in enumerate(centers_groups):
        for element_idx, element in enumerate(group):
            latlong = utm_to_latlong(element)
            region_idx = str(latlong[0])[0:5]
            region_list_idx = int(region_idx[3:]) - 70
            center = [region_idx, group_idx, element_idx]
            center.extend(element)
            center.append(group_radius[group_idx])
            center.append(group_radius_square[group_idx])
            flattened_list[region_list_idx].append(center)

    saving_folder_name = "group_centers_by_lat"

    if not os.path.exists(saving_folder_name):
        os.makedirs(saving_folder_name)
        print(f"Folder '{saving_folder_name}' has been created successfully!")
    else:
        print(f"Folder '{saving_folder_name}' already exists.")

    save_to_txt(flattened_list, "group_centers_by_lat/group_centers")