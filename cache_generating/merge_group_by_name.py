import os


saving_folder_name = "group_info_merge_name"
if not os.path.exists(saving_folder_name):
    os.makedirs(saving_folder_name)
    print(f"Folder '{saving_folder_name}' has been created successfully!")
else:
    print(f"Folder '{saving_folder_name}' already exists.")

for i in range(70, 82):
    input_file = 'group_info/lat37.'+str(i)+'_group.txt'
    output_file = 'group_info_merge_name/lat37.'+str(i)+'.txt'

    keyword_dict = {}

    with open(input_file, 'r') as file:
        for line in file:
            data = line.strip().split(' ')
            keyword = data[2]
            features = ' '.join(data[0:2])

            if keyword in keyword_dict:
                keyword_dict[keyword].append(features)
            else:
                keyword_dict[keyword] = [features]

    with open(output_file, 'w') as file:
        for keyword, features_list in keyword_dict.items():
            orientations = [0, 1]
            for ori in orientations:
                temp_list = keyword.split('@')
                temp_list[9] = str(ori*30)
                keyword_change_orientation = '@'.join(temp_list)
                output_line = keyword_change_orientation + ' ' + ' '.join(features_list) + '\n'
                file.write(output_line)
