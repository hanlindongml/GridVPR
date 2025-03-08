The official implementation of "GridVPR: Grid Cell Inspired Visual Place Recognition." In this paper, we proposed a grid cell-inspired VPR method. Our method achieves flexible and effective label assignment. It improves the baseline work and achieves higher data utilization and better performance.

![mapping](https://github.com/user-attachments/assets/70fc3f4a-e561-43d2-a48d-c3e990c88731)

This figure shows the mapping from a grid cell to the real-world geographic space, illustrating the core ideas of the GridVPR. 

# Train

### Generating cache file

As the implementation steps outlined in our paper, we use the command `python generate_centers_by_region.py` to generate the centers of groups. The center represents the center of each firing field of a grid cell. We generate the centers by latitude to speed up the computation process.

Also, the SF-XL dataset is split into different files by split_dataset_txt_by_lat.py.

Then each image in the dataset is divided into different groups and then merged together using the command `python cal_group_and_classes_by_lat.py` and `python merge_group_by_name.py`.

After that, you will receive information about the groups and classes to which each image belongs. You can make the cache file by using `python make_cache.py`. After this step, you will be ready to start the training process! Remember to replace the cache file name in /datasets/train_dataset.py before training. 

### Training

Firstly, ensure that the arguments in parsers.py are set properly, such as the dataset path or the number of groups. Then use the command `python train.py`.

You can also use "--" to control the argument setting. Running `python train.py -h` will provide you with the hyperparameters. Once the training is complete, you can find the model in the ./logs directory.

# Test

You can use the `python eval.py --resume_model path_of_trained_model` command to test on the SF-XL dataset. The network setting is same to the training. For the other dataset, we recomend you to follow [this work](https://github.com/gmberton/VPR-datasets-downloader) to download them and replace the dataset path. 

# Issue

If you have any questions about our work or the implementation, please feel free to contact 51265900020@stu.ecnu.edu.cn. We'd love to hear from you!

# Acknowledgement

Some of the code is reused from [CosPlace](https://github.com/gmberton/CosPlace). You can also find the way to download the SF-XL dataset through this link. Thanks for their great work.
