The official implementation of "GridVPR: Grid Cell Inspired Visual Place Recognition." In this paper, we propose a grid cell-inspired localization neural network. Our method performs multi-scale, rotation-invariant and translation-invariant geo-features aggregation. It improves the baseline work and achieves a better performance than it.

![github](https://github.com/user-attachments/assets/92d7dfc3-b9b1-4ca2-8cad-193828cbe9be)

This figure illustrates the core ideas of the GridVPR. We drew inspiration from the grid cell's spacing encoding method to partition the actual geographical spacing. And our work has demonstrated its effectiveness.

# Train

### Generating cache file

As the implementation steps outlined in our paper, we use the command `python generate_centers_by_region.py` to generate the centers of groups. The center represents the center of each firing field of a grid cell. We generate the centers by latitude to speed up the computation process.

Also, the SF-XL dataset is split into different files by split_dataset_txt_by_lat.py.

Then each image in the dataset is divided into different groups and then merged together using the command `python cal_group_and_classes_by_lat.py` and `python merge_group_by_name.py`.

After that, you will receive information about the groups and classes to which each image belongs. You can make the cache file by using `python make_cache.py`. After this step, you will be ready to start the training process! Remember to replace the cache file name in /datasets/train_dataset.py before training. Additionally, we have provided the cache files we used for training in our paper. You can download them from the links below.

| Features Setting | Cache                                                                                      |
|:----------------:|:------------------------------------------------------------------------------------------:|
| Group1           | [Link](https://drive.google.com/file/d/1jMpVjBuQu4Uvz4BcBtYNgitJivZqo7LD/view?usp=sharing) |
| Group2           | [Link](https://drive.google.com/file/d/1ro3QuvtUiPh-AYgI6WJ7ejg92rc9ysOK/view?usp=sharing) |

Their settings are as follows:

<img src="https://github.com/user-attachments/assets/3b02dfff-20ae-4625-b4e0-616a4357ee09" width="400"/>

### Training

Firstly, ensure that the arguments in parsers.py are set properly, such as the dataset path or the number of groups. Then  use the command `python train.py`.

You can also use "--" to control the argument setting. Running `python train.py -h` will provide you with the hyperparameters. Once the training is complete, you can find the model in the ./logs directory.

# Test

You can use the `python eval.py --resume_model path_of_trained_model` command to test on the SF-XL dataset. The network setting is same to the training. For the other dataset, we recomend you to follow [this work](https://github.com/gmberton/VPR-datasets-downloader) to download them and replace the dataset path. We also provide our trained model as follows:

### Traind Model

| Feature Setting | Model                                                                                      |
|:---------------:|:------------------------------------------------------------------------------------------:|
| Group1          | [Link](https://drive.google.com/file/d/1v3SrCrZ7GZvglY8w_IRy384XcI_xx0GG/view?usp=sharing) |
| Group2          | [Link](https://drive.google.com/file/d/1o_O1M9G3Y1JKjC2si59YzekGMQagweSk/view?usp=sharing) |

# Issue

If you have any questions about our work or the implementation, please feel free to contact 51265900020@stu.ecnu.edu.cn. We'd love to hear from you!

# Acknowledgement

Some of the code is reused from [CosPlace](https://github.com/gmberton/CosPlace). You can also find the way to download the SF-XL dataset through this link. Thanks for their great work.
