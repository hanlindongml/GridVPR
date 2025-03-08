import sys
import torch
import logging
import numpy as np
from tqdm import tqdm
import multiprocessing
from datetime import datetime
import torchvision.transforms as T

import test_utils
import util
import parsers
import commons
import cosface_loss
import augmentations
from cosplace_model import cosplace_network
from datasets.test_dataset import TestDataset
from datasets.train_dataset import TrainDataset

torch.backends.cudnn.benchmark = True  # Provides a speedup

import warnings
if __name__ == '__main__':
    multiprocessing.freeze_support()

    args = parsers.parse_arguments()

    test_ds = TestDataset(args.test_set_folder, positive_dist_threshold=args.positive_dist_threshold,
                          image_size=args.image_size, resize_test_imgs=args.resize_test_imgs)

    model = cosplace_network.GeoLocalizationNet(args.backbone, args.fc_output_dim, args.train_all_layers)

    best_model_state_dict = torch.load('input your path')
    model.load_state_dict(best_model_state_dict["model_state_dict"])
    model.eval().cuda()

    logging.info(f"Now testing on the test set: {test_ds}")
    recalls, recalls_str = test_utils.test(args, test_ds, model, args.num_preds_to_save)
    print(recalls, recalls_str)
    logging.info(f"{test_ds}: {recalls_str}")
