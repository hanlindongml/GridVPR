import argparse


def parse_arguments(is_training: bool = True):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--groups_num", type=int, default=16, help="_")
    # Model parameters
    parser.add_argument("--backbone", type=str, default="VGG16",
                        choices=["VGG16",
                                 "ResNet18", "ResNet50", "ResNet101", "ResNet152",
                                 "EfficientNet_B0", "EfficientNet_B1", "EfficientNet_B2",
                                 "EfficientNet_B3", "EfficientNet_B4", "EfficientNet_B5", 
                                 "EfficientNet_B6", "EfficientNet_B7"], help="_")
    parser.add_argument("--fc_output_dim", type=int, default=512,
                        help="Output dimension of final fully connected layer")
    parser.add_argument("--train_all_layers", default=False, action="store_true",
                        help="If true, train all layers of the backbone")
    # Training parameters
    parser.add_argument("--use_amp16", action="store_true",
                        help="use Automatic Mixed Precision")
    parser.add_argument("--augmentation_device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="on which device to run data augmentation")
    parser.add_argument("--batch_size", type=int, default=64, help="_")
    parser.add_argument("--epochs_num", type=int, default=48, help="_")
    parser.add_argument("--iterations_per_epoch", type=int, default=10000, help="_")
    parser.add_argument("--lr", type=float, default=0.00001, help="_")
    parser.add_argument("--classifiers_lr", type=float, default=0.01, help="_")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Width and height of training images (1:1 aspect ratio))")
    parser.add_argument("--resize_test_imgs", default=False, action="store_true",
                        help="If the test images should be resized to image_size along"
                          "the shorter side while maintaining aspect ratio")
    # Data augmentation
    parser.add_argument("--brightness", type=float, default=0.7, help="_")
    parser.add_argument("--contrast", type=float, default=0.7, help="_")
    parser.add_argument("--hue", type=float, default=0.5, help="_")
    parser.add_argument("--saturation", type=float, default=0.7, help="_")
    parser.add_argument("--random_resized_crop", type=float, default=0.5, help="_")
    # Validation / test parameters
    parser.add_argument("--infer_batch_size", type=int, default=32,
                        help="Batch size for inference (validating)")
    parser.add_argument("--test_batch_size", type=int, default=256,
                        help="Batch size for inference (testing)")
    parser.add_argument("--positive_dist_threshold", type=int, default=25,
                        help="distance in meters for a prediction to be considered a positive")
    # Resume parameters
    parser.add_argument("--resume_train", type=str, default=None,
                        help="path to checkpoint to resume, e.g. logs/.../last_checkpoint.pth")
    parser.add_argument("--resume_model", type=str, default=None,
                        help="path to model to resume, e.g. logs/.../best_model.pth")
    # Other parameters
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="_")
    parser.add_argument("--seed", type=int, default=0, help="_")
    parser.add_argument("--num_workers", type=int, default=1, help="_")
    parser.add_argument("--val_num_workers", type=int, default=2, help="_")
    parser.add_argument("--test_num_workers", type=int, default=2, help="_")
    parser.add_argument("--num_preds_to_save", type=int, default=0,
                        help="At the end of training, save N preds for each query. "
                        "Try with a small number like 3")
    parser.add_argument("--save_only_wrong_preds", action="store_true",
                        help="When saving preds (if num_preds_to_save != 0) save only "
                        "preds for difficult queries, i.e. with uncorrect first prediction")
    # Paths parameters
    if is_training:  # train and val sets are needed only for training
        parser.add_argument("--train_set_folder", type=str, default='/root/database',
                            help="path of the folder with training images")
        parser.add_argument("--val_set_folder", type=str, default='/root/processed/val',
                            help="path of the folder with val images (split in database/queries)")
    parser.add_argument("--test_set_folder", type=str, default='/root/processed/test',
                        help="path of the folder with test images (split in database/queries)")
    parser.add_argument("--save_dir", type=str, default="default",
                        help="name of directory on which to save the logs, under logs/save_dir")
    
    args = parser.parse_args()
    
    return args

