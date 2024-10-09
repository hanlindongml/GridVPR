import sys
import torch
import logging
import numpy as np
from tqdm import tqdm
import multiprocessing
from datetime import datetime
import torchvision.transforms as T

import test
import util
import parsers
import commons
import cosface_loss
import augmentations
from cosplace_model import cosplace_network
from datasets.test_dataset import TestDataset
from datasets.train_dataset import TrainDataset

torch.backends.cudnn.benchmark = True  # Provides a speedup

def main():
    # 参数日志相关
    args = parsers.parse_arguments()
    start_time = datetime.now()
    args.output_folder = f"/root/autodl-tmp/logs/{args.save_dir}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
    commons.make_deterministic(args.seed)
    commons.setup_logging(args.output_folder, console="debug")
    logging.info(" ".join(sys.argv))
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {args.output_folder}")
    logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")

    #### Model
    model = cosplace_network.GeoLocalizationNet(args.backbone, args.fc_output_dim, args.train_all_layers)

    # 模型断点训练
    if args.resume_model is not None:
        logging.debug(f"Loading model from {args.resume_model}")
        model_state_dict = torch.load(args.resume_model)
        model.load_state_dict(model_state_dict)

    model = model.to(args.device).train()  # 加载到cuda

    #### Optimizer
    criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    #### Datasets 对每个组获取一个traindataset类，并返回一个groups列表：【组1, 组2, ...】
    groups = [TrainDataset(args, args.train_set_folder, M=args.M, alpha=args.alpha, N=args.N, L=args.L,
                           current_group=n, min_images_per_class=args.min_images_per_class) for n in range(args.groups_num)]  # 定义分组，返回列表
    # Each group has its own classifier, which depends on the number of classes in the group
    classifiers = [cosface_loss.MarginCosineProduct(args.fc_output_dim, len(group)) for group in groups]  # 为每组定义分类器
    classifiers_optimizers = [torch.optim.Adam(classifier.parameters(), lr=args.classifiers_lr) for classifier in classifiers]  # 为每个分类器定义优化器

    # 输出分组信息
    logging.info(f"Using {len(groups)} groups")
    logging.info(f"The {len(groups)} groups have respectively the following number of classes {[len(g) for g in groups]}")
    logging.info(f"The {len(groups)} groups have respectively the following number of images {[g.get_images_num() for g in groups]}")

    # 读取验证集和测试集
    val_ds = TestDataset(args.val_set_folder, positive_dist_threshold=args.positive_dist_threshold,
                         image_size=args.image_size, resize_test_imgs=args.resize_test_imgs)
    test_ds = TestDataset(args.test_set_folder, queries_folder="queries_v1",
                          positive_dist_threshold=args.positive_dist_threshold,
                          image_size=args.image_size, resize_test_imgs=args.resize_test_imgs)
    logging.info(f"Validation set: {val_ds}")
    logging.info(f"Test set: {test_ds}")

    #### Resume 如果是从断点恢复训练
    if args.resume_train:
        model, model_optimizer, classifiers, classifiers_optimizers, best_val_recall1, start_epoch_num = \
            util.resume_train(args, args.output_folder, model, model_optimizer, classifiers, classifiers_optimizers)
        model = model.to(args.device)
        # start_epoch_num += 1
        epoch_num = start_epoch_num - 1
        logging.info(f"Resuming from epoch {start_epoch_num} with best R@1 {best_val_recall1:.1f} from checkpoint {args.resume_train}")
    else:
        best_val_recall1 = start_epoch_num = 0

    #### Train / evaluation loop
    # 输出训练信息
    logging.info("Start training ...")
    logging.info(f"There are {len(groups[0])} classes for the first group, " +
                 f"each epoch has {args.iterations_per_epoch} iterations " +
                 f"with batch_size {args.batch_size}, therefore the model sees each class (on average) " +
                 f"{args.iterations_per_epoch * args.batch_size / len(groups[0]):.1f} times per epoch")

    # 进行数据增强
    if args.augmentation_device == "cuda":
        gpu_augmentation = T.Compose([
                # 颜色调整的转换，调整图像的亮度、对比度、饱和度和色调
                augmentations.DeviceAgnosticColorJitter(brightness=args.brightness,
                                                        contrast=args.contrast,
                                                        saturation=args.saturation,
                                                        hue=args.hue),
                # 随机裁剪，随机裁剪输入图像为指定的大小
                augmentations.DeviceAgnosticRandomResizedCrop([args.image_size, args.image_size],
                                                              scale=[1-args.random_resized_crop, 1]),
                # 归一化，将图像的像素值归一化为指定的均值和标准差
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    # 混合精度训练
    if args.use_amp16:
        scaler = torch.cuda.amp.GradScaler()

    # 对每个epoch(epoch * group_num)
    for epoch_num in range(start_epoch_num, args.epochs_num):

        #### Train
        epoch_start_time = datetime.now()  # 获取当前时间
        # Select classifier and dataloader according to epoch
        current_group_num = epoch_num % args.groups_num  # 计算当前epoch对应的组
        # current_group_num = 16
        classifiers[current_group_num] = classifiers[current_group_num].to(args.device)   # 选择对应的分类器
        util.move_to_device(classifiers_optimizers[current_group_num], args.device)  # 将优化器移到设备上

        dataloader = commons.InfiniteDataLoader(groups[current_group_num], num_workers=args.num_workers,
                                                batch_size=args.batch_size, shuffle=True,
                                                pin_memory=(args.device == "cuda"), drop_last=True)  # 加载数据

        dataloader_iterator = iter(dataloader)  # 数据集迭代器
        model = model.train()  # 模型训练迭代

        epoch_losses = np.zeros((0, 1), dtype=np.float32)  # 初始化epoch_loss
        for iteration in tqdm(range(args.iterations_per_epoch), ncols=100):  # 每一代中训练的元素个数（iteration是什么？为什么是复数？）
            images, targets, _ = next(dataloader_iterator)  # 取数据并放到设备中
            images, targets = images.to(args.device), targets.to(args.device)

            if args.augmentation_device == "cuda":
                images = gpu_augmentation(images)

            model_optimizer.zero_grad()
            classifiers_optimizers[current_group_num].zero_grad()  # 模型与对应的分类器梯度清零

            if not args.use_amp16:
                descriptors = model(images)  # 图片放入model得到描述符
                output = classifiers[current_group_num](descriptors, targets)  # 放入分类器，这里输入targets的原因是把输出拉到同一维度
                loss = criterion(output, targets)  # 损失计算回传
                loss.backward()
                epoch_losses = np.append(epoch_losses, loss.item())
                del loss, output, images
                model_optimizer.step()
                classifiers_optimizers[current_group_num].step()
            else:  # Use AMP 16
                with torch.cuda.amp.autocast():
                    descriptors = model(images)
                    output = classifiers[current_group_num](descriptors, targets)
                    loss = criterion(output, targets)
                scaler.scale(loss).backward()
                epoch_losses = np.append(epoch_losses, loss.item())
                del loss, output, images
                scaler.step(model_optimizer)
                scaler.step(classifiers_optimizers[current_group_num])
                scaler.update()

        # 将分类器和优化器放回到CPU上
        classifiers[current_group_num] = classifiers[current_group_num].cpu()
        util.move_to_device(classifiers_optimizers[current_group_num], "cpu")

        # 输出信息
        logging.debug(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                      f"loss = {epoch_losses.mean():.4f}")

        #### Evaluation
        recalls, recalls_str = test.test(args, val_ds, model)  # 计算召回率
        logging.info(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, {val_ds}: {recalls_str[:20]}")
        # 如果出现了更高的召回率，那么更新并保存
        is_best = recalls[0] > best_val_recall1
        best_val_recall1 = max(recalls[0], best_val_recall1)
        # Save checkpoint, which contains all training parameters
        util.save_checkpoint({
            "epoch_num": epoch_num + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": model_optimizer.state_dict(),
            "classifiers_state_dict": [c.state_dict() for c in classifiers],
            "optimizers_state_dict": [c.state_dict() for c in classifiers_optimizers],
            "best_val_recall1": best_val_recall1
        }, is_best, args.output_folder, 'epoch_'+str(epoch_num)+'.pth')


    logging.info(f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

    #### Test best model on test set v1
    best_model_state_dict = torch.load(f"{args.output_folder}/best_model.pth")
    model.load_state_dict(best_model_state_dict)

    logging.info(f"Now testing on the test set: {test_ds}")
    recalls, recalls_str = test.test(args, test_ds, model, args.num_preds_to_save)
    logging.info(f"{test_ds}: {recalls_str}")

    logging.info("Experiment finished (without any errors)")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()