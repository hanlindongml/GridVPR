
import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm
from typing import Tuple
from argparse import Namespace
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader, Dataset

import visualizations


# Compute R@1, R@5, R@10, R@20
RECALL_VALUES = [1, 5, 10, 20]  # 测试指标


def test(args: Namespace, eval_ds: Dataset, model: torch.nn.Module,
         num_preds_to_save: int = 0) -> Tuple[np.ndarray, str]:
    """Compute descriptors of the given dataset and compute the recalls."""
    
    model = model.eval()  # 模型切换到推理模式
    with torch.no_grad():
        logging.debug("Extracting database descriptors for evaluation/testing")
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))  # 索引database图像
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.test_num_workers,
                                         batch_size=args.infer_batch_size, pin_memory=(args.device == "cuda"))  # 设置读取器
        all_descriptors = np.empty((len(eval_ds), args.fc_output_dim), dtype="float32")  # 创建所有的空描述符
        for images, indices in tqdm(database_dataloader, ncols=100):
            descriptors = model(images.to(args.device))  # 输入模型得到database描述符
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors  # 为database对应的描述符赋值
        
        logging.debug("Extracting queries descriptors for evaluation/testing using batch size 1")
        queries_infer_batch_size = 1  # 每次处理一条查询
        queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, eval_ds.database_num+eval_ds.queries_num)))  # 索引查询图像
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.test_num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=(args.device == "cuda"))  # 设置读取器
        for images, indices in tqdm(queries_dataloader, ncols=100):  # 输入模型得到query描述符
            descriptors = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors  # 为query对应的描述符赋值
    
    queries_descriptors = all_descriptors[eval_ds.database_num:]  # query描述符
    database_descriptors = all_descriptors[:eval_ds.database_num]  # database描述符
    
    # Use a kNN to find predictions
    faiss_index = faiss.IndexFlatL2(args.fc_output_dim)
    faiss_index.add(database_descriptors)
    del database_descriptors, all_descriptors
    
    logging.debug("Calculating recalls")
    _, predictions = faiss_index.search(queries_descriptors, max(RECALL_VALUES))
    
    #### For each queries, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()
    recalls = np.zeros(len(RECALL_VALUES))
    for query_index, preds in enumerate(predictions):
        for i, n in enumerate(RECALL_VALUES):
            if np.any(np.in1d(preds[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    
    # Divide by queries_num and multiply by 100, so the recalls are in percentages
    recalls = recalls / eval_ds.queries_num * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(RECALL_VALUES, recalls)])
    
    # Save visualizations of predictions
    if num_preds_to_save != 0:
        # For each queries save num_preds_to_save predictions
        visualizations.save_preds(predictions[:, :num_preds_to_save], eval_ds, args.output_folder, args.save_only_wrong_preds)
    
    return recalls, recalls_str