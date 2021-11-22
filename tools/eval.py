# -*- coding: utf-8 -*-
"""
This file is a part of project "Aided-Diagnosis-System-for-Cervical-Cancer-Screening".
See https://github.com/ShenghuaCheng/Aided-Diagnosis-System-for-Cervical-Cancer-Screening for more information.

File name: eval.py
Description: eval models
"""
import os
import sys
import argparse
import random

import numpy as np
from sklearn import metrics
from loguru import logger
from tabulate import tabulate

from core.config import load_config
from core.common import check_dir


def cal_metrics(name, labels, scores, thresh=0.5):
    classifications = (scores > thresh).astype(int)
    acc = metrics.accuracy_score(labels, classifications)
    if len(set(labels.tolist())) > 1:
        auc = metrics.roc_auc_score(labels, scores)
    else:
        auc = 0.
    nb = len(scores)
    return (name, nb, acc, auc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Train parser")
    parser.add_argument("-n", "--config_name", type=str, default=None, help="train config name")
    parser.add_argument("-f", "--config_file", type=str, default=None, help="train config file")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("-w", "--weights", default=None, type=str, nargs=argparse.ONE_OR_MORE, help="weights to eval")
    parser.add_argument("-t", "--threshold", default=0.5, type=float, help="threshold for classification")
    args = parser.parse_args()
    config = load_config(args.config_file, args.config_name)

    random.seed(42)  # fix global seed

    weights = args.weights
    if weights is None:
        print("Please specify some weights.")
        sys.exit(1)
    threshold = args.threshold
    assert 0. <= threshold <= 1., "range of threshold should be [0, 1]"
    model = config.create_model(weights)
    model.trainable = False
    logger.info(f"evaluate weight: {weights}")

    test_loader = config.get_test_loader()
    test_loader.batch_size = args.batch_size

    labels, p_images, p_masks, grp_names = test_loader.get_cur_labels()

    results = model.predict_generator(
        generator=test_loader,
        max_queue_size=config.max_queue_size,
        verbose=1
    )
    results = results.flatten()

    save_dir = os.path.join(config.output_dir, "eval", config.config_name)
    check_dir(save_dir, True, logger)
    weight_name = os.path.basename(weights[0]).split('.')[0]
    # record original results
    with open(os.path.join(save_dir, weight_name + ".txt"), "w+") as f:
        for i in range(len(labels)):
            f.write(','.join(
                    [
                        grp_names[i],
                        str(labels[i]),
                        str(results[i]),
                        p_images[i],
                        p_masks[i] if p_masks[i] is not None else "null"
                    ]
                ) + '\n'
            )
    logger.info(f"write raw results to {os.path.join(save_dir,weight_name + '.txt')}")
    labels = np.array(labels)
    # group
    group_index = {
        "Total": list(range(len(labels))),
        "Pos": np.where(labels == 1)[0].tolist(),
        "Neg": np.where(labels == 0)[0].tolist()
    }
    for idx, grp in enumerate(grp_names):
        if grp not in group_index.keys():
            group_index[grp] = [idx]
        else:
            group_index[grp].append(idx)
    # record grouped results
    table_header = ["Name", "Num", "Acc/Recall", "AUC"]
    res_table = []
    # cal metrics
    for k, v in group_index.items():
        res_table.append(cal_metrics(k, labels[v], results[v], threshold))
    logger.info(f"Results: \n {tabulate(res_table, headers=table_header, tablefmt='fancy_grid')}")
