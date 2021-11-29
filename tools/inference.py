# -*- coding: utf-8 -*-
"""
This file is a part of project "Aided-Diagnosis-System-for-Cervical-Cancer-Screening".
See https://github.com/ShenghuaCheng/Aided-Diagnosis-System-for-Cervical-Cancer-Screening for more information.

File name: inference.py
Description: Do inference to WSI.
"""

import os
import argparse
import time

from loguru import logger

from core.config import load_config
from core.driver import Inference

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Do inference to WSI, getting the recommended results.")
    parser.add_argument("-c", "--config_file", type=str, help="path to config file.")
    parser.add_argument("-f", "--wsi_files", type=str, nargs=argparse.ONE_OR_MORE,
                        help="path to WSI file or txt recording WSI file list.")
    parser.add_argument("--intermediate", default=False, action="store_true",
                        help="save intermediate results")
    args = parser.parse_args()

    config_file = args.config_file
    wsi_files = args.wsi_files
    intermediate = args.intermediate

    cfg = load_config(config_file, None)
    logger.info(f"inference config: \n{cfg}")
    inference = Inference(cfg)

    # parse wsi file
    logger.info("parsing wsi file ...")
    wsi_dict = {"others": []}
    for wsi_file in wsi_files:
        if wsi_file.endswith(".txt"):
            grp_name = os.path.basename(wsi_file).split('.')[0]
            with open(wsi_file, 'r') as f:
                wsi_dict[grp_name] = [l.strip() for l in f.readlines()]
            logger.info(f"Group: {grp_name} - {len(wsi_file[grp_name])} slides")
        else:
            wsi_dict["others"].append(wsi_file)
    logger.info(f"Group: others - {len(wsi_dict['others'])} slides")

    # process per group
    for grp_n, wsi_ls in wsi_dict.items():
        start_time = time.time()
        for wsi_p in wsi_ls:
            logger.info(f"processing {grp_n} - {wsi_p}")
            since = time.time()
            # set wsi path, group name and recorder config
            inference.set_wsi(wsi_p, grp_n)
            inference.intermediate = intermediate
            # do inference
            inference.inference()
            logger.info("time: {:.2f} s".format(time.time()-since))

        total_time = time.time() - start_time
        logger.info(
            "finish Group {} - {:d} slides, total time: {:.2f} mins, avg time: {:.2f} mins/slide".format(
                grp_n, len(wsi_ls), total_time/60, total_time/60/len(wsi_ls)
            )
        )
    logger.info("finish inference")
