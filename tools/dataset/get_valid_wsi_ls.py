# -*- coding: utf-8 -*-
"""
This file is a part of project "Aided-Diagnosis-System-for-Cervical-Cancer-Screening".
See https://github.com/ShenghuaCheng/Aided-Diagnosis-System-for-Cervical-Cancer-Screening for more information.

File name: get_valid_wsi_ls
Description: check the wsi image dir and generate wsi image dir list file for rnn training.
"""

import os
import argparse
from glob2 import glob
from loguru import logger

from core.common import verify_image, check_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Check dir and write wsi dir to txt")
    parser.add_argument("-d", "--wsi_dirs", nargs=argparse.ONE_OR_MORE, default=[], help="wsi directory to check.")
    parser.add_argument("-s", "--save_root", type=str, default=None, help="dir for save list file.")
    parser.add_argument("-n", "--file_names", nargs=argparse.ONE_OR_MORE, default=[])
    args = parser.parse_args()

    wsi_dirs = args.wsi_dirs
    file_names = args.file_names
    save_root = args.save_root

    if save_root is None:
        logger.info("save file in path same as dir")
        file_path = [fp + '.txt' for fp in wsi_dirs]
    else:
        assert len(wsi_dirs) == len(file_names), "nb of dirs and nb of names not match"
        check_dir(save_root, True, logger)
        file_path = [os.path.join(save_root, f"{fn}.txt") for fn in file_names]

    for wsi_dir, save_f in zip(wsi_dirs, file_path):
        logger.info(f"process {wsi_dir}")
        wsi_total = os.listdir(wsi_dir)
        nb_valid = 0
        ls_valid = []
        for wsi_n in wsi_total:
            wsi_img_d = os.path.join(wsi_dir, wsi_n)
            val = True
            for idx in range(50):
                img_p = glob(os.path.join(wsi_img_d, "{:0>3d}_[01].*.[jpt][pni][gf]".format(idx)))[0]
                val &= verify_image(img_p, logger)
            if val:
                ls_valid.append(wsi_img_d + '\n')
                nb_valid += 1
        logger.info(f"number of valid wsis: {nb_valid}/{len(wsi_total)}")
        try:
            with open(save_f, "w+") as f:
                f.writelines(ls_valid)
            logger.info(f"write to {save_f}")
        except Exception as exc:
            logger.exception(exc)



