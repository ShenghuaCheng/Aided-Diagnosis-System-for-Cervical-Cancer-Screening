# -*- coding: utf-8 -*-
"""
This file is a part of project "Aided-Diagnosis-System-for-Cervical-Cancer-Screening".
See https://github.com/ShenghuaCheng/Aided-Diagnosis-System-for-Cervical-Cancer-Screening for more information.

File name: get_valid_img_ls
Description: check the image and generate image list file for training.
"""

import os
import argparse
from loguru import logger

from core.common import verify_image, check_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Check dir and write img path to txt")
    parser.add_argument("-d", "--img_dirs", nargs=argparse.ONE_OR_MORE, default=[], help="image directory to check.")
    parser.add_argument("-s", "--save_root", type=str, default=None, help="dir for save list file.")
    parser.add_argument("-n", "--file_names", nargs=argparse.ONE_OR_MORE, default=[])
    args = parser.parse_args()

    img_dirs = args.img_dirs
    file_names = args.file_names
    save_root = args.save_root

    if save_root is None:
        logger.info("save file in path same as dir")
        file_path = [fp + '.txt' for fp in img_dirs]
    else:
        assert len(img_dirs) == len(file_names), "nb of dirs and nb of names not match"
        check_dir(save_root, True, logger)
        file_path = [os.path.join(save_root, f"{fn}.txt") for fn in file_names]

    for img_d, save_f in zip(img_dirs, file_path):
        logger.info(f"process {img_d}")
        img_total = os.listdir(img_d)
        nb_valid = 0
        ls_valid = []
        for img_n in img_total:
            img_p = os.path.join(img_d, img_n)
            val = verify_image(img_p, logger)
            if val:
                ls_valid.append(img_p + '\n')
                nb_valid += 1
        logger.info(f"number of valid images: {nb_valid}/{len(img_total)}")
        try:
            with open(save_f, "w+") as f:
                f.writelines(ls_valid)
            logger.info(f"write to {save_f}")
        except Exception as exc:
            logger.exception(exc)



