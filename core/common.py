# -*- coding: utf-8 -*-
"""
This file is a part of project "Aided-Diagnosis-System-for-Cervical-Cancer-Screening".
See https://github.com/ShenghuaCheng/Aided-Diagnosis-System-for-Cervical-Cancer-Screening for more information.

File name: common
Description: useful functions.
"""

import os
from PIL import Image


def check_dir(dir, create=False, logger=None):
    existed = os.path.exists(dir)
    if not existed and create:
        if logger is not None:
            logger.debug(f'create {dir}')
        existed = True if os.makedirs(dir) is None else False
    return existed


def verify_image(image_path, logger=None):
    """ inspired by https://github.com/ultralytics/yolov5/issues/916#issuecomment-862208988
        Check whether the image is corrupted.
    """
    suffix = image_path.split('.')[-1]
    im = Image.open(image_path)
    # file error
    try:
        im.verify()  # PIL verify
    except Exception as exc:
        if logger is not None:
            logger.error(f"{image_path} error! {exc}")
        return False
    # content error
    res = True
    f = open(image_path, 'rb')
    if suffix == "jpg":
        # head FF D8 FF
        # tail FF D9
        f.seek(-2, 2)
        res = f.read() == b'\xff\xd9'
    elif suffix == "png":
        # head 89 50 4E 47
        # tail AE 42 60 82
        f.seek(-4, 2)
        res = f.read() == b'\xae\x42\x60\x82'
    elif suffix == "tif":
        # head 49 49 2A 00
        f.seek(4, 0)
        res = f.read() == b'\x49\x49\x2a\x00'
    elif suffix == "bmp":
        # head 42 4D
        f.seek(2, 0)
        res = f.read() == b'\x42\x4d'
    else:
        # TODO: how to check content of image in other format.
        pass
    f.close()

    if logger is not None and not res:
        logger.error(f"{image_path} error! not a valid image.")
    return res
