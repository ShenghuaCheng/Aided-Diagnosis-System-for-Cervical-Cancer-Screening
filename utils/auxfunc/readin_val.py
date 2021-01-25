# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: readin_val.py
@Date: 2020/1/14 
@Time: 15:30
@Desc:简单的读取验证数据并前处理
'''
import cv2
import numpy as np
from skimage import exposure


def readin_val(dir, re_size, crop_size, norm=True, gamma=False):
    """简单读入并前处理测试所需的数据
    :param dir: 路径
    :param re_size: 读入后resize到特定的分辨率 w, h
    :param crop_size: 在特定分辨率下裁剪 w, h
    :param norm: 是否归一化处理，默认做
    :param gamma: 是否进行0.6 gamma变化，默认不做
    :return: 返回读入的图片
    """
    img = cv2.imread(dir)
    img = cv2.resize(img, re_size)
    start = ((np.array(re_size)-np.array(crop_size))/2).astype(int)
    img = img[start[1]:start[1]+crop_size[1], start[0]:start[0]+crop_size[0], :]
    if gamma:
        exposure.adjust_gamma(img, 0.6)
    if norm:
        img = (img/255.-0.5)*2
    return img
