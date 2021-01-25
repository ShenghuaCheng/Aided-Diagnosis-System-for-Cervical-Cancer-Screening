# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: Screening
@File: temp.py
@Date: 2019/5/3 
@Time: 10:49
@Desc:
'''
import os
from functools import partial

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import exposure

def szsq_normal(img_szsq):
    """深圳生强系统数据规范
    :param img_szsq:
    :return:
    """
    img_szsq_trans = exposure.adjust_gamma(img_szsq,0.6)
# =============================================================================
#     hsv = cv2.cvtColor(img_szsq_trans, cv2.COLOR_BGR2HSV)
#     hsv =  np.float64(hsv)
#     h = hsv[...,0]*0.973 + 7
#     s = hsv[...,1]*0.929 + 18
#     v = hsv[...,2]*0.976 + 6
#     h[h>255] = 255
#     s[s>255] = 255
#     v[v>255] = 255
#     hsv[...,0] = h
#     hsv[...,1] = s
#     hsv[...,2] = v
#     hsv = np.uint8(hsv)
#     img_szsq_trans = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
# =============================================================================
    return img_szsq_trans

def img_wrt(src_dir, dst_dir):
    img = cv2.imread(src_dir)
    dst_img = szsq_normal(img)
    cv2.imwrite(dst_dir, dst_img)
    return None


data_root = [r'H:\AdaptDATA\test\szsq\sdpc_sfy3']

src_flds = ['origin/nplus']
dst_flds = ['gamma/nplus']

for root in data_root:
    for src_fld in src_flds:
        src_path = os.path.join(root, src_fld)
        dst_path = os.path.join(root, dst_flds[src_flds.index(src_fld)])
        print('%s to %s' % (src_path, dst_path))
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        src_img_name = os.listdir(src_path)
        src_img_name = [cz for cz in src_img_name if '.tif' in cz]
        src_img_dir = [os.path.join(src_path, img) for img in src_img_name]
        dst_img_dir = [os.path.join(dst_path, img) for img in src_img_name]
        for src_dir, dst_dir in zip(src_img_dir, dst_img_dir):
            img_wrt(src_dir, dst_dir)
      






