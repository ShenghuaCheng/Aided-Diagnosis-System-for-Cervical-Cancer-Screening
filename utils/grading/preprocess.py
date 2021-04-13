# -*- coding:utf-8 -*-
import random
import cv2
from skimage import exposure


def trans_img(img_dir):
    tmp = cv2.imread(img_dir)
    tmp = cv2.resize(tmp, (384, 384))
    tmp = tmp[int((384 - 256)/2): int((384 - 256)/2)+256, int((384 - 256) / 2): int((384 - 256) / 2) + 256, :]
    tmp = (tmp / 255 - 0.5) * 2
    return tmp


def enhance_img(img_dir):
    tmp = cv2.imread(img_dir)
    tmp = cv2.resize(tmp, (384, 384))
    h_start = random.randint(0, 384 - 256)
    w_start = random.randint(0, 384 - 256)
    tmp = tmp[h_start: h_start+256, w_start: w_start + 256, :]
    gamma = 0.6 + random.random()*0.8
    tmp = exposure.adjust_gamma(tmp, gamma)
    tmp = (tmp / 255 - 0.5) * 2
    return tmp

