# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 15:58:18 2018

@author: yujingya
"""
import os
import numpy as np
from numpy import *
import random
import cv2
from skimage import  morphology
import scipy.ndimage as ndi
import math
import matplotlib.pyplot as plt
from skimage import exposure

"""
BinarySP:对输入的图片img取前景
3d机器一般：threColor = 10
our机器一般：threColor = 28
level = 1时
threVol = 1000
"""
def BinarySP(img, threColor = 10, threVol= 1000):
    wj1 = img.max(axis=2)
    wj2 = img.min(axis=2)
    wj3 = wj1 - wj2
    imgBin = wj3 > threColor
    imgBin = ndi.binary_fill_holes(imgBin)
    s = np.array([[0,1,0],[1,1,1],[0,1,1]], dtype=np.bool)
    imgBin = ndi.binary_opening(imgBin, s)#可去掉
    imgCon, numCon = ndi.label(imgBin)
    imgConBig = morphology.remove_small_objects(imgCon, min_size=threVol)
    imgBin = imgConBig > 0
    return imgBin  
