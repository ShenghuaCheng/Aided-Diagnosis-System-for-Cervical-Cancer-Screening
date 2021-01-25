# -*- coding: utf-8 -*-
"""
Created on Mon May  6 16:17:48 2019

@author: A-WIN10
"""

from skimage import exposure
import cv2
import numpy as np
import os 
from glob import glob

def szsq_normal(img_szsq):
    img_szsq_trans = exposure.adjust_gamma(img_szsq,0.6)
    hsv = cv2.cvtColor(img_szsq_trans, cv2.COLOR_BGR2HSV)
    hsv =  np.float64(hsv)
    h = hsv[...,0]*0.973 + 7
    s = hsv[...,1]*0.929 + 18
    v = hsv[...,2]*0.976 + 6
    h[h>255] = 255
    s[s>255] = 255
    v[v>255] = 255
    hsv[...,0] = h
    hsv[...,1] = s
    hsv[...,2] = v
    hsv = np.uint8(hsv)
    img_szsq_trans = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)                                      
    return img_szsq_trans


path = r'F:\yujingya\test\new'
filelist = glob(path + '/*.tif')
for file in filelist:
    img_szsq = cv2.imread(file)
    img_szsq_trans = szsq_normal(img_szsq)
    img_save = np.concatenate((img_szsq,img_szsq_trans),axis=1)
    cv2.imwrite(file.replace('new','new_adapt'), img_save)