# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: auxfunc.py
@Date: 2019/11/28 
@Time: 15:12
@Desc:
该文件存储用于前向推理的辅助函数
'''
import configparser
import numpy as np
import scipy.ndimage as ndi
from skimage import morphology, measure, exposure
import cv2


def ini_parser(ini_dir):
    config = configparser.ConfigParser()
    config.read(ini_dir)
    config_dict = {}
    for k in config['DEFAULT']:
        if k in ['m1is', 'm2is']:
            config_dict[k] = [int(s) for s in config['DEFAULT'][k].split(',')]
        elif k == 'fovmode':
            config_dict[k] = config.getboolean('DEFAULT', k)
        else:
            config_dict[k] = config.getfloat('DEFAULT', k)
    return config_dict


def estimate_region(ors, leveltemp=5, threColor=8):
    '''
    估计切片的主要区域，并返回在leveltemp下的二值图片
    :return: 切片主要区域的二值Mask
    '''
    level = leveltemp
    position = (0, 0)
    size = ors.level_dimensions[level]
    img = ors.read_region(position, level, size)
    img = np.array(img)
    img = img[:, :, 0: 3]
    wj1 = img.max(axis=2)
    wj2 = img.min(axis=2)
    wj3 = wj1 - wj2
    imgBin = wj3 > threColor
    threVol = 4
    imgBin = morphology.remove_small_objects(imgBin, min_size = threVol)
    imgBinB = imgBin
    imgBin = cv2.blur(np.float32(imgBin), (30, 30)) > 0
    imgCon, numCon = ndi.label(imgBin)
    num = np.zeros((numCon,))
    for i in range(1, 1+numCon):
        tempImg = imgCon == i
        num[i-1] = np.sum(tempImg)
    ind = np.argmax(num)
    imgBin = imgCon == ind+1
    imgBin = ndi.binary_fill_holes(imgBin)
    if np.sum(imgBin) < 6000000:
        imgBin = imgBinB
        imgBin = cv2.blur(np.float32(imgBin), (50, 50)) > 0
        imgCon, numCon = ndi.label(imgBin)
        num = np.zeros((numCon,))
        for i in range(1, 1+numCon):
            tempImg = imgCon == i
            num[i-1] = np.sum(tempImg)
        ind = np.argmax(num)
        imgBin = imgCon == ind+1
        imgBin = ndi.binary_fill_holes(imgBin)
        if np.sum(imgBin) < 5800000:
            imgBin = imgBinB
            imgBin = cv2.blur(np.float32(imgBin), (80, 80)) > 0
            imgCon, numCon = ndi.label(imgBin)
            num = np.zeros((numCon,))
            for i in range(1, 1+numCon):
                tempImg = imgCon == i
                num[i-1] = np.sum(tempImg)
            ind = np.argmax(num)
            imgBin = imgCon == ind+1
            imgBin = ndi.binary_fill_holes(imgBin)
    return imgBin
