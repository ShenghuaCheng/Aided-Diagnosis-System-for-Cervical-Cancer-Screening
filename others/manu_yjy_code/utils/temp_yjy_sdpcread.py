# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: Screening
@File: temp_yjy_sdpcread.py
@Date: 2019/5/8 
@Time: 13:42
@Desc:
'''
import os
import sys
sys.path.append('./')
import time
from multiprocessing import Manager, Lock
import multiprocessing.dummy as multiprocessing

import numpy as np
import cv2
from PIL import Image

from sdpc_python import sdpc


def Get_predictimgMultiprocess(startpointlist, ors, level, sizepatch, lock, saveimg=False, pathsave=None):
    """ 添加参数lock，保证多线程读图
    """
    start = time.time()
    pool = multiprocessing.Pool(16)
    imgTotals = []
    manager = Manager()
    lock = manager.Lock()
    for k in range(len(startpointlist)):
        imgTotal = pool.apply_async(GetImg_whole, args=(startpointlist, k, ors, level, sizepatch, lock, saveimg, pathsave))
        imgTotals.append(imgTotal)
    pool.close()
    pool.join()
    imgTotals = np.array([x.get() for x in imgTotals])
    end = time.time()
    print('多线程读图耗时{}\n{}张\nlevel = {}读入大小{} resize到{}'.format(end-start,len(startpointlist),level,sizepatch, sizepatch_predict))
    return imgTotals


def GetImg_whole_sdpc(startpointlist, k, ors, level, sizepatch, lock, saveimg=False, pathsave=None):
    """ 返回RGB通道 sizepatch大小图片
    """
    lock.acquire()
    img = ors.getTile(level, startpointlist[k][1], startpointlist[k][0], int(sizepatch[0]), int(sizepatch[1]))
    lock.release()

    img = np.ctypeslib.as_array(img)
    img.dtype = np.uint8
    img = img.reshape((int(sizepatch[1]), int(sizepatch[0]), 3))

    if saveimg:
        cv2.imwrite(pathsave + '{}_{}.tif'.format(k, startpointlist[k]), img)
    return img


def Get_startpointlist_sdpc(ors, level, levelratio, sizepatch, widthOverlap, flag, threColor = 8):
    sizepatch_y, sizepatch_x = sizepatch
    widthOverlap_y, widthOverlap_x = widthOverlap
    ratio = levelratio ** level
    if flag =='0':
        imgmap = EstimateRegion(ors, threColor=threColor)
        start_xmin = imgmap.nonzero()[0][0] * 32
        end_xmax = imgmap.nonzero()[0][-1] * 32
        start_ymin = np.min(imgmap.nonzero()[1]) * 32
        end_ymax = np.max(imgmap.nonzero()[1]) * 32
    elif flag == '1':
        """仅获取尺寸方式发生改变
        """
        attr = ors.getAttrs()
        size = (attr['width'], attr['height'])

        # size = ors.level_dimensions[level]
        start_xmin = 0
        end_xmax = size[1]
        start_ymin = 0
        end_ymax = size[0]
    numblock_x = np.ceil(((end_xmax-start_xmin)/ratio - sizepatch_x) / (sizepatch_x - widthOverlap_x)+1)
    numblock_y = np.ceil(((end_ymax-start_ymin)/ratio - sizepatch_y) / (sizepatch_y - widthOverlap_y)+1)
    # 得到所有开始点的坐标(排除白色起始点）
    startpointlist = []
    for j in range(int(numblock_x)):
        for i in range(int(numblock_y)):
            startpointtemp = (start_ymin + i*(sizepatch_y-widthOverlap_y)*ratio,start_xmin+j*(sizepatch_x-widthOverlap_x)*ratio)
            startpointlist.append(startpointtemp)
    return startpointlist


def main():
    pathsvs = 'H:\\CTDATA\\SZSQ_originaldata\\Shengfuyou_3th\\positive\\Shengfuyou_3th_positive_40X\\1100037 0893106.sdpc'

    # 保证多线程读图
    manager = Manager()
    lock = manager.Lock()

    # 创建sdpc读图对象
    ors = sdpc.Sdpc()
    ors.open(pathsvs)

    # 读大图
    startpointlist = Get_startpointlist_sdpc(ors, level, levelratio, sizepatch, widthOverlap, flag)
    print('Read in %s' % pathsvs)
    imgTotal = Get_predictimgMultiprocess_sdpc(startpointlist, ors, level, sizepatch, lock=lock)
