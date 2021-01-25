# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 00:43:20 2018
@author: yujingya
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import ResNetClassification as RNC
from keras.utils import multi_gpu_model
import tensorflow as tf
import time
import numpy as np
import multiprocessing.dummy as multiprocessing
import cv2
from openslide import OpenSlide
from matplotlib import pyplot as plt
from lxml import etree
def Get_startpointlist(ors, level, levelratio, sizepatch,widthOverlap):
    sizepatch_y,sizepatch_x = sizepatch
    widthOverlap_y,widthOverlap_x = widthOverlap
    ratio = levelratio ** level
    size = ors.level_dimensions[level]
    start_xmin = 0
    end_xmax = size[1]
    start_ymin = 0
    end_ymax = size[0]
    numblock_x = np.around(((end_xmax-start_xmin)/ratio - sizepatch_x) / (sizepatch_x - widthOverlap_x))
    numblock_y = np.around(((end_ymax-start_ymin)/ratio - sizepatch_y) / (sizepatch_y - widthOverlap_y))
    # 得到所有开始点的坐标(排除白色起始点）
    startpointlist = []
    for j in range(int(numblock_x)):
        for i in range(int(numblock_y)):
            startpointtemp = (start_ymin + i*(sizepatch_y-widthOverlap_y)*ratio,start_xmin+j*(sizepatch_x-widthOverlap_x)*ratio)
            startpointlist.append(startpointtemp)
    return startpointlist

def GetImg_color(startpointlist, k, ors, level, sizepatch, sizepatch_predict):
    img = ors.read_region(startpointlist[k], level, sizepatch)
    img = np.array(img)#RGBA
    img = img[:, :, 0 : 3]
    img = cv2.resize(img,sizepatch_predict)
    return img

def Get_predictimgMultiprocess(startpointlist, ors, level, sizepatch, sizepatch_predict):    
    # 多线程读图
    pool = multiprocessing.Pool(20)
    imgTotals = []
    for k in range(len(startpointlist)):
        imgTotal = pool.apply_async(GetImg_color,args=(startpointlist, k, ors, level, sizepatch, sizepatch_predict))
        imgTotals.append(imgTotal)
    pool.close()
    pool.join()
    imgTotal = np.array([x.get() for x in imgTotals])
    imgTotals = []
    return imgTotal

def Split_into_small(imgTotal,sizepatch_predict,sizepatch_predict_small, widthOverlap_predict):
    sizepatch_predict_y,sizepatch_predict_x = sizepatch_predict
    sizepatch_predict_small_y,sizepatch_predict_small_x = sizepatch_predict_small
    widthOverlap_predict_y,widthOverlap_predict_x = widthOverlap_predict
        
    numblock_y = np.around(( sizepatch_predict_y - sizepatch_predict_small_y) / (sizepatch_predict_small_y - widthOverlap_predict_y)+1)
    numblock_x = np.around(( sizepatch_predict_x - sizepatch_predict_small_x) / (sizepatch_predict_small_x - widthOverlap_predict_x)+1)
    startpointlist = []
    for j in range(int(numblock_x)):
        for i in range(int(numblock_y)):
            startpointtemp = (i*(sizepatch_predict_small_y- widthOverlap_predict_y),j*(sizepatch_predict_small_x-widthOverlap_predict_x))
            startpointlist.append(startpointtemp)
   
    imgTotals = []
    for k in range(len(imgTotal)):
        for i in range(len(startpointlist)):
            startpointtemp = startpointlist[i]
            startpointtempy,startpointtempx = startpointtemp
            img = imgTotal[k][startpointtempx:startpointtempx+512,startpointtempy:startpointtempy+512]
            imgTotals.append(img)
    imgTotals = np.array(imgTotals)
    return imgTotals,int(numblock_x*numblock_y),startpointlist


if __name__ == '__main__':
    # 参数设定
    level = 0
    levelratio = 4
    
    #20x
    sizepatch = (int(1216*1.2/0.6),int(1936*1.2/0.6))#2432 3872
    sizepatch_small = (int(512/0.6),int(512/0.6))
    widthOverlap = (int(197/0.6),int(60/0.6))
    #10x
    sizepatch_predict = (int(1216*1.2),int(1936*1.2))#1459 2323
    sizepatch_predict_small = (512,512)
    widthOverlap_predict = (197,60)
    
    num_recom = 16

    pathfolder_svs = 'H:/TCTDATA/our/Positive/Shengfuyou_4th/'
    pathfolder_xml = 'H:/recom_30/our/Positive/Shengfuyou_4th/310_test/w_1our_55/'
    pathweight = 'H:/weights/w_TestModel20190228/w_1our/Block_55.h5'
    # 建立并行预测模型
    gpu_num = 2
    with tf.device('/cpu:0'):
        model = RNC.ResNet(input_shape=(512, 512, 3))
        model.load_weights(pathweight)
    parallel_model = multi_gpu_model(model, gpus=gpu_num)
    # 获取操作对象列表
    CZ = os.listdir(pathfolder_svs)
    CZ = [cz for cz in CZ if '.svs' in cz]
    
    for cz in CZ:
        pathsvs = pathfolder_svs + cz
        print(pathsvs)
        ors = OpenSlide(pathsvs)

        startpointlist = Get_startpointlist(ors, level, levelratio, sizepatch, widthOverlap)
        imgTotal = Get_predictimgMultiprocess(startpointlist, ors, level, sizepatch, sizepatch_predict)

        imgTotal,num,startpointlist_split = Split_into_small(imgTotal,sizepatch_predict,sizepatch_predict_small, widthOverlap_predict)
        