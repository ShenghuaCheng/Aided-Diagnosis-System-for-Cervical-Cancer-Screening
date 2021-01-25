# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 00:43:20 2018
@author: yujingya
"""
import sys 
sys.path.append('../')
from utils.model_set import Resnet_location_and_predict as Model1_LP
from utils.model_set import Resnet_location as Model1_L
from utils.model_set import ResNet_predict as Model1_P
from utils.model_set import ResNet_predict as Model2_2
from utils.model_set import ResNet_predict_multi as Model2_multi
from utils.function_set import *
from utils.parameter_set import Size_set,TCT_set
import os
from keras.utils import multi_gpu_model
import tensorflow as tf
import time
import numpy as np
import cv2
from openslide import OpenSlide
from matplotlib import pyplot as plt


def Predict_whole_slide_model1(pathfolder_svs, pathsave, filename, filetype):
    pathsvs = 'Z:/yujingya/test_whole/1100037.svs'
    ors = OpenSlide(pathsvs)
    # 读大图
    startpointlist = Get_startpointlist(ors, level, levelratio, sizepatch, widthOverlap,flag)
    
    imgTotal = Get_predictimgMultiprocess(startpointlist, ors, level, sizepatch, sizepatch_predict)
    
    for index,img in enumerate(imgTotal):
        cv2.imwrite('Z:/yujingya/test_whole/1100037/%s.tif'%index,img[...,::-1])
    # 裁小图
    imgTotal1 = Split_into_small(imgTotal,startpointlist_split, sizepatch_predict_small1)
    imgTotal1 = trans_Normalization(imgTotal1,channal_trans_flag= True)
    imgTotal = []
    # 预测定位
    imgMTotal1 = model1.predict(imgTotal1, batch_size = 32*gpu_num, verbose=1)
    predict1 = imgMTotal1[0].copy()
    feature = imgMTotal1[1].copy()
    # predict1[9140]== 0.00235909
    # 保存结果
    np.save('Z:/yujingya/test_whole/1100037_s.npy',startpointlist)
    np.save('Z:/yujingya/test_whole/1100037_p.npy',predict1)
    np.save('Z:/yujingya/test_whole/1100037_f.npy',feature) 
    return None

if __name__ == '__main__':
    input_str = {'flag':'our',
                 'level' :0,
                 'read_size':(1216,1936),
                 'model1_input':(512,512,3),
                 'model2_input':(256,256,3),
                 'scan_over':(120,120),
                 'ratio':1,
                 'cut':'3*5_adapt_shu'}
    
    device_id = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = device_id
    gpu_num = len(device_id.split(','))
    model1 = Model1_LP(input_shape = input_str['model1_input'])
    model1.load_weights('../weights/model1_340_3d1_3d2_our_10x_adapt.h5',by_name=True)
    model1.load_weights('../weights/model1_340417_3d1_3d2_our_10x_adapt.h5',by_name=True)
    
    size_set = Size_set(input_str)
    pathfolder_svs_set, pathsave_set, _ = TCT_set()
    sizepatch = size_set['sizepatch']
    widthOverlap = size_set['widthOverlap']
    sizepatch_small1 = size_set['sizepatch_small1']
    sizepatch_small2 = size_set['sizepatch_small2']
    sizepatch_predict = size_set['sizepatch_predict']
    sizepatch_predict_small1 = size_set['sizepatch_predict_small1']
    sizepatch_predict_small2 = size_set['sizepatch_predict_small2']
    widthOverlap_predict = size_set['widthOverlap_predict']
    flag = size_set['flag']
    filetype = size_set['filetype']
    level = size_set['level']
    levelratio = size_set['levelratio']
    
    num,startpointlist_split = Startpointlist_split(sizepatch_predict, sizepatch_predict_small1, widthOverlap_predict)
        
    for pathfolder_svs,pathsave in zip(pathfolder_svs_set['10x'][0:1],pathsave_set['10x'][0:1]):
        while not os.path.exists(pathsave):
            os.mkdir(pathsave)     
        CZ = os.listdir(pathfolder_svs)
        CZ = [cz[:-len(filetype)] for cz in CZ if filetype in cz]
        np.save(pathsave + 'startpointlist_split.npy',startpointlist_split)
        for filename in CZ[0:1]:
            print(pathfolder_svs + filename + filetype)
            print(pathsave)
            pathsave = 'Z:/yujingya/test_whole/'
            Predict_whole_slide_model1(pathfolder_svs, pathsave, filename, filetype)
