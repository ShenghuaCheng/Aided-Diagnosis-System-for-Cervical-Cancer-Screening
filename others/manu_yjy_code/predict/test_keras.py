# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:24:02 2019

@author: A-WIN10
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


input_str = {'flag':'3d',
             'level' :0,
             'read_size':(1216,1936),
             'model1_input':(512,512,3),
             'model2_input':(256,256,3),
             'scan_over':(120,120),
             'ratio':1,
             'cut':'3*5_adapt_shu'}
   
device_id = '6'
os.environ['CUDA_VISIBLE_DEVICES'] = device_id
gpu_num = len(device_id.split(','))
model1 = Model1_LP(input_shape = input_str['model1_input'])
model1.load_weights('../weights/model1_340_3d1_3d2_our_10x_adapt.h5',by_name=True)
model1.load_weights('../weights/model1_340417_3d1_3d2_our_10x_adapt.h5',by_name=True)
model2 = Model2_2(input_shape = input_str['model2_input'])
model2.load_weights('../weights/model2_800_3d1_3d2_our_10x_adapt_subsup.h5')

size_set = Size_set(input_str)
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
pathfolder_svs_set, pathsave_set, _ = TCT_set()
num,startpointlist_split = Startpointlist_split(sizepatch_predict, sizepatch_predict_small1, widthOverlap_predict)

path = 'Z:/yujingya/10_images/'
pathsave = 'Z:/yujingya/10_images/keras/'
CZ = os.listdir(path)
CZ =[cz for cz in CZ if '.tif' in cz]
for cz in CZ:
    print(cz)
    imgTotal = cv2.imread(path +cz)
    # model1 预测
    imgTotal1 = Split_into_small([imgTotal],startpointlist_split,sizepatch_predict_small1)
    imgTotal1_predict = trans_Normalization(imgTotal1,channal_trans_flag= False)
    # 预测定位
    imgMTotal1 = model1.predict(imgTotal1_predict, batch_size = 32*gpu_num, verbose=1)
    predict1 = imgMTotal1[0].copy()
    feature = imgMTotal1[1].copy()
    for index, item in enumerate(feature):
        cv2.imwrite(pathsave + 'model1/' + cz[:-4] + '_{}_{}.tif'.format(index,predict1[index]),imgTotal1[index])
        if predict1[index] > 0.5:
            Local_3 = region_proposal(item[...,1], sizepatch_predict_small1, img=None, threshold=0.7)
            for indexx,local in enumerate(Local_3):
                startpointlist_2_x = startpointlist_split[index%15][1] + local[1]-64
                startpointlist_2_y = startpointlist_split[index%15][0] + local[0]-64
                img = imgTotal[max(startpointlist_2_x,0):min(startpointlist_2_x+128,sizepatch[1]),max(startpointlist_2_y,0):min(startpointlist_2_y+128,sizepatch[0])]
                if img.shape[0:2]!= sizepatch_small2[::-1]:
                    img_temp = np.zeros((sizepatch_small2[1],sizepatch_small2[0],3),np.uint8)
                    x_temp = max(-startpointlist_2_x,0)
                    y_temp = max(-startpointlist_2_y,0)
                    img_temp[x_temp:x_temp+img.shape[0],y_temp:y_temp+img.shape[1]] = img
                    img = img_temp
                imgTotal2 = cv2.resize(img, sizepatch_predict_small2)
                imgTotal2 = trans_Normalization([imgTotal2], channal_trans_flag = False)
                predict2 = model2.predict(imgTotal2,batch_size = 1,verbose=1)        
                cv2.imwrite(pathsave + 'model2/' + cz[:-4] + '_{}_{}_{}.tif'.format(index,indexx,predict2),img)
                
        









      
# 裁图
# =============================================================================
# pathsvs_folder = 'H:/TCTDATA/our/10x/Positive/2018/'
# pathsave = 'F:/yujingya/keras_10_3872_2432/big_adapt/5_positive/'
# CZ = os.listdir(pathsvs_folder)
# CZ =[cz for cz in CZ if '.svs' in cz]
# for cz in CZ:
#     pathsvs = pathsvs_folder + cz 
#     ors = OpenSlide(pathsvs)
#     img = ors.read_region((7000,7000),0, (sizepatch))
#     img = np.array(img)#RGBA
#     img = img[:, :, 0 : 3][...,::-1]
#     cv2.imwrite(pathsave + cz[:-4]+'_7000'+'.tif',img)
# =============================================================================
    
# 裁小图预测保存
