# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 23:24:06 2018

@author: yujingya
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
import ResNetClassification as RNC
from keras.utils import multi_gpu_model
import tensorflow as tf
import time
import numpy as np
import cv2
from openslide import OpenSlide
from predict_310_function import *

pathweight = 'H:/weights/model1_135_3d1_3d2_our.h5'
# 建立并行预测模型
gpu_num = 2
with tf.device('/cpu:0'):
    model = RNC.ResNet(input_shape=(512, 512, 3))
    model.load_weights(pathweight)
parallel_model = multi_gpu_model(model, gpus=gpu_num)

sizepatch_predict = (int(1216*1.2),int(1936*1.2))#1459 2323
sizepatch_predict_small = (512,512)
widthOverlap_predict = (40,60)#3*5
#widthOverlap_predict = (197,150) #4*6
# 获取操作对象列表
pathtest = 'F:/yujingya/100_test/50_negative/'
filename = os.listdir(pathtest)
filename = [cz  for cz in filename if '.tif' in cz]
imgTotal = [] 
for file in filename:
    img = cv2.imread(pathtest + file)
    img = cv2.resize(img,sizepatch_predict)
    imgTotal.append(img)
    
#将拍摄小图拆分成适合网络大小的小图    
imgTotal,num,startpointlist_split = Split_into_small(imgTotal,sizepatch_predict,sizepatch_predict_small, widthOverlap_predict)
imgTotal = np.float32(imgTotal) / 255.
imgTotal = imgTotal - 0.5
imgTotal = imgTotal * 2. 

imgMTotal = parallel_model.predict(imgTotal, batch_size = 32*gpu_num, verbose=1)
#max
imgMTotal_combine = [max(imgMTotal[i*num:(i+1)*num])for i in range(50)]       
imgMTotal_combine  = np.array(imgMTotal_combine)
#111
import heapq 
imgMTotal_combine = []
for i in range(50):
    score = heapq.nlargest(3,imgMTotal[i*num:(i+1)*num]) 
    score1,score2,score3 = np.sort(score)
    if (score1-score2)>0.5:
        imgMTotal_combine.append(score1[0])
    elif (score1-score3)>0.5:
        imgMTotal_combine.append(np.average([score1,score2]))
    else :
        imgMTotal_combine.append(np.average(score))  
imgMTotal_combine  = np.array(imgMTotal_combine)
#411
imgMTotal_combine = []
for i in range(50):
    score = heapq.nlargest(3,imgMTotal[i*num:(i+1)*num]) 
    score1,score2,score3 = np.sort(score)
    imgMTotal_combine.append((4*score1+score2+score3)/6)  
imgMTotal_combine  = np.array(imgMTotal_combine)