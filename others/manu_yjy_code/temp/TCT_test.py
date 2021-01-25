# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 15:47:03 2018
@author: yujingya
"""
from __future__ import print_function
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import numpy as np
import random
from model_re_dep import Resnet_location_and_predict as RLP
from model_re_dep import Resnet_location as RL
from keras.utils import multi_gpu_model
import tensorflow as tf
import time
from glob import glob
import cv2
from matplotlib import pyplot as plt
from Readsample_re_dep import GetImgMultiprocess,trans_Normalization,region_proposal,IOU
#%%
path_mask = ["I:/BigData/localmask/"]# 
path_test = ["I:/BigData/test/"]#test train
classes = ["1_ASCUS","1_HL","2_ASCUS","2_HL", 
           "our_ASCUS","our_HL", "10x_our_ASCUS","10x_our_HL" ]#
vol_test = 1000
resize_shape = (2188,3484)#(768,768)
crop_shape = (2188,3484)#(768,768)
output_shape = (1216,1936)#(512,512)
enhance_flag = False
path_mask = path_mask
mask_shape = (38,61)#(16,16)
local= None
pathsave = 'F:/yujingya/code/re_dep2.0/result/adapt_big/test/'
threshold=0.15
#%%
# 建立并行训练模型
gpu_num = 1
model1 = RL(input_shape = (output_shape[0],output_shape[1],3))
#model1.load_weights('H:/weights/model1_340417_3d1_3d2_our_10x_adapt.h5',by_name=True)
model1.load_weights('F:/weights/model1_5_3d1_3d2_our_adapt_big.h5',by_name=True)
#%%
# 创建读图对象

for item1 in classes:
    filelist = glob(path_test[0] + item1 + '/*.tif')
    #%%
    # 随机选择图片
    dirList = random.sample(filelist,min(vol_test,len(filelist)))
    #%%
    # 读入图片和对应mask
    imgTotaltest,img_maskTotaltest,local = GetImgMultiprocess(dirList, resize_shape,crop_shape,output_shape,enhance_flag,path_mask, mask_shape,local=None)
    #%%
    # 归一化
    imgTotaltest = trans_Normalization(imgTotaltest,channal_trans_flag = False)
    #%%s
    # 训练
    imgMTotal = model1.predict(imgTotaltest,batch_size = 8*gpu_num,verbose=1)
    feature = imgMTotal.copy()
    #%%计算IOU并写入txt文档
    Iou_2big = IOU(img_maskTotaltest,feature,threshold)
    max_score = np.array([np.max(x[...,1])for x in feature])
    accuaracy = np.sum(max_score<=threshold)/len(max_score)
    #%%
    if 'ASCUS' in item1 or 'HL' in item1:
        accuaracy = 1-accuaracy
    with open(pathsave+'IOU.txt','a') as f:
        f.write(item1 +' ' + str(Iou_2big) +' ' + str(accuaracy)+'\n')
        f.write(item1 +' ' + str(Iou_2big) +'\n')
    
    #%%可视化结果
    try:
        os.mkdir(pathsave+ item1 )
    except:
        print('exist')
    for index, item2 in enumerate(imgMTotal):
        img_prop, _,_ = region_proposal(item2[...,1], output_shape,imgTotaltest[index], threshold,ratio=0.2)
        img_mask = cv2.resize(item2[...,1],img_prop.shape[:-1][::-1])
        img_mask = img_mask*255
        img_maskk = img_prop.copy()
        img_maskk[...,0]=img_maskk[...,1]=img_maskk[...,2]=img_mask
        img_save = np.concatenate((img_prop,img_maskk),axis = 1)
        cv2.imwrite(pathsave+ item1 + '/' +str(dirList[index].split('\\')[-1]),img_save)
