# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 15:47:03 2018
@author: yujingya
"""
from __future__ import print_function
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import random
from model_re_dep import Resnet_atrous as RA
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
import tensorflow as tf
import time

#%%  
path_train = ['H:/yujingya_data/train/']# 
path_mask = ['H:/yujingya_data/label_map/']# 
path_test = ["H:/yujingya_data/test/"]#
classes = ["1_ASCUS","1_HL","1_n","1_n_w","1_nplus","2_ASCUS","2_HL","2_n","2_n_w"]#
vol_train = np.uint16((np.array([1000,1000,970,50,970,1000,1000,970,50]))/2)
vol_test = np.array([50,50,45,10,45,50,50,45,10,45])
level = 1
#%%
# 建立并行训练模型
gpu_num = 1
#with tf.device('/cpu:0'):
model = RA(input_shape = (512,512,3))
model.compile(optimizer= Adam(lr=0.01), loss=["categorical_crossentropy"], metrics=["categorical_accuracy"])
model.load_weights('F:/weights/Genertate_mask/Epoch_15.h5',by_name=True)
#parallel_model = multi_gpu_model(model, gpus=gpu_num)
#parallel_model.compile(optimizer= Adam(lr=0.01), loss=["categorical_crossentropy"], metrics=["categorical_accuracy"])
#%%
# 创建读图对象
from Readsample_re_dep import Readsample_for_list as Read
from Readsample_re_dep import GetImgMultiprocess,trans_Normalization
obj_train =  Read(path_train,classes)
obj_train.Get_filelist()
obj_test =  Read(path_test,classes)
obj_test.Get_filelist()
#%%
# 外循环为训练block数
for i in range(16,1600):
    print('Block ' + str(i))  
    #%%
    start = time.time()
    print('loading 3d1 2 _samples')
    #%%
    # 随机选择图片
    dirlist_train = obj_train.Get_dirlist(vol_train)
    dirlist_test = obj_test.Get_dirlist(vol_test)
    #%%
    # 读入图片和对应mask
    imgTotaltrain, img_maskTotaltrain = GetImgMultiprocess(dirlist_train,level,'color','trans',path_mask)#2
    imgTotaltest, img_maskTotaltest = GetImgMultiprocess(dirlist_test,level,'color','un_trans',path_mask)#3
    #%%
    # 归一化
    imgTotaltrain = trans_Normalization(imgTotaltrain)
    imgTotaltest = trans_Normalization(imgTotaltest)
    end = time.time()
    print(end - start)
    #%%
    # 打乱顺序
    index = [i for i in range(len(imgTotaltrain))]
    random.shuffle(index)
    imgTotaltrain = imgTotaltrain[index]
    img_maskTotaltrain = img_maskTotaltrain[index]
    #%%
    # 训练
    hist = model.fit(imgTotaltrain,img_maskTotaltrain,batch_size = 8*gpu_num, epochs=1, verbose=1,  validation_data=(imgTotaltest,img_maskTotaltest))
    #%%
    #权重保存
    if i%5==0:
        model.save_weights('F:/weights/Genertate_mask/Epoch_%s.h5'%(i))
    with open('F:/weights/Genertate_mask/history.txt','a') as f:
        f.write('Block' + str(i) + '\n')
        f.write(str(hist.history)+'\n')