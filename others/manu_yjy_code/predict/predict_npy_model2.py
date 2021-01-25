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
from matplotlib import pyplot as plt
from tqdm import tqdm

def Predict_whole_slide_model2():
    predict1 = np.load(pathsave1 + filename + '_p.npy')
    imgTotal2 = np.load(pathsave2_core + filename + '_imgTotal2_core.npy')
    dictionary =  np.load(pathsave2_core + filename + '_dictionary_core.npy')
    
    imgTotal2 = imgTotal2[...,::-1]#BGR
    imgTotal2 = np.float32(imgTotal2) / 255.
    imgTotal2 = imgTotal2 - 0.5
    imgTotal2 = imgTotal2 * 2.
    
    predict2 = model2.predict(imgTotal2,batch_size = 16*gpu_num,verbose=1)  
    
    np.save(pathsave2 + filename + '_p2.npy',predict2)
    np.save(pathsave2 + filename + '_dictionary.npy',dictionary)
    
    predict12 = []
    for index, item in enumerate(dictionary):
        if item!=0:
            a = predict2[int(np.sum(dictionary[:index])):int(np.sum(dictionary[:index])+item)]
            a = a.max()
            predict12.append(a)
        else:
            predict12.append(predict1[index])
    predict12 = np.vstack(predict12)
    np.save(pathsave2 + filename + '_p12.npy',predict12)
    #####################
    return None

if __name__ == '__main__':
    # 列 行
    device_id = '7'
    os.environ['CUDA_VISIBLE_DEVICES'] = device_id
    gpu_num = len(device_id.split(','))
    model2 = Model2_2((256,256,3))
    model2.load_weights(r'H:\weights\w_sdpc\model2\stage_new_class\szsq_Block_1625_szsq.h5')
    
    pathfolder_set, _ = TCT_set()
    for pathfolder in pathfolder_set['szsq_sdpc']:
#        break
        pathsave1 = 'F:/recom/model1_szsq615_model2_szsq1050/'+pathfolder[11:]
        pathsave2_core = pathsave1 + 'model2_core/'
        pathsave2 = pathsave1 + 'model2_szsq_1625_top500/'
        while not os.path.exists(pathsave2):
            os.makedirs(pathsave2)
        CZ = os.listdir(pathsave2_core)
        CZ = [cz[:-len('_imgTotal2_core.npy')] for cz in CZ if '_imgTotal2_core.npy' in cz]
        for filename in tqdm(CZ):
#            break
            Predict_whole_slide_model2()
           
                
            
