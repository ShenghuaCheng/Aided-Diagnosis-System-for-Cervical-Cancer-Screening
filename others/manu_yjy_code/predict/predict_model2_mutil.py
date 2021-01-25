# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 00:43:20 2018
@author: yujingya
"""
import sys 
sys.path.append('../')
from utils.model_set import ResNet_predict_multi as Model2_multi
from utils.function_set import *
from utils.parameter_set import Size_set,TCT_set,test_sample
import os
from keras.utils import multi_gpu_model
import tensorflow as tf
import time
import numpy as np
import cv2
from matplotlib import pyplot as plt
def Predict_whole_slide_model2():
    pathfile = pathfolder + filename + size_set['filetype']
    level = size_set['level']
    levelratio = size_set['levelratio']
    widthOverlap = size_set['widthOverlap']
    sizepatch_read = size_set['sizepatch_read']
    sizepatch_predict_model1 = size_set['sizepatch_predict_model1']
    sizepatch_predict_model2 = size_set['sizepatch_predict_model2']
    sizepatch_predict_small1 = size_set['sizepatch_predict_small1']
    sizepatch_predict_small2 = size_set['sizepatch_predict_small2']
    widthOverlap_predict = size_set['widthOverlap_predict']
    # 读大图
    try:
        startpointlist = Get_startpointlist(pathfile, level, levelratio, sizepatch_read, widthOverlap)
    except:
        startpointlist = Get_startpointlist(pathfile, level, levelratio, sizepatch_read, widthOverlap, threColor = 1)
    imgTotal = Get_predictimgMultiprocess(pathfile, startpointlist, level, sizepatch_read,pathsave_img)
    imgTotal_model2 = [cv2.resize(img,sizepatch_predict_model2) for img in imgTotal]
    imgTotal_model2 = np.array(imgTotal_model2)
    # 保存结果
    startpointlist_split = np.load(pathsave1 + 'startpointlist_split.npy')
    startpointlist = np.load(pathsave1 + filename + '_s.npy')
    predict1 = np.load(pathsave1 + filename + '_p.npy')
    feature = np.load(pathsave1 + filename + '_f.npy')
    num = len(startpointlist_split)
    #####################
    #  Model2  #####################
    if online:
        Index_max = [i*num+np.argmax(predict1[i*num:(i+1)*num]) for i in range(len(startpointlist))]
    else:
        Index_max = []
    startpointlist_2 = []
    imgTotal2 = []
    dictionary = []
    for index, item in enumerate(feature):
        #if predict1[index] > 0.5 or index in Index_max:
        Local_3 = region_proposal(item[...,1], sizepatch_predict_small1, img=None, threshold=0.7)
        dictionary.append(len(Local_3))
        for local in Local_3:
            ratio_temp = input_str['model2_resolution']/input_str['model1_resolution']
            startpointlist_2_w = int(startpointlist_split[index%15][0] + local[0]- (sizepatch_predict_small2[0]*ratio_temp)/2)
            startpointlist_2_h = int(startpointlist_split[index%15][1] + local[1]- (sizepatch_predict_small2[1]*ratio_temp)/2)
            startpointlist_2.append((startpointlist_2_w,startpointlist_2_h))
            startpointlist_2_w = int(startpointlist_2_w/ratio_temp)
            startpointlist_2_h = int(startpointlist_2_h/ratio_temp)
            img = imgTotal_model2[index//15][max(startpointlist_2_h,0):min(startpointlist_2_h+sizepatch_predict_small2[1],sizepatch_predict_model2[1]),
                          max(startpointlist_2_w,0):min(startpointlist_2_w+sizepatch_predict_small2[0],sizepatch_predict_model2[0])]
            if img.shape[0:2]!= sizepatch_predict_small2[::-1]:
                img_temp = np.zeros((sizepatch_predict_small2[1],sizepatch_predict_small2[0],3),np.uint8)
                h_temp = max(-startpointlist_2_h,0)
                w_temp = max(-startpointlist_2_w,0)
                img_temp[h_temp:h_temp+img.shape[0],w_temp:w_temp+img.shape[1]] = img
                img = img_temp
            imgTotal2.append(img)
            
    if sum(dictionary)!=0:        
        imgTotal2 = np.array(imgTotal2)
        imgTotal2 = trans_Normalization(imgTotal2, channal_trans_flag = True)
        predict2 = model2.predict(imgTotal2,batch_size = 16*gpu_num,verbose=1)  
    # 保存model2预测结果
    else :
        predict2 = []
    np.save(pathsave2 + filename + '_s2.npy',startpointlist_2)
    np.save(pathsave2 + filename + '_p2.npy',predict2)
    np.save(pathsave2 + filename + '_dictionary.npy',dictionary)
    #####################
    return None

if __name__ == '__main__':
    # 列 行
    input_str = {'filetype': '.svs',# .svs .sdpc .mrxs
                 'level': 0,
                 'read_size_model1_resolution': (1216, 1936), #1936 1216 横 model1_resolution下
                 'scan_over': (120, 120),
                 'model1_input': (512, 512, 3),
                 'model2_input': (256, 256, 3),
                 'model1_resolution':0.586,
                 'model2_resolution':0.293,
                 'input_resolution':0.293,#0.179
                 'cut':'3*5'}
    online = True
    pathsave_img = None
    device_id = '5'
    os.environ['CUDA_VISIBLE_DEVICES'] = device_id
    gpu_num = len(device_id.split(','))
    model2 = Model2_multi(input_shape = input_str['model2_input'])
    model2.load_weights('H:/weights/sdpc_model2_483_pred_multi.h5')
    
    size_set = Size_set(input_str)
    pathfolder_set, _ = TCT_set()
    for pathfolder in pathfolder_set['20x'][0:1]:
        break
        pathsave1 = 'F:/recom/model2_core_new_v5/ShengFY-P-L240 (origin date)/'
        if not os.path.exists(pathsave1):
            os.makedirs(pathsave1) 
        pathsave2 = pathsave1+pathfolder[24:] + 'model2_multi/'
        if not os.path.exists(pathsave2):
            os.makedirs(pathsave2)   
        CZ = os.listdir(pathsave1)
        CZ = [cz[:-len('_p.npy')] for cz in CZ if '_p.npy' in cz ]
#        CZ = [cz for cz in CZ if cz in test_sample['test_p_our_sfy3']]
        for filename in CZ:
#            break
            print(pathsave1)
            Predict_whole_slide_model2()
           
                
            
