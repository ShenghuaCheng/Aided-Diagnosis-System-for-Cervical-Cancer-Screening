# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 00:43:20 2018
@author: YuJingya
"""
import sys, time
from tqdm import tqdm
sys.path.append('../')
from utils.function_set import *
from utils.parameter_set import Size_set,TCT_set
from keras.utils import multi_gpu_model
import os
import tensorflow as tf
import time
import numpy as np
import cv2
from openslide import OpenSlide
from matplotlib import pyplot as plt
from utils.Recommend_class import Recommend
from coreSegmentation.model import unet
from coreSegmentation.postProcess import postProcess

if __name__ == '__main__':
    # 1.参数设定
    input_str = {'filetype': '.sdpc',# .svs .sdpc .mrxs
                 'level': 0,
                 'read_size_model1_resolution': (1216, 1936), #1936 1216 横 model1_resolution下
                 'scan_over': (120, 120),
                 'model1_input': (512, 512, 3),
                 'model2_input': (256, 256, 3),
                 'model1_resolution':0.586,
                 'model2_resolution':0.293,
                 'input_resolution':0.18,#0.18
                 'cut':'3*5'}
    os.environ['CUDA_VISIBLE_DEVICES']='3'
    input_size=(256,256,3)
    gpu_num = 1                                                                 
    batch_size = 16                                                  
    weight_path = '../weights/unet_core_block320.h5' 
    #加载模型
    if gpu_num > 1:
        with tf.device('/cpu:0'):
            model = unet(input_size=input_size)
            if weight_path is not None:
                print('Load weights %s' % weight_path)
                model.load_weights(weight_path)
        parallel_model = multi_gpu_model(model, gpus=gpu_num)
    else:
        parallel_model = unet(input_size=input_size)
        if weight_path is not None:
            print('Load weights %s' % weight_path)
            parallel_model.load_weights(weight_path)


    size_set = Size_set(input_str)
    pathfolder_set, _ = TCT_set()
    for pathfolder in pathfolder_set['szsq_sdpc'][-1:]:
#        break
        pathsave1 = 'F:/recom/model1_szsq615_model2_szsq1050/'+pathfolder[11:]
        pathsave2 = pathsave1 + 'model2/'
        pathsave2_core = pathsave1 + 'model2_core/'
        while not os.path.exists(pathsave2_core):
            os.makedirs(pathsave2_core)
            
        CZ = os.listdir(pathsave2)
        CZ = [cz[:-len('_p2.npy')] for cz in CZ if '_p2.npy' in cz]
        for filename in tqdm(CZ[40:60]):
            break
#            filename=CZ[1]
            pathsave_xml =None
            num_recom = 20
            object_recommend = Recommend(pathsave1, pathsave_xml,
                                         filename, input_str, num_recom,
                                         pathsave2, pathsave2_core)
            imgTotal2_core,dictionary = object_recommend.get_imgTotal2_core(pathfolder)#RGB
            
            
            np.save(pathsave2_core + filename + '_imgTotal2_core.npy',np.array(imgTotal2_core))
            np.save(pathsave2_core + filename + '_dictionary_core.npy',np.array(dictionary))
            
            imgTotal2_core = np.float32(imgTotal2_core) / 255.
            imgTotal2_core = imgTotal2_core - 0.5
            imgTotal2_core = imgTotal2_core * 2.
            
            #imgTotal2_core = np.load(pathsave2_core + filename + '_imgTotal2_core.npy')
            result = parallel_model.predict(imgTotal2_core, batch_size=batch_size*gpu_num, verbose=1)
            postProcess1 = postProcess()
            allContours, allAreas = postProcess1.postprocess(result)
            np.save(pathsave2_core + filename + '_predict_core.npy',np.array([allContours, allAreas]))
           
# =============================================================================
#             a = cv2.drawContours(imgTotal2_core[120],allContours[120],-1,(0,255,0),3)
#             plt.imshow(a)
# =============================================================================
            
# =============================================================================
#             imgTotal2_morphology = [img[192:192+128,192:192+128,:] for img in imgTotal2_core]
#             imgTotal2_morphology = np.array(imgTotal2_morphology)
#             imgTotal2_morphology = trans_Normalization(imgTotal2_morphology, channal_trans_flag = True)
#             imgMTotal2_morphology = model_class_7.predict(imgTotal2_morphology, batch_size = 32*gpu_num, verbose=1)
#             imgMTotal2_morphology = [np.argmax(item) for item in imgMTotal2_morphology]
# =============================================================================
#            object_recommend.recommend_region_combine_core(imgTotal2_core, predict2_core, imgMTotal2_morphology,num_recom)
    
               
