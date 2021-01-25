# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 00:43:20 2018
@author: YuJingya
"""

import sys, time
from tqdm import tqdm
sys.path.append('../')
from utils.function_set import trans_Normalization
import nucluesSegmentation.model as modellib 
import nucluesSegmentation.cell as cell
from utils.parameter_set import Size_set,TCT_set,test_sample
import os
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
from keras.utils import multi_gpu_model
import tensorflow as tf
import time
import numpy as np
import cv2
from openslide import OpenSlide
from matplotlib import pyplot as plt
from utils.Recommend_class import Recommend
from utils.Mobilenet import MobileNet_MC
def Predict_whole_slide_model2_core():
    size_crop = 512
    if len(imgTotal2_core)!=0:  
        #补全
        Completion_batch = batch_size - len(imgTotal2_core)%batch_size
        Completion_img = np.zeros((Completion_batch,size_crop,size_crop,3),np.uint8)
        imgTotal2_core_temp = np.concatenate((imgTotal2_core,Completion_img),axis=0)
        #预测
        num_batch = len(imgTotal2_core_temp)//batch_size
        predict2_core = []
        for i in tqdm(range(num_batch)):
            predict2_core += model_core_seg.detect(imgTotal2_core_temp[i*batch_size:(i+1)*batch_size],verbose = 0)#rgb 原图
        predict2_core = predict2_core[:-Completion_batch]   
    else :
        predict2_core = []
    return predict2_core

if __name__ == '__main__':
    # 1.参数设定
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
    
    device_id = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = device_id
    gpu_num = len(device_id.split(','))
    #******create model in inference mode*****
    batch_size = 32
    class InferenceConfig(cell.CellConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = batch_size
    config = InferenceConfig()
    model_core_seg = modellib.MaskRCNN(mode ="inference",
                              config=config,
                              model_dir=MODEL_DIR)
    model_path = '../weights/mask_rcnn_kernelhunter_0010.h5'
    model_core_seg .load_weights(model_path, by_name=True)
    #形态学model2模型load
    model_path = '../weights/morphology_classify_7_320.h5'
    model_class_7 = MobileNet_MC(input_shape = (128,128,3),alpha=0.5)
    model_class_7.load_weights(model_path)
    
    num_recom = 22
    size_set = Size_set(input_str)
    pathfolder_set, _ = TCT_set()
    for pathfolder in pathfolder_set['20x'][0:1]:
        pathsave1 = 'Y:/recom/model2_core_new_v5/ShengFY-P-L240 (origin date)/'
        pathsave2 = pathsave1 + 'model2/'
        pathsave2_core = pathsave1 + 'model2_core/'
        pathsave_xml = pathsave1 + 'xml/'
        while not os.path.exists(pathsave2_core):
            os.makedirs(pathsave2_core)
        while not os.path.exists(pathsave_xml):
            os.makedirs(pathsave_xml)
        CZ = os.listdir(pathsave2)
        
        CZ = [cz[:-len('_p2.npy')] for cz in CZ if '_p2.npy' in cz]
        CZ = [cz for cz in CZ if  cz in test_sample['test_p_our_sfy3']]
       
        for filename in tqdm(CZ):
            object_recommend = Recommend(pathsave1, pathsave_xml,
                                         filename, input_str, num_recom,
                                         pathsave2, pathsave2_core)
            # 1.读图
            # 1.1为细胞核分割读入图片maskrcnn 图片大小：level1 512*512
            imgTotal2_core = object_recommend.get_imgTotal2_nuclues(pathfolder)
            # 1.2为形态学分类模型准备图片 图片大小：level1 128*128
            imgTotal2_morphology = [img[192:192+128,192:192+128,:] for img in imgTotal2_core]
            imgTotal2_morphology = np.array(imgTotal2_morphology)
            imgTotal2_morphology = trans_Normalization(imgTotal2_morphology, channal_trans_flag = True)
            # 2.预测
            # 2.1 核分割结果预测
            predict2_core = Predict_whole_slide_model2_core()
            # 2.2 形态学模型预测
            imgMTotal2_morphology = model_class_7.predict(imgTotal2_morphology, batch_size = 32*gpu_num, verbose=1)
            imgMTotal2_morphology = [np.argmax(item) for item in imgMTotal2_morphology]
            # 3.保存图片核分割图片和核分割结果
            np.save(pathsave2_core + filename + '_imgTotal2_core.npy',np.array(imgTotal2_core))    
            np.save(pathsave2_core + filename + '_p2_core.npy',predict2_core)
            # 4.综合核分割结果 形态学分类结果 预测
            object_recommend.recommend_region_combine_core(imgTotal2_core, predict2_core, imgMTotal2_morphology,num_recom)
