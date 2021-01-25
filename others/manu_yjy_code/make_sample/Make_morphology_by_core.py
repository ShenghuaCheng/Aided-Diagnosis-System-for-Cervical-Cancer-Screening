# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 21:02:41 2019

@author: YuJingya
"""
import sys
sys.path.append('../')
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import multiprocessing.dummy as multiprocessing
from tqdm import tqdm 
from utils.parameter_set import Size_set,TCT_set
from utils.nucleus_analysis import cal_focus_nucleus_feature_v3 as cal_nucleus_v3
from utils.function_set import Get_predictimgMultiprocess,region_proposal
import nucluesSegmentation.model as modellib 
import nucluesSegmentation.cell as cell
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
from openslide import OpenSlide

def Save_morpholigy_class(imgTotal2_core,startpointlist_core,Type_img,pathsave_sample,index):
     pathsave = pathsave_sample + Type_img[index]
     while not os.path.exists(pathsave):
        os.makedirs(pathsave)  
     cv2.imwrite(pathsave+ '/{}_{}_{}_{}_{}.tif'.format(pathsave1.split('/')[-2],filename,index,startpointlist_core[index][0],startpointlist_core[index][1]) ,imgTotal2_core[index][...,::-1])
     # 计算model2定位坐标起始点并读图
def Get_imgTotal2_core():
    pathsvs = pathfolder_svs + filename + filetype
    print(pathfolder_svs + filename + filetype)
    ors = OpenSlide(pathsvs)
    startpointlist0 = np.load(pathsave1 + filename + '_s.npy')
    predict1 = np.load(pathsave1 + filename + '_p.npy')
    feature = np.load(pathsave1 + filename + '_f.npy') 
    startpointlist_split = np.load(pathsave1 + 'startpointlist_split.npy') 
    imgTotal = Get_predictimgMultiprocess(startpointlist0, ors, level, sizepatch, sizepatch_predict)
    ratio = input_str['ratio']
    num = len(startpointlist_split)
    startpointlist_core = []
    imgTotal2_core = []
    thero_200 = predict1[np.argsort(predict1,axis=0)[-200][0]][0]
    for index, item in enumerate(feature):
        if predict1[index] > max(0.5,thero_200):
            Local_3 = region_proposal(item[...,1], sizepatch_predict_small1, img=None, threshold=0.7)
            for local in Local_3:
                startpointlist_2_x = startpointlist_split[index%num][1] + local[1]- int(size_crop/2)
                startpointlist_2_y = startpointlist_split[index%num][0] + local[0]- int(size_crop/2)
                img = imgTotal[index//num][max(startpointlist_2_x,0):min(startpointlist_2_x+size_crop,sizepatch[1]),max(startpointlist_2_y,0):min(startpointlist_2_y+size_crop,sizepatch[0])]
                if img.shape[0:2]!= (size_crop,size_crop):
                    img_temp = np.zeros((size_crop,size_crop,3),np.uint8)
                    x_temp = max(-startpointlist_2_x,0)
                    y_temp = max(-startpointlist_2_y,0)
                    img_temp[x_temp:x_temp+img.shape[0],y_temp:y_temp+img.shape[1]] = img
                    img = img_temp
                imgTotal2_core.append(img)
                points = (startpointlist0[index//15][0]+int(startpointlist_2_y*ratio),
                          startpointlist0[index//15][1]+int(startpointlist_2_x*ratio))
                startpointlist_core.append(points)
    return imgTotal2_core,startpointlist_core
       
 #预测        
def Make_sample_for_morphology(imgTotal2_core,startpointlist_core):
    Completion_batch = batch_size - len(imgTotal2_core)%batch_size
    Completion_img = np.zeros((Completion_batch,size_crop,size_crop,3),np.uint8)
    imgTotal2_core = np.concatenate((imgTotal2_core,Completion_img),axis=0)
    num_batch = len(imgTotal2_core)//batch_size
    predict2_core = []
    for i in tqdm(range(num_batch)):
        predict2_core += model_core_seg.detect(imgTotal2_core[i*batch_size:(i+1)*batch_size],verbose = 0)#rgb 原图
    imgTotal2_core = imgTotal2_core[:-Completion_batch]
    predict2_core = predict2_core[:-Completion_batch]
    #预测
    Areas, Type_img, Contours_core_relat = cal_nucleus_v3(imgTotal2_core,predict2_core)
    pool = multiprocessing.Pool(20)
    for index,item in enumerate(imgTotal2_core):
        pool.apply_async(Save_morpholigy_class,
                         args=(imgTotal2_core,startpointlist_core,Type_img,pathsave_sample,index))
    pool.close()
    pool.join()
    return None

if __name__ == '__main__':
    # 1.参数设定
    
    device_id = '6'
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
    model_path ='../weights/mask_rcnn_kernelhunter_0010.h5'
    model_core_seg .load_weights(model_path, by_name=True)
    
    input_str = {'flag':'our',
                 'level' : 0,
                 'read_size':(1216,1936),#1936,1216heng
                 'model1_input':(512,512,3),
                 'model2_input':(256,256,3),
                 'scan_over':(120,120),
                 'ratio':1,
                 'cut':'3*5_adapt_shu'}
    ##
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
    for pathfolder_svs,pathsave1 in zip(pathfolder_svs_set['10x'][4:],pathsave_set['10x'][4:]):
        pathsave2_core = pathsave1 + 'model2_core/'
        pathsave_sample = 'H:/morphology/morphology_by_core/train/our_10x_'
        CZ = os.listdir(pathsave1)
        CZ = [cz[:-len('_p.npy')] for cz in CZ if '_p.npy' in cz]
        for filename in tqdm(CZ):
            size_crop = 512
            imgTotal2_core,startpointlist_core = Get_imgTotal2_core()
            Make_sample_for_morphology(imgTotal2_core,startpointlist_core)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
           
   