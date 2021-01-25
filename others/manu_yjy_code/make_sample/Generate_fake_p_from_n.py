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
from multiprocessing import Manager, Lock
from tqdm import tqdm 
from utils.parameter_set import Size_set,TCT_set,test_sample
from utils.function_set import Get_predictimgMultiprocess,region_proposal
import nucluesSegmentation.model as modellib 
import nucluesSegmentation.cell as cell
from openslide import OpenSlide
import random
from utils.sdpc_python import sdpc
def Save_img(predict,filename,ors, startpointlist2, k,size_crop, pathsave):
    print(k)
    img = ors.read_region(startpointlist2[k], 0, (size_crop,size_crop))
    img = np.array(img)
    img = img[:, :, 0 : 3]
    flag = 0
    if predict[k] > 0.9:
        pathsave_sample = pathsave + '\\n_9\\'
    elif predict[k] > 0.8:
        pathsave_sample = pathsave + '\\n_8\\'
    elif predict[k] > 0.5:
        pathsave_sample = pathsave + '\\n_5\\'
    elif predict[k] > 0.2:
        pathsave_sample = pathsave + '\\n_2\\'
    elif predict[k] > 0.1:
        pathsave_sample = pathsave + '\\n_1\\'
    elif predict[k] >= 0:
        flag =  random.randint(0,30)
        pathsave_sample = pathsave + '\\n_0\\'
    if not os.path.exists(pathsave_sample):
        os.makedirs(pathsave_sample)
        print('Save in %s' % pathsave_sample)
    if not flag:
        cv2.imwrite(pathsave_sample + filename +'_%5f_%s.tif'%(predict[k],startpointlist2[k]),img[...,::-1])
    return None 

def Save_img_sdpc(predict,filename,ors, startpointlist2, k,size_crop, pathsave):
    print(k)
    img = ors.getTile(0, startpointlist2[k][1], startpointlist2[k][0], size_crop, size_crop)
    if  img==None:
        return None
    img = np.ctypeslib.as_array(img)
    img.dtype = np.uint8
    img = img.reshape((size_crop, size_crop, 3))
    flag = 0
    if predict[k] > 0.9:
        pathsave_sample = pathsave + '\\n_9\\'
    elif predict[k] > 0.8:
        pathsave_sample = pathsave + '\\n_8\\'
    elif predict[k] > 0.5:
        pathsave_sample = pathsave + '\\n_5\\'
    elif predict[k] > 0.2:
        pathsave_sample = pathsave + '\\n_2\\'
    elif predict[k] > 0.1:
        pathsave_sample = pathsave + '\\n_1\\'
    elif predict[k] >= 0:
        flag =  random.randint(0,10)
        pathsave_sample = pathsave + '\\n_0\\'
    if not os.path.exists(pathsave_sample):
        os.makedirs(pathsave_sample)
    if not flag:
        cv2.imwrite(pathsave_sample + filename +'_%s_%5f.tif'%(startpointlist2[k],predict[k]),img)
    return None 

def Get_nplus():
    pathsvs = pathfolder_svs + filename + size_set['filetype']
    print(pathfolder_svs + filename + size_set['filetype'])
    
    startpointlist0 = np.load(pathsave1 + filename + '_s.npy')
    predict1 = np.load(pathsave1 + filename + '_p.npy')
    feature = np.load(pathsave1 + filename + '_f.npy') 
    startpointlist_split = np.load(pathsave1 + 'startpointlist_split.npy') 
    sizepatch_predict_small1 = size_set['sizepatch_predict_small1']
    ratio = input_str['model1_resolution']/input_str['input_resolution']
    
    num = len(startpointlist_split)
    startpointlist2 = []
    predict = []
    for index, item in enumerate(feature):
        Local_3 = region_proposal(item[...,1], sizepatch_predict_small1, img=None, threshold=0.7)
        for local in Local_3:
            startpointlist_2_x = int(startpointlist0[index//num][1]+\
            (startpointlist_split[index%num][1] + local[1])*ratio
            - int(size_crop/2))
            
            startpointlist_2_y = int(startpointlist0[index//num][0]+\
            (startpointlist_split[index%num][0] + local[0])*ratio
            - int(size_crop/2))
            startpointlist2.append((startpointlist_2_y,startpointlist_2_x))
            predict.append(predict1[index])
            
    if size_set['filetype'] == '.sdpc':
        ors = sdpc.Sdpc()
        ors.open(pathsvs)
    else:
        ors = OpenSlide(pathsvs)
    
    if size_set['filetype'] != '.sdpc':
        pool = multiprocessing.Pool(5)
        for k in range(len(startpointlist2)):#len(startpointlist2)
            pool.apply_async(Save_img,
                             args=(predict,filename,ors, startpointlist2, k,size_crop, pathsave))
        pool.close()
        pool.join()
     
    else:
        for k in range(len(startpointlist2)):#len(startpointlist2)
           Save_img_sdpc(predict,filename,ors, startpointlist2, k,size_crop, pathsave)
      
    return None

if __name__ == '__main__':
    # 1.参数设定
    input_str = {'filetype': '.mrxs',# .svs .sdpc .mrxs
                 'level': 0,
                 'read_size_model1_resolution': (1216, 1936), #1936 1216 横 model1_resolution下
                 'scan_over': (120, 120),
                 'model1_input': (512, 512, 3),
                 'model2_input': (256, 256, 3),
                 'model1_resolution':0.586,
                 'model2_resolution':0.293,
                 'input_resolution':0.243,#0.179
                 'cut':'3*5'}
    
    size_set = Size_set(input_str)
    
    pathfolder_svs_set,_= TCT_set()
    
    pathfolder_svs = pathfolder_svs_set['3d2'][1]
    pathsave1 = r'F:\recom\340_adapt\Shengfuyou_2th\Negative' +'\\'

    pathsave_dict = {'train':r'H:\AdaptDATA\train\3d\sfy2\n',
                     'test':r'H:\AdaptDATA\test\3d\sfy2\n'}
    testlist = test_sample['test_n_2']
    
    CZ = os.listdir(pathsave1)
    CZ = [cz[:-len('_p.npy')] for cz in CZ if '_p.npy' in cz]
    
    for filename in CZ[200:]:
        size_crop = 1843
        if filename in testlist:
            print('yes')
            pathsave  =  pathsave_dict['test']
        else:
            pathsave  =  pathsave_dict['train']
        Get_nplus()
        
            
            
            
            
            
            
            
            
            
            
            
            
            
           
   