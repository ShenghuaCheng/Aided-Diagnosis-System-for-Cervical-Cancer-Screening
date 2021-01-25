# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 21:02:41 2019

@author: yujingya
"""
import sys
sys.path.append('../')
from recommend.Generate_heatmap_from_npy import Generate_heatmap
from utils.parameter_set import Size_set,TCT_set
from utils.function_set import *
import os
from matplotlib import pyplot as plt
import time
import numpy as np
import cv2
from openslide import OpenSlide
import numpy as np
import os
import heapq

if __name__ == '__main__':
    input_str = {'flag':'our',
                 'level' : 0,
                 'read_size':(1216,1936),
                 'model1_input':(512,512,3),
                 'model2_input':(256,256,3),
                 'scan_over':(120,120),
                 'ratio':1,
                 'cut':'3*5_adapt'}
    sizepatch, widthOverlap, sizepatch_small1,sizepatch_small2, sizepatch_predict, sizepatch_predict_small1, sizepatch_predict_small2, widthOverlap_predict, flag, filetype, level, levelratio = Size_set(input_str)
    pathfolder_svs_set, pathsave_set, _ = TCT_set()
    for pathfolder_svs,pathsave1 in zip(pathfolder_svs_set['10x'],pathsave_set['10x']):
#        break
        pathsave2 = pathsave1 + 'model2/'
        pathsave2_1030 = pathsave1 + 'model2_1030/'
        pathsave_img = pathsave1 + 'heatmap/'
        while not os.path.exists(pathsave_img):
            os.mkdir(pathsave_img)  
        CZ = os.listdir(pathsave2)
        CZ = [cz[:-len('_s2.npy')] for cz in CZ if '_s2.npy' in cz]
        for filename in CZ[4:]:
#            break
            
            pathsvs = pathfolder_svs + filename + filetype
            print(pathsvs)
            ors = OpenSlide(pathsvs)
            img_whole  = ors
            feature_map,predict_map1,predict_map12 = Generate_heatmap(pathsave1,filename, pathsave2)
            predict_map1_temp = cv2.resize(predict_map1,feature_map.shape[0:2][::-1],interpolation = cv2.INTER_NEAREST)
            predict_map12_temp = cv2.resize(predict_map12,feature_map.shape[0:2][::-1],interpolation = cv2.INTER_NEAREST)
            
            _, _,predict_map12_1030 = Generate_heatmap(pathsave1,filename, pathsave2_1030)
            predict_map12_temp_1030 = cv2.resize(predict_map12_1030,feature_map.shape[0:2][::-1],interpolation = cv2.INTER_NEAREST)
            
            img0 = np.hstack((predict_map1_temp,feature_map))
            img1 = np.hstack((predict_map12_temp,predict_map12_temp_1030))
            img = np.vstack((img0, img1))
            cv2.imwrite(pathsave_img + filename + '.tif',img*255)
            