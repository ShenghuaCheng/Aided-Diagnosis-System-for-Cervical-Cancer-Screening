# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 21:02:41 2019

@author: YuJingya
"""
import sys
sys.path.append('../')
from utils.parameter_set import TCT_set,Size_set
import os
from matplotlib import pyplot as plt
import time
import numpy as np
import cv2
from openslide import OpenSlide
import numpy as np
import os
import heapq
from tqdm import tqdm 
from utils.function_set import saveContours_xml
from utils.Recommend_class import Recommend
    
if __name__ == '__main__':
    input_str = {'filetype': '.sdpc',# .svs .sdpc .mrxs
                 'level': 0,
                 'read_size_model1_resolution': (1216, 1936), #1936 1216 横 model1_resolution下
                 'scan_over': (120, 120),
                 'model1_input': (512, 512, 3),
                 'model2_input': (256, 256, 3),
                 'model1_resolution':0.586,
                 'model2_resolution':0.293,
                 'input_resolution':0.293,#0.179
                 'cut':'3*5'}
    num_recom = 22
    size_set = Size_set(input_str)
    pathfolder_set, _ = TCT_set()
    for pathfolder in pathfolder_set['20x'][0:1]:
        pathsave1 = 'F:/recom/model2_core_new_v5/2018/'
        pathsave2 = 'F:/recom/model2_core_new_v5/2018/model2_multi/'
        pathsave_xml = pathsave2+'xml_multi/'
        while not os.path.exists(pathsave_xml):
            os.makedirs(pathsave_xml)  
        CZ = os.listdir(pathsave2)
        CZ = [cz[:-len('_p2.npy')] for cz in CZ if '_p2.npy' in cz]
        for filename in tqdm(CZ):
            
            
            predict2_multi
            
            num_recom = self.num_recom
            Num_recom = min((len(predict),num_recom))
            Index = np.argsort(np.array(predict),axis=0)[::-1][:Num_recom]
            startpointlist_recom = [startpointlist[index[0]] for index in Index]
            Sizepatch_small_recom = [Sizepatch_small[index[0]] for index in Index]
            predict_recom = [str(i)+'_'+str(predict[Index[i][0]][0]) for i in range(len(Index))]
            if num_red!=None:
                Num_red = num_red
            else:
                predict_recom_temp = [predict[Index[i][0]][0] for i in range(len(Index))]
                predict_recom_temp = np.array(predict_recom_temp)
                Num_red = np.sum((predict_recom_temp>0.5))
            color = ['#ff0000'] * Num_red + ['#00ff00'] * (Num_recom-Num_red) 
            contourslist_small_recom = Get_rectcontour(startpointlist_recom,Sizepatch_small_recom)
            return contourslist_small_recom, predict_recom, color
   