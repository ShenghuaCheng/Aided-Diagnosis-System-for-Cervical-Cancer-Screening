# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 19:09:03 2019

@author: yujingya
"""
import sys
import os
from matplotlib import pyplot as plt
import time
import numpy as np
import cv2
from openslide import OpenSlide
import numpy as np
import os
import heapq
from utils.nucleus_analysis import cal_focus_nucleus_feature as cal_nucleus
from utils.function_set import *


def Recommend_from_model2_core(pathsave1,pathsave2,pathsave2_core,pathsave_xml,filename,input_str,,sizepatch_small2,num_recom):
    startpointlist = np.load(pathsave1 + filename + '_s.npy')
    startpointlist_split = np.load(pathsave1 + 'startpointlist_split.npy')
    predict1 = np.load(pathsave1 + filename + '_p.npy')
    dictionary = np.load(pathsave2 + filename + '_dictionary.npy')
    predict2 = np.load(pathsave2 + filename + '_p2.npy')
    predict2_core = np.load(pathsave2_core + filename + '_p2_core.npy')
    startpointlist2_relat = np.load(pathsave2 + filename + '_s2.npy')
    sizepatch_small1 = Size_set(input_str)['sizepatch_small1']
    sizepatch_small2 = Size_set(input_str)['sizepatch_small2']
    ratio = input_str['ratio']
    startpointlist_split = np.int16(np.array(startpointlist_split)*ratio)
    startpointlist2_relat = np.int16(np.array(startpointlist2_relat)*ratio)
    
    # 计算startpointlist1
    startpointlist1 = []
    for i in range(len(startpointlist)):
        for j in range(len(startpointlist_split)):
            points = (startpointlist[i][0]+startpointlist_split[j][0],startpointlist[i][1]+startpointlist_split[j][1])
            startpointlist1.append(points)
    Sizepatch_small1 = [sizepatch_small1]*len(startpointlist1)
    # 计算startpointlist2
    startpointlist2 = []
    for index, item in enumerate(dictionary):
        count = sum(dictionary[:index])
        for i in range(item):
            points = (startpointlist[index//15][0] + startpointlist2_relat[count][0],startpointlist[index//15][1] + startpointlist2_relat[count][1])
            startpointlist2.append(points)
            count = count+1
    Sizepatch_small2 = [tuple(np.array(sizepatch_small2)*ratio)]*len(startpointlist2)
# =============================================================================
#      # 计算startpointlist12
#     startpointlist_12 = startpointlist1 + startpointlist2
#     Sizepatch_small_12 = Sizepatch_small1 + Sizepatch_small2
#     predict12 = np.vstack((predict1,predict2))
# =============================================================================
   
    # 计算核特征 起始点
    Areas,Contours_core_relat = cal_nucleus(predict2_core)
    Contours_core = []
    for index, item in enumerate(dictionary):
        count = sum(dictionary[:index])
        for i in range(item):
            contours_temp = []
            for j,points in enumerate(Contours_core_relat[count]):
                points = (startpointlist[index//15][0] + startpointlist2_relat[count][0] + (points[...,0]-192)*ratio,
                          startpointlist[index//15][1] + startpointlist2_relat[count][1] + (points[...,1]-192)*ratio)
                points= np.array(points)
                points= np.transpose(points)
                contours_temp.append(points)
            count = count+1
            Contours_core.append(contours_temp)
    
    
    if len(Areas)!=len(predict2):
        raise ValueError('model2和核分割处理对象不同')
        
    # 开始推荐
    Num_recom = min((len(predict2),num_recom))
    Areas_max = [np.max(item) for item in Areas]
    Index_core = np.argsort(np.array(Areas_max),axis=0)[::-1]
    Index = np.argsort(np.array(predict2),axis=0)[::-1][:Num_recom]
    
    startpointlist2_recom = [startpointlist2[index[0]] for index in Index]
    Sizepatch_small2_recom = [Sizepatch_small2[index[0]] for index in Index]
    contourslist_small2_recom = Get_rectcontour(startpointlist2_recom,Sizepatch_small2_recom)
    predict2_recom = [str(i)+'_'+str(predict2[Index[i][0]][0]) for i in range(len(Index))]
    color2 = ['#ff0000'] * 22  + ['#00ff00'] * (Num_recom-22) 
              
    contourslist_core_recom = [Contours_core[index] for index in Index_core]
    predict2_core_recom = [Areas[index] for index in Index_core]
    dictionary_core = [len(x) for x in contourslist_core_recom]
    
    contourslist_core_recom_flat = []
    predict2_core_recom_flat = []
    for index,b in enumerate(contourslist_core_recom):
        for indexx,c in enumerate(b):
            contourslist_core_recom_flat.append(c)
            predict2_core_recom_flat.append(str(index)+'_'+ str(predict2_core_recom[index][indexx]))
    
    
    num_core_recom = np.sum(dictionary_core)
    num_core_22 = np.sum(dictionary_core[:22])
    color2_core = ['#ff0000'] * num_core_22  + ['#00ff00'] * (num_core_recom - num_core_22) 
              
    saveContours_xml([contourslist_small2_recom,contourslist_core_recom_flat],[predict2_recom,predict2_core_recom_flat],[color2,color2_core],pathsave_xml + filename + '.xml')
    return None    


