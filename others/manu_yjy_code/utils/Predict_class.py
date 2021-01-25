# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 11:05:56 2019

@author: YuJingya
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
from utils.nucleus_analysis import cal_focus_nucleus_feature_v1 as cal_nucleus_v1
from utils.nucleus_analysis import cal_focus_nucleus_feature_v2 as cal_nucleus_v2
from utils.parameter_set import Size_set
from lxml import etree
from utils.function_set import * 

class Recommend(object):
    """从预测的npy文件中 推荐一定数目区域 并保存成xml格式"""
    def __init__(self, pathsave1,pathsave_xml,
                 filename, input_str, num_recom,
                 pathsave2=None, pathsave2_core=None,):
       self.pathsave1 = pathsave1
       self.pathsave2 = pathsave2
       self.pathsave2_core = pathsave2_core
       self.pathsave_xml = pathsave_xml
       self.filename = filename
       self.input_str = input_str
       self.num_recom = num_recom
       
    def get_startpointlist1(self):
        """计算model1在全片上的绝对坐标"""
        pathsave1 = self.pathsave1
        filename = self.filename
        input_str = self.input_str
        ratio = input_str['ratio']
        sizepatch_small1 = Size_set(input_str)['sizepatch_small1']
        
        startpointlist0 = np.load(pathsave1 + filename + '_s.npy')
        startpointlist_split = np.load(pathsave1 + 'startpointlist_split.npy')
        predict1 = np.load(pathsave1 + filename + '_p.npy')
        startpointlist_split = np.int16(np.array(startpointlist_split)*ratio)
        startpointlist1 = []
        for i in range(len(startpointlist0)):
            for j in range(len(startpointlist_split)):
                points = (startpointlist0[i][0]+startpointlist_split[j][0],startpointlist0[i][1]+startpointlist_split[j][1])
                startpointlist1.append(points)
        Sizepatch_small1 = [sizepatch_small1]*len(startpointlist1)
        return startpointlist1, predict1, Sizepatch_small1
            
    def get_startpointlist2(self):
        """计算model2在全片上的绝对坐标"""
        pathsave1 = self.pathsave1
        pathsave2 = self.pathsave2
        filename = self.filename
        input_str = self.input_str
        ratio = input_str['ratio']
        sizepatch_small2 = Size_set(input_str)['sizepatch_small2']
        
        startpointlist0 = np.load(pathsave1 + filename + '_s.npy')
        dictionary = np.load(pathsave2 + filename + '_dictionary.npy')
        startpointlist2_relat = np.load(pathsave2 + filename + '_s2.npy')
        predict2 = np.load(pathsave2 + filename + '_p2.npy')
        startpointlist2_relat = np.int16(np.array(startpointlist2_relat)*ratio)    
        sizepatch_small2
        startpointlist2 = []
        for index, item in enumerate(dictionary):
            count = sum(dictionary[:index])
            for i in range(item):
                points = (startpointlist0[index//15][0] + startpointlist2_relat[count][0],startpointlist0[index//15][1] + startpointlist2_relat[count][1])
                startpointlist2.append(points)
                count = count+1
        Sizepatch_small2 = [tuple(np.array(sizepatch_small2)*ratio)]*len(startpointlist2)
        return startpointlist2, predict2, Sizepatch_small2
    
    def get_startpointlist12(self):
        """model1 和 model2 绝对坐标简单合并"""
        startpointlist1, predict1, Sizepatch_small1 = self.get_startpointlist1()
        startpointlist2, predict2, Sizepatch_small2 = self.get_startpointlist2()
        
        startpointlist12 = startpointlist1 + startpointlist2
        Sizepatch_small12 = Sizepatch_small1 + Sizepatch_small2
        predict12 = np.vstack((predict1,predict2))
        self.startpointlist = startpointlist12 
        self.predict = predict12
        self.Sizepatch_small = Sizepatch_small12
        return None
    
    def recommend_region(self,startpointlist, predict, Sizepatch_small,num_red = None):
        """按照概率值推荐前num_recom个区域 
        并保存xml结果 阳性为红色 阴性为绿色"""
        num_recom = self.num_recom
        Num_recom = min((len(predict),num_recom))
        Index = np.argsort(np.array(predict),axis=0)[::-1][:Num_recom]
        startpointlist_recom = [startpointlist[index[0]] for index in Index]
        Sizepatch_small_recom = [Sizepatch_small[index[0]] for index in Index]
        predict_recom = [str(i)+'_'+str(predict[Index[i][0]][0]) for i in range(len(Index))]
        if num_red!=None:
            Num_red = num_red
        else:
            Num_red = np.sum((predict_recom>0.5))
        color = ['#ff0000'] * Num_red + ['#00ff00'] * (Num_recom-Num_red) 
        contourslist_small_recom = Get_rectcontour(startpointlist_recom,Sizepatch_small_recom)
        return contourslist_small_recom, predict_recom, color
     
    def get_imgTotal2_core(self,pathfolder_svs):
        pathsave1 = self.pathsave1
        filename = self.filename
        input_str = self.input_str
        
        filetype = Size_set(input_str)['filetype']
        level = Size_set(input_str)['level']
        sizepatch = Size_set(input_str)['sizepatch']
        sizepatch_predict = Size_set(input_str)['sizepatch_predict']
        sizepatch_predict_small1 = Size_set(input_str)['sizepatch_predict_small1']
        
        pathsvs = pathfolder_svs + filename + filetype
        ors = OpenSlide(pathsvs)
        startpointlist = np.load(pathsave1 + filename + '_s.npy')
        predict1 = np.load(pathsave1 + filename + '_p.npy')
        feature = np.load(pathsave1 + filename + '_f.npy') 
        startpointlist_split = np.load(pathsave1 + 'startpointlist_split.npy') 
        imgTotal = Get_predictimgMultiprocess(startpointlist, ors, level, sizepatch, sizepatch_predict)
        # 计算model2定位坐标起始点并读图
        size_crop = 512
        startpointlist_2 = []
        imgTotal2_core = []
        dictionary = []
        for index, item in enumerate(feature):
            if predict1[index] > 0.5:
                Local_3 = region_proposal(item[...,1], sizepatch_predict_small1, img=None, threshold=0.7)
                dictionary.append(len(Local_3))
                for local in Local_3:
                    startpointlist_2_x = startpointlist_split[index%15][1] + local[1]- int(size_crop/2)
                    startpointlist_2_y = startpointlist_split[index%15][0] + local[0]- int(size_crop/2)
                    img = imgTotal[index//15][max(startpointlist_2_x,0):min(startpointlist_2_x+size_crop,sizepatch[1]),max(startpointlist_2_y,0):min(startpointlist_2_y+size_crop,sizepatch[0])]
                    if img.shape[0:2]!= (size_crop,size_crop):
                        img_temp = np.zeros((size_crop,size_crop,3),np.uint8)
                        x_temp = max(-startpointlist_2_x,0)
                        y_temp = max(-startpointlist_2_y,0)
                        img_temp[x_temp:x_temp+img.shape[0],y_temp:y_temp+img.shape[1]] = img
                        img = img_temp
                    imgTotal2_core.append(img)
                    startpointlist_2.append((startpointlist_2_y,startpointlist_2_x))
            else:
                dictionary.append(0)
        return imgTotal2_core
        
    def get_contours_and_feature_of_core(self,imgTotal2_core = None):
        """计算核分割后核的全局坐标"""
        pathsave2_core = self.pathsave2_core
        filename  = self.filename 
        input_str = self.input_str
        ratio = input_str['ratio']
        predict2_core = np.load(pathsave2_core + filename + '_p2_core.npy')
        # 计算核特征 起始点
        if imgTotal2_core == None:
            Areas,Contours_core_relat = cal_nucleus_v1(predict2_core)
            Valid_flag = [True]*len(Areas)
        else:   
            Areas, Valid_flag, Contours_core_relat = cal_nucleus_v2(imgTotal2_core,predict2_core)
        startpointlist2, _, _ = self.get_startpointlist2()
        Contours_core = []
        for index, item in enumerate(Contours_core_relat):
            contours_temp = []
            for points in item:
                points = (startpointlist2[index][0] + (points[...,0]-192)*ratio,
                          startpointlist2[index][1] + (points[...,1]-192)*ratio)
                points= np.array(points)
                points= np.transpose(points)
                contours_temp.append(points)
            Contours_core.append(contours_temp)
# =============================================================================
#             if len(Areas)!=len(predict2):
#                 raise ValueError('model2和核分割处理对象不同')
# =============================================================================
        self.Areas = Areas
        self.Valid_flag = Valid_flag
        self.Contours_core = Contours_core
        return Areas, Valid_flag, Contours_core
        
    def recommend_core(self,Areas, Valid_flag, Contours_core):     
        """按照概率值推荐前num_recom个核 
        前22为红色 其他为绿色"""
#            num_recom = self.num_recom
#        Num_recom = min((len(Areas),num_recom))
        Areas_max_Ture = [np.max(Areas[i])*Valid_flag[i] for i  in range(len(Areas))]
        Index_core_Ture = np.argsort(np.array(Areas_max_Ture),axis=0)[::-1]
        Areas_max_False = [np.max(Areas[i])*(1-Valid_flag[i]) for i  in range(len(Areas))]
        Index_core_False = np.argsort(np.array(Areas_max_False),axis=0)[::-1]
        Index_core =  Index_core_Ture[:Valid_flag.count(True)].tolist() + Index_core_False[Valid_flag.count(True):].tolist()
                  
        contourslist_core_recom = [Contours_core[index] for index in Index_core]
        predict2_core_recom = [Areas[index] for index in Index_core]
        annotation_Valid_flag = [Valid_flag[index] for index in Index_core]
        dictionary_core = [len(x) for x in contourslist_core_recom]
        # 将数组拉平
        contourslist_core_recom_flat = []
        predict2_core_recom_flat = []
        for index,b in enumerate(contourslist_core_recom):
            for indexx,c in enumerate(b):
                contourslist_core_recom_flat.append(c)
                predict2_core_recom_flat.append(str(index)+'_'+str(annotation_Valid_flag[index])+'_'+str(predict2_core_recom[index][indexx]))
        # 上色
        num_core_recom = np.sum(dictionary_core)
        num_core_22 = np.sum(dictionary_core[:22])
        color2_core = ['#ff0000'] * num_core_22  + ['#00ff00'] * (num_core_recom - num_core_22) 
        return contourslist_core_recom_flat, predict2_core_recom_flat, color2_core
    
    def recommend_region_combine_core(self,Areas, Valid_flag, Contours_core, num_recom = 22):     
        """按照概率值推荐前num_recom个核 
        前22为红色 其他为绿色"""
        startpointlist2, predict2, Sizepatch_small2 = self.get_startpointlist2()
        Num_recom = min((len(predict2),num_recom))
        Index = np.argsort(np.array(predict2),axis=0)[::-1][:Num_recom].reshape(Num_recom,)
        
        Areas_max_Ture = [np.max(Areas[i])*Valid_flag[i] for i  in range(len(Areas))]
        Index_core_Ture = np.argsort(np.array(Areas_max_Ture),axis=0)[::-1]
        Areas_max_False = [np.max(Areas[i])*(1-Valid_flag[i]) for i  in range(len(Areas))]
        Index_core_False = np.argsort(np.array(Areas_max_False),axis=0)[::-1]
        Index_core_whole = np.concatenate((Index_core_Ture[:Valid_flag.count(True)],
                     Index_core_False[Valid_flag.count(True):]),axis = 0)
        Index_core = Index_core_whole[:Num_recom]
         
        
        # 前22区域重合部分 加 各补上一部分 得到最终的 Index
        Index_union = list(set(Index) & set(Index_core))
        num_complet = Num_recom - len(Index_union)
        num_complet_model2 = int(num_complet/2)
        num_complet_model2_core = num_complet - num_complet_model2
        Index_complet = [x  for x in Index if x not in Index_union][:num_complet_model2]
        
        Index_core_complet1 = [x  for x in Index_core_whole if x not in Index_union + Index_complet and predict2[x]>0.5]
        Index_core_complet2 = [x for x in Index_core_whole if x not in Index_union + Index_complet and predict2[x]<=0.5 ]
        Index_core_complet = Index_core_complet1 + Index_core_complet2
        Index_core_complet = Index_core_complet[:num_complet_model2_core]
        
        Index_final = Index_union + Index_complet + Index_core_complet
        # model2 相关xml参数
        startpointlist_recom2 = [startpointlist2[index] for index in Index_final]
        Sizepatch_small_recom2 = [Sizepatch_small2[index] for index in Index_final]
        predict_recom2 = [str(predict2[index][0]) for index in Index_final]
        contourslist_small_recom2 = Get_rectcontour(startpointlist_recom2,Sizepatch_small_recom2)
        color2 = ['#ff0000'] * len(Index_union + Index_complet) + ['#00ff00'] * len(Index_core_complet)
          
        # model2_core 相关xml参数          
        contourslist_core_recom = [Contours_core[index] for index in Index_final]
        predict2_core_recom = [Areas[index] for index in Index_final]
        annotation_Valid_flag = [Valid_flag[index] for index in Index_final]
        dictionary_core = [len(x) for x in contourslist_core_recom]
        # 将数组拉平
        contourslist_core_recom_flat = []
        predict2_core_recom_flat = []
        for index,b in enumerate(contourslist_core_recom):
            for indexx,c in enumerate(b):
                contourslist_core_recom_flat.append(c)
                predict2_core_recom_flat.append(str(annotation_Valid_flag[index])+'_'+str(predict2_core_recom[index][indexx]))
        # 上色
        num_core_union = np.sum(dictionary_core[:len(Index_union)])
        num_core_self = np.sum(dictionary_core[-len(Index_core_complet):])
        num_core_recom = np.sum(dictionary_core)
        color2_core = ['#ff0000'] * num_core_union  + ['#00ff00'] * (num_core_recom - num_core_union - num_core_self)  + ['#ff0000'] * (num_core_self)
        return contourslist_small_recom2, predict_recom2, color2,contourslist_core_recom_flat, predict2_core_recom_flat, color2_core

    