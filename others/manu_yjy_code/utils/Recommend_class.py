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
from utils.nucleus_analysis import cal_focus_nucleus_feature_v3 as cal_nucleus_v3
from utils.parameter_set import Size_set
from lxml import etree
from utils.function_set import * 

class Recommend(object):
    """从预测的npy文件中 推荐一定数目区域 并保存成xml格式"""
    def __init__(self, pathsave1,pathsave_xml,
                 filename, input_str, num_recom_min, num_recom_max,
                 pathsave2=None, pathsave2_core=None,):
       self.pathsave1 = pathsave1
       self.pathsave2 = pathsave2
       self.pathsave2_core = pathsave2_core
       self.pathsave_xml = pathsave_xml
       self.filename = filename
       self.input_str = input_str
       self.num_recom_min = num_recom_min
       self.num_recom_max = num_recom_max
       ratio = input_str['model1_resolution']/input_str['input_resolution']
       self.ratio = ratio
       
    def get_startpointlist1(self):
        """计算model1在全片上的绝对坐标"""
        pathsave1 = self.pathsave1
        filename = self.filename
        input_str = self.input_str
        ratio = self.ratio
        
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
        ratio = self.ratio
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
        Sizepatch_small2 = [sizepatch_small2]*len(startpointlist2)
        return startpointlist2, predict2, Sizepatch_small2
    
    def get_startpointlist12(self):
        """model1 和 model2 绝对坐标简单合并"""
        startpointlist1, predict1, Sizepatch_small1 = self.get_startpointlist1()
        startpointlist2, predict2, Sizepatch_small2 = self.get_startpointlist2()
        filename = self.filename
        pathsave2 = self.pathsave2
        dictionary = np.load(pathsave2 + filename + '_dictionary.npy')
        startpointlist12 = []
        predict12 = []
        Sizepatch_small12 = []
        for index, item in enumerate(dictionary):
            if item!=0:
                a = predict2[int(np.sum(dictionary[:index])):int(np.sum(dictionary[:index])+item)]
                index_max = int(np.sum(dictionary[:index])) + a.argmax()
                startpointlist12.append(startpointlist2[index_max])
                predict12.append(a.max())
                Sizepatch_small12.append(Sizepatch_small2[index_max])
            else:
                startpointlist12.append(startpointlist1[index])
                predict12.append(predict1[index])
                Sizepatch_small12.append(Sizepatch_small1[index])
        predict12 = np.vstack(predict12)
        return startpointlist12, predict12, Sizepatch_small12
  
    
    def recommend_region(self,startpointlist, predict, Sizepatch_small,num_red = None):
        """按照概率值推荐前num_recom个区域 
        并保存xml结果 阳性为红色 阴性为绿色"""
        num_recom_min = self.num_recom_min
        num_recom_max = self.num_recom_max
        
        Num_recom = min(sum(predict>0.5)[0],num_recom_max)
        Num_recom = max(Num_recom,num_recom_min)
        print(Num_recom)
        Index = np.argsort(np.array(predict),axis=0)[::-1][:Num_recom]
        startpointlist_recom = [startpointlist[index[0]] for index in Index]
        Sizepatch_small_recom = [Sizepatch_small[index[0]] for index in Index]
        predict_recom = [str(i)+'_'+str(predict[Index[i][0]][0]) for i in range(len(Index))]
        
        predict_recom_temp = [predict[Index[i][0]][0] for i in range(len(Index))]
        predict_recom_temp = np.array(predict_recom_temp)
        
        color = ['#ff0000'] * Num_recom
        contourslist_small_recom = Get_rectcontour(startpointlist_recom,Sizepatch_small_recom)
        return contourslist_small_recom, predict_recom, color
    
    def recommend_region_exclude_near(self,startpointlist, predict, Sizepatch_small,num_red = None):
        """按照概率值推荐前num_recom个区域 
        并保存xml结果 阳性为红色 阴性为绿色"""
        num_recom_min = self.num_recom_min
        num_recom_max = self.num_recom_max
        Num_recom = min(sum(predict>0.5),num_recom_max)
        Num_recom = max(Num_recom,num_recom_min)
        
        input_str = self.input_str
        
        sizepatch_small2 = Size_set(input_str)['sizepatch_small2']
        size_dis = int(sizepatch_small2[0]/4*3)
        print('size_dis='+str(size_dis))
        near_index = []#临近框框index
        for index_point1,point1 in enumerate(startpointlist): 
            near_point1 = [index_point1]
            for index_point2,point2 in enumerate(startpointlist[index_point1+1:]):
                if np.abs(point1[0]-point2[0])<size_dis and np.abs(point1[1]-point2[1])<size_dis:
                    near_point1.append(index_point1+index_point2+1)
            if len(near_point1)!=1:
                near_index.append(near_point1)
        for item in near_index:
            max_predict_index = item[0]
            max_predict = predict[item[0]]
            for itemm in item:
                if predict[itemm] > max_predict:
                    max_predict_index = itemm
                    max_predict = predict[itemm]
            for itemm in item:
                if itemm != max_predict_index:
                    predict[itemm] = 0
        
        Index = np.argsort(np.array(predict),axis=0)[::-1][6:Num_recom]
        startpointlist_recom = [startpointlist[index[0]] for index in Index]
        Sizepatch_small_recom = [Sizepatch_small[index[0]][0] for index in Index]
        predict_recom = [str(i)+'_'+str(predict[Index[i][0]][0]) for i in range(len(Index))]
        
        color = ['#ff0000'] * Num_recom
        contourslist_small_recom = Get_rectcontour(startpointlist_recom,Sizepatch_small_recom)
        
        return contourslist_small_recom, predict_recom, color
    
    
    def get_imgTotal2_nuclues(self,pathfolder):
        pathsave1 = self.pathsave1
        filename = self.filename
        input_str = self.input_str
        size_set = Size_set(input_str)
        filetype = size_set['filetype']
        level = size_set['level']
        sizepatch_read = size_set['sizepatch_read']
        sizepatch_predict_model1 = size_set['sizepatch_predict_model1']
        sizepatch_predict_small1 = size_set['sizepatch_predict_small1']
        startpointlist = np.load(pathsave1 + filename + '_s.npy')
        predict1 = np.load(pathsave1 + filename + '_p.npy')
        feature = np.load(pathsave1 + filename + '_f.npy') 
        startpointlist_split = np.load(pathsave1 + 'startpointlist_split.npy') 
        
        pathfile = pathfolder + filename + filetype
        imgTotal = Get_predictimgMultiprocess(pathfile, startpointlist, level, sizepatch_read)
        imgTotal_model1 = [cv2.resize(img,sizepatch_predict_model1) for img in imgTotal]
        imgTotal_model1 = np.array(imgTotal_model1)
    
        num = len(startpointlist_split)
       
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
                    img = imgTotal_model1[index//15][max(startpointlist_2_x,0):min(startpointlist_2_x+size_crop,
                                         sizepatch_predict_model1[1]),max(startpointlist_2_y,0):min(startpointlist_2_y+size_crop,sizepatch_predict_model1[0])]
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
    
    def get_imgTotal2_core(self,pathfolder,online = False):
        pathsave1 = self.pathsave1
        filename = self.filename
        input_str = self.input_str
        ratio = 2
        size_set = Size_set(input_str)
        filetype = size_set['filetype']
        level = size_set['level']
        sizepatch_read = size_set['sizepatch_read']
        sizepatch_predict_model2 = size_set['sizepatch_predict_model2']
        sizepatch_predict_small2 = size_set['sizepatch_predict_small2']
        sizepatch_predict_small1 = size_set['sizepatch_predict_small1']
        startpointlist = np.load(pathsave1 + filename + '_s.npy')
        predict1 = np.load(pathsave1 + filename + '_p.npy')
        feature = np.load(pathsave1 + filename + '_f.npy') 
        startpointlist_split = np.load(pathsave1 + 'startpointlist_split.npy') 
        
        pathfile = pathfolder + filename + filetype
        imgTotal = Get_predictimgMultiprocess(pathfile, startpointlist, level, sizepatch_read)
        imgTotal_model2 = [cv2.resize(img,sizepatch_predict_model2) for img in imgTotal]
        imgTotal_model2 = np.array(imgTotal_model2)
        
        num = len(startpointlist_split)
        #取前500
        score_500 = np.sort(np.vstack(predict1),axis=0)[-500]
        # 计算model2定位坐标起始点并读图
        size_crop = 256
        startpointlist_2 = []
        imgTotal2_core = []
        dictionary = []
        for index, item in enumerate(feature):
#            break
            if predict1[index] >= score_500:
#                break
                Local_3 = region_proposal(item[...,1],(1024,1024), img=None, threshold=0.7)
                dictionary.append(len(Local_3))
                for local in Local_3:
#                    break
                    startpointlist_2_x = int(startpointlist_split[index%15][1]*ratio + local[1]- int(size_crop/2))
                    startpointlist_2_y = int(startpointlist_split[index%15][0]*ratio + local[0]- int(size_crop/2))
                    img = imgTotal_model2[index//15][max(startpointlist_2_x,0):min(startpointlist_2_x+size_crop,
                                         sizepatch_predict_model2[1]),max(startpointlist_2_y,0):min(startpointlist_2_y+size_crop,sizepatch_predict_model2[0])]
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
        return imgTotal2_core,dictionary
    
    def get_contours_and_feature_of_core(self,imgTotal2_core,predict2_core):
        """计算核分割后核的全局坐标"""
        pathsave2_core = self.pathsave2_core
        filename  = self.filename 
        input_str = self.input_str
        ratio = self.ratio
        # 计算核特征 起始点
        Areas, Type_img, Contours_core_relat = cal_nucleus_v3(imgTotal2_core,predict2_core)
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
        self.Type_img = Type_img
        self.Contours_core = Contours_core
        return Areas, Type_img, Contours_core
        
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
    
    def recommend_region_combine_core(self,imgTotal2_core, predict2_core, imgMTotal2_morphology, num_recom = 22): #level = 0 上对应的大小    
        """按照概率值推荐前num_recom个核 
        前22为红色 其他为绿色"""#object_recommend.
        Areas, Type_img, Contours_core = self.get_contours_and_feature_of_core(imgTotal2_core,predict2_core)
        Type_img = imgMTotal2_morphology
        
        startpointlist2, predict2, Sizepatch_small2 = self.get_startpointlist2()
        if len(imgTotal2_core)!= len(predict2):
            raise ValueError('len(imgTotal2_core)!=len(predict2)')
        # 后处理 去除覆盖的框框 和框框内的细胞核
        input_str = self.input_str
        ratio = self.ratio
        size_model2 = (input_str['model2_input'][0]*ratio/2,input_str['model2_input'][1]*ratio/2)
        size_recom = (input_str['model2_input'][0]*ratio,input_str['model2_input'][1]*ratio)
        size_dis = (96*ratio,96*ratio)
        near_index = []#临近框框index
        for index_point1,point1 in enumerate(startpointlist2): 
            near_point1 = [index_point1]
            for index_point2,point2 in enumerate(startpointlist2[index_point1+1:]):
                if np.abs(point1[0]-point2[0])<size_dis[0] and np.abs(point1[1]-point2[1])<size_dis[1]:
                    near_point1.append(index_point1+index_point2+1)
            if len(near_point1)!=1:
                near_index.append(near_point1)
        for item in near_index:
            max_predict2_index = item[0]
            max_predict2 = predict2[item[0]]
            for itemm in item:
                if predict2[itemm] > max_predict2:
                    max_predict2_index = itemm
                    max_predict2 = predict2[itemm]
            for itemm in item:
                if itemm != max_predict2_index:
                    predict2[itemm] = 0
                    Type_img[itemm] = 5#'Nothing'
                    Contours_core[max_predict2_index] += Contours_core[itemm]
                    Areas[max_predict2_index] += Areas[itemm]
                    Areas[itemm] = [0]
                    Contours_core[itemm] = []
       
         # 按照核面积/model2预测值从大到小排列【按类别顺序优先】
        Type_img = np.array(Type_img)
        num_Right = np.sum(Type_img==2) + np.sum(Type_img==1) + np.sum(Type_img== 0) #'Right'
        num_Black_group = np.sum(Type_img==3) + np.sum(Type_img==6) #'Black_group'
        num_False = np.sum(Type_img==4) + np.sum(Type_img==5)
        Index_whole = []
        Index_core_whole = []
        for type_img in [[2,1,0],[3],[6,4,5]]:
#            print(type_img)
            predict2_temp = [predict2[i]*(Type_img[i] in type_img) for i  in range(len(Areas))]
            predict2_temp = np.array(predict2_temp)
            num_one_type = np.sum(predict2_temp!=0)
            Areas_max_temp = [[np.max(Areas[i])*(Type_img[i] in type_img)] for i  in range(len(Areas))]
            Index_whole.append(np.argsort(predict2_temp,axis=0)[::-1][:num_one_type])
            num_one_type = min(num_one_type,np.sum(np.array(Areas_max_temp)!=0))
            Index_core_whole.append(np.argsort(np.array(Areas_max_temp),axis=0)[::-1][:num_one_type])
        
        Index_whole = np.vstack(Index_whole)
        Index_core_whole = np.vstack(Index_core_whole)
        Index_whole = np.hstack(Index_whole).tolist() 
        Index_core_whole = np.hstack(Index_core_whole) .tolist()
        Index_core_whole += list(set(Index_whole) - set(Index_core_whole))
        # 前22各index
        # 按照model2概率从大到小排列，优先排Right，然后是黑团团，然后是其他类型的
        Num_recom_black = 2
        Num_recom = min((len(predict2),num_recom))-Num_recom_black
        Index = Index_whole[:Num_recom]
        Index_core = Index_core_whole[:Num_recom]
        # 前20区域重合部分 加 各补上一部分 得到最终的 Index
        Index_union = list(set(Index) & set(Index_core))
        num_complet = Num_recom - len(Index_union)
        num_complet_model2 = int(num_complet/2)
        num_complet_model2_core = num_complet - num_complet_model2
        # 不够20的model2顺位补一半
        Index_complet = [x  for x in Index if x not in Index_union][:num_complet_model2]
        # 不够20的核顺位补一半【model2>0.9优先】
        Index_core_whole = [x  for x in Index_core_whole if x not in Index_union + Index_complet ]
        Index_core_complet = [x  for x in Index_core_whole if predict2[x]>0.9 and (Type_img[x] in [2,1,0])]
        Index_core_complet += [x for x in Index_core_whole if (predict2[x]<=0.9 and predict2[x]>0.8) and (Type_img[x] in [2,1,0])]
        Index_core_complet += [x  for x in Index_core_whole if predict2[x]>0.9 and (Type_img[x] in [3])]
        Index_core_complet += [x for x in Index_core_whole if (predict2[x]<=0.9 and predict2[x]>0.8) and (Type_img[x] in [3])]
        Index_core_complet += [x  for x in Index_core_whole if predict2[x]<=0.8 and (Type_img[x] in [2,1,0])]
        Index_core_complet += [x  for x in Index_core_whole if predict2[x]<=0.8 and (Type_img[x] in [3])]
        Index_core_complet += [x for x in Index_core_whole if Type_img[x] in [6,4,5]]
        Index_core_complet = Index_core_complet[:num_complet_model2_core]
        # 剩下两个黑团团
        Index_20 = Index_union + Index_complet + Index_core_complet
        Index_dark = [x for x in Index_whole if x not in (Index_20+Index_whole[:num_Right])][:Num_recom_black]
        Index_final = Index_union + Index_complet + Index_core_complet + Index_dark
        # model2 相关xml参数
        startpointlist_recom2 = [startpointlist2[index] for index in Index_final]
        Sizepatch_small_recom2 = [Sizepatch_small2[index] for index in Index_final]
        predict_recom2 = [str(Type_img[index]) + '_' + str(predict2[index][0]) for index in Index_final]
        contourslist_small_recom2 = Get_rectcontour(startpointlist_recom2,Sizepatch_small_recom2)
        # 框上色
        color2 = ['#ff0000'] * len(Index_union) + \
                 ['#ffff00'] * len(Index_complet) + \
                 ['#0000ff'] * len(Index_core_complet)+ \
                 ['#00ffff'] * len(Index_dark)
        # model2_core 相关xml参数          
        contourslist_core_recom = [Contours_core[index] for index in Index_final]
        predict2_core_recom = [Areas[index] for index in Index_final]
        annotation_Valid_flag = [Type_img[index] for index in Index_final]
        dictionary_core = [len(x) for x in contourslist_core_recom]
        # 将数组拉平
        contourslist_core_recom_flat = []
        predict2_core_recom_flat = []
        for index,b in enumerate(contourslist_core_recom):
            for indexx,c in enumerate(b):
                contourslist_core_recom_flat.append(c)
                predict2_core_recom_flat.append(str(annotation_Valid_flag[index])+'_'+str(predict2_core_recom[index][indexx]))
        # 核上色
        num_core = int(np.sum(dictionary_core))
        color2_core = ['#ff0000'] * num_core
        # 保存xml
        pathsave_xml = self.pathsave_xml
        filename = self.filename
        saveContours_xml([contourslist_small_recom2,contourslist_core_recom_flat],
                         [predict_recom2,predict2_core_recom_flat],
                         [color2,color2_core],
                         pathsave_xml + filename + '1.xml')
        return None
    