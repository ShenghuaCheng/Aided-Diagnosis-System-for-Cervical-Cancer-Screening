# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 00:43:20 2018
@author: yujingya
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
from matplotlib import pyplot as plt
from model_re_dep import Resnet_atrous as RA
from ResNetClassification import ResNet as Model2
from ResNetClassification_79_multiclasses import ResNet as Model2_multi
from keras.utils import multi_gpu_model 
import tensorflow as tf
import time
import numpy as np
import cv2
from openslide import OpenSlide
from predict_310_function import *
from matplotlib import pyplot as plt
# 1:positive 0:negative
def Startpointlist_split(sizepatch_predict,sizepatch_predict_small, widthOverlap_predict):
    sizepatch_predict_y,sizepatch_predict_x = sizepatch_predict
    sizepatch_predict_small_y,sizepatch_predict_small_x = sizepatch_predict_small
    widthOverlap_predict_y,widthOverlap_predict_x = widthOverlap_predict
       
    numblock_y = np.around(( sizepatch_predict_y - sizepatch_predict_small_y) / (sizepatch_predict_small_y - widthOverlap_predict_y)+1)
    numblock_x = np.around(( sizepatch_predict_x - sizepatch_predict_small_x) / (sizepatch_predict_small_x - widthOverlap_predict_x)+1)
    startpointlist = []
    for j in range(int(numblock_x)):
        for i in range(int(numblock_y)):
            startpointtemp = (i*(sizepatch_predict_small_y- widthOverlap_predict_y),j*(sizepatch_predict_small_x-widthOverlap_predict_x))
            startpointlist.append(startpointtemp)
    return startpointlist

if __name__ == '__main__':
    # 参数设定
# =============================================================================
#     level = 0
#     levelratio = 2
#     flag ='0'
#     #20x
#     sizepatch = (int(1216*1.2*2),int(1936*1.2*2))#2918 4646
#     sizepatch_small = (int(512*2),int(512*2))
#     widthOverlap = (int(110*2),int(110*2))
#     #10x
#     sizepatch_predict = (int(1216*1.2),int(1936*1.2))#1459 2323
#     sizepatch_predict_small = (512,512)
#     widthOverlap_predict = (40,60)#3*5
#     pathfolder_svs = 'H:/TCTDATA/tongji_1th/Positive/20180419/'
#     pathfolder_xml = 'H:/recom_30/tongji_1th/Positive/310_test/20180419/w_1our_55/'
#     pathweight = 'H:/weights/model1_55_3d1_our.h5'
# =============================================================================
    level = 0
    levelratio = 4
    flag ='1'
    strategy = 'max'
    model2_type = '2'
    #10x
    sizepatch = (1216,1936)#2432 3872
    sizepatch_small = (512,512)
    widthOverlap = (120,120)
    #predict
    sizepatch_predict = (1216,1936)#1459 2323
    sizepatch_predict_small = (512,512)
    widthOverlap_predict = (160,156)#3*5
    num = 15
    #widthOverlap_predict = (197,150) #4*6
    #20x model2
    sizepatch_small2 = (128,128)
    sizepatch_predict_small2 = (256,256)
    pathfolder_svs = 'H:/TCTDATA/our/10x/Positive/ShengFY-P-L240 (origin date)/'#ShengFY-P-L240 (origin date)
    pathfolder_xml = 'H:/recom_30/model12/10x/ShengFY-P-L240 (origin date)/'
    # 建立并行预测模型
    gpu_num = 1
    model1 = RA(input_shape = (512,512,3))
    model1.load_weights('H:/weights/model1_340_3d1_3d2_our_10x_adapt.h5',by_name=True)
    model1.load_weights('H:/weights/model1_340417_3d1_3d2_our_10x_adapt.h5',by_name=True)
    model2 = Model2(input_shape = (256,256,3))
    model2.load_weights('H:/weights/model2_670_3d1_3d2_our_10x_adapt.h5')#

    # 获取操作对象列表
    CZ = os.listdir(pathfolder_svs)
    CZ = [cz for cz in CZ if '.svs' in cz]
    testfile240  = ['1149531.svs',
                     '1154896.svs',
                     '1159715.svs',
                     '1161421.svs',
                     '1161645.svs',
                     '1162026.svs',
                     '1179935.svs',
                     '1179944.svs',
                     '1159977.svs',
                     '1159646.svs',
                     '1162017.svs',
                     '1136007.svs',
                     '1161688.svs',
                     '1161394.svs',
                     '1179435.svs',
                     '1179383.svs',
                     '1179676.svs',
                     '1179551.svs',
                     '1180006.svs',
                     '1159936.svs',
                     '1150002.svs']
    testfile2018 = ['1160655.svs',
                     '1163605.svs',
                     '1179380.svs',
                     '1169519.svs',
                     '1163234.svs',
                     '1166950.svs',
                     '1166803.svs',
                     '1168804.svs',
                     '1179415.svs']
    for cz in CZ[0:1]:
        filename = cz[:-4]
        pathsvs = pathfolder_svs + cz
        print(pathsvs)
        ors = OpenSlide(pathsvs)
        # 还原拍摄过程的小图片并读入
# =============================================================================
#         start = time.time()
#         startpointlist = Get_startpointlist(ors, level, levelratio, sizepatch, widthOverlap,flag)      
#         imgTotal = Get_predictimgMultiprocess(startpointlist, ors, level, sizepatch, sizepatch_predict)
#         end= time.time()
#         print('还原拍摄过程的小图片并读入_耗时>>>>'+str(end -start)[0:5] + 's' )
#         print('还原拍摄过程的小图片并读入_耗时>>>>'+str(end -start)[0:5] + 's' )
#         #将拍摄小图拆分成适合网络大小的小图
#         start = time.time()
#         imgTotal,num,startpointlist_split = Split_into_small(imgTotal,sizepatch_predict,sizepatch_predict_small, widthOverlap_predict)
#         imgTotal = trans_Normalization(imgTotal)
#         #获取512*512小图起始坐标点
#         startpointlist_split = np.uint16(np.array(startpointlist_split))
#         end = time.time()
#         print('将拍摄小图拆分成适合网络大小的小图并归一化_耗时>>>>'+str(end -start)[0:5] + 's' )
#         # model1开始预测
#         start = time.time()
#         imgMTotal1 = model1.predict(imgTotal, batch_size = 32*gpu_num, verbose=1)
#         end = time.time()
#         predict1 = imgMTotal1[0].copy()
#         feature = imgMTotal1[1].copy()
#         print('model1开始预测_耗时>>>>'+str(end -start)[0:5] + 's' )
#         # 保存model1预测文件
#         np.save(pathfolder_xml + filename + '_s.npy',startpointlist)
#         np.save(pathfolder_xml + filename + '_p.npy',predict1)
#         np.save(pathfolder_xml + filename + '_f.npy',feature) 
# =============================================================================
        ######################################################################
        startpointlist = np.load(pathfolder_xml + filename + '_s.npy')
        predict1 = np.load(pathfolder_xml + filename + '_p.npy')
        feature = np.load(pathfolder_xml + filename + '_f.npy') 
        # 计算model1预测小块起始坐标
        startpointlist_split = Startpointlist_split(sizepatch_predict,sizepatch_predict_small, widthOverlap_predict)
        startpointlist_split = np.uint16(np.array(startpointlist_split))
        startpointlist_1 = []
        for i in range(len(startpointlist)):
            for j in range(num):
                startpointlist_1.append((startpointlist[i][0]+startpointlist_split[j][0],startpointlist[i][1]+startpointlist_split[j][1]))
        # 返回model1定位结果
        startpointlist_2 = []
        count = []
        for index, item in enumerate(feature):
            if predict1[index] > 0.5:
                _, _, Local_3 = region_proposal(item[...,1])
                count.append(len(Local_3))
                for x in Local_3:
                    x = np.uint16(np.array(x))
                    startpointlist_2.append((startpointlist_1[index][0]+x[1]-int(sizepatch_small2[0]/2),startpointlist_1[index][1]+x[0]-int(sizepatch_small2[1]/2))) 
            else:
                count.append(0)
        # 为model2读入定位结果图并归一化
        imgTotal2 = Get_predictimgMultiprocess(startpointlist_2, ors,0,sizepatch_small2,sizepatch_predict_small2)
        imgTotal2 = trans_Normalization(imgTotal2)
        # model2开始预测
        predict2 = model2.predict(imgTotal2,batch_size = 8*gpu_num,verbose=1)
        if model2_type == 'multi':
            predict2_class_color = {0: '#ff0000', 1: '#00ff00', 2: '#f0f000'}
            predict2_color = [predict2_class_color[np.argsort(np.array(predict2[i][1:]),axis=0)[-1]] for i in range(len(predict2))]
            predict2 = predict2[...,1]*3 + predict2[...,2]*2 + predict2[...,3]
            predict2 = np.expand_dims(predict2,axis = 1)
        elif model2_type == '2':
            predict2_color = ['#00ff00' for i in range(len(predict2))] 
        # 合并所有结果  推荐最终一定个数个大块
        start = time.time()
        startpointlist_12 = startpointlist_1.copy()
        predict_12 = predict1.copy().tolist()
        Sizepatch_small12 = count.copy()
        Predict_color12 = count.copy()
        for index, item in enumerate(count):
            Sizepatch_small12[index] = sizepatch_small
            Predict_color12[index] = '#0000ff'
            if item!=0:
                a = predict2[int(np.sum(count[:index])):int(np.sum(count[:index])+item)]
                b = startpointlist_2[int(np.sum(count[:index])):int(np.sum(count[:index])+item)]
                a = a.tolist()
                predict_12[index] = a
                startpointlist_12[index] = b
                Sizepatch_small12[index] = [sizepatch_small2]*len(a)
                c = predict2_color[int(np.sum(count[:index])):int(np.sum(count[:index])+item)]
                Predict_color12[index] = c
        imgMTotal_combine = []
        for i in range(len(startpointlist)):
            predict_12_temp = np.vstack(predict_12[i*num:(i+1)*num])
            score = np.sort(predict_12_temp,axis= 0)
            if strategy == '0.9':
                if score[-1]<0.9:
                    imgMTotal_combine.append(score[-1])
                else:
                    score[score<0.9]=0
                    imgMTotal_combine.append(np.sum(score))
            elif strategy == 'max':
                imgMTotal_combine.append(score[-1])
      
        
        Index = np.argsort(np.array(imgMTotal_combine),axis=0)[::-1][:num_recom]

        start_sizepatch = []
        label_annotation = []
        predict_color = []
        start_sizepatch_small12 = []
        sizepatch_small12 = []
        label_annotation_small12 = []
        predict_color12 = []
        for item in Index:
            item = int(item)
            start_sizepatch.append(startpointlist[item])
            label_annotation.append(imgMTotal_combine[item])
            predict_color.append('#ff0000')
            predict_12_temp = np.vstack(predict_12[item*num:(item+1)*num])
            startpointlist_12_temp = np.vstack(startpointlist_12[item*num:(item+1)*num])
            Sizepatch_small12_temp = np.vstack(Sizepatch_small12[item*num:(item+1)*num])
            predict_color12_temp   = np.hstack(Predict_color12[item*num:(item+1)*num])
            index = np.argsort(np.array(predict_12_temp),axis=0)[::-1]
            if strategy == '0.9':
                if predict_12_temp[index[0]]<0.9:
                    start_sizepatch_small12.append(startpointlist_12_temp[index[0]][0])
                    label_annotation_small12.append(predict_12_temp[index[0]][0])
                    sizepatch_small12.append(Sizepatch_small12_temp[index[0]][0])
                    predict_color12.append(predict_color12_temp[index[0]][0])
                else:
                    i=0
                    while (predict_12_temp[index[i]]>=0.9):
                        start_sizepatch_small12.append(startpointlist_12_temp[index[i]][0])
                        label_annotation_small12.append(predict_12_temp[index[i]][0])
                        sizepatch_small12.append(Sizepatch_small12_temp[index[i]][0])
                        predict_color12.append(predict_color12_temp[index[i]][0])
                        i+=1
                        if i == len(predict_12_temp):
                            break
            elif strategy == 'max':    
                start_sizepatch_small12.append(startpointlist_12_temp[index[0]][0])
                label_annotation_small12.append(predict_12_temp[index[0]][0])
                sizepatch_small12.append(Sizepatch_small12_temp[index[0]][0])
                predict_color12.append(predict_color12_temp[index[0]][0])
                i=1
                while (predict_12_temp[index[i]]>=0):
                    start_sizepatch_small12.append(startpointlist_12_temp[index[i]][0])
                    label_annotation_small12.append(predict_12_temp[index[i]][0])
                    sizepatch_small12.append(Sizepatch_small12_temp[index[i]][0])
                    predict_color12.append(predict_color12_temp[index[i]][0])
                    i+=1
                    if i == len(predict_12_temp):
                        break
                        
        contourslist = Get_rectcontour(start_sizepatch,sizepatch)
        contourslist_small12 = Get_rectcontour(start_sizepatch_small12,sizepatch_small12)
        #saveContours_xml([contourslist,contourslist_small12],[label_annotation,label_annotation_small12],[predict_color,predict_color12],pathfolder_xml + filename + '.xml')
        saveContours_xml([contourslist_small12],[label_annotation_small12],[predict_color12],pathfolder_xml + filename + '.xml')
        
        end = time.time()
        print('开始推荐区域_耗时>>>>'+str(end -start)[0:5] + 's' )
# =============================================================================
#     testfile  = ['1615888 2226197.svs',
#                  '1615953 2226091.svs',
#                  '1615902 2226204.svs',
#                  '1615959 2226152.svs',
#                  '1615919 2226238.svs',
#                  '1157108 0893020.svs',
#                  '1165441 0893036.svs',
#                  '1615512 2226206.svs',
#                  '1157240 0893023.svs',
#                  '1164444 0893061.svs']
# =============================================================================
