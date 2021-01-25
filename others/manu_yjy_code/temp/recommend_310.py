# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 21:02:41 2019

@author: A-WIN10
"""
from predict_310_function import *
import numpy as np
import os
import heapq

pathfolder_xml = 'H:/recom_30/our/Positive/Shengfuyou_4th/310_test/ranslation200_200/w_12our_135/'
CZ = os.listdir(pathfolder_xml)
CZ = [cz for cz in CZ if '_m.npy' in cz]
# 参数设定

level = 0
levelratio = 4
flag ='1'
num_recom = 16
recommend_strategy = '111'
#20x
sizepatch = (int(1216*1.2/0.6),int(1936*1.2/0.6))#2432 3872
sizepatch_small = (int(512/0.6),int(512/0.6))
widthOverlap = (int(110/0.6),int(110/0.6))
#10x
sizepatch_predict = (int(1216*1.2),int(1936*1.2))#1459 2323
sizepatch_predict_small = (512,512)
widthOverlap_predict = (40,60)#3*5
#widthOverlap_predict = (197,150)#3*5
#计算startpointlist_split
sizepatch_predict_y,sizepatch_predict_x = sizepatch_predict
sizepatch_predict_small_y,sizepatch_predict_small_x = sizepatch_predict_small
widthOverlap_predict_y,widthOverlap_predict_x = widthOverlap_predict
numblock_y = np.around(( sizepatch_predict_y - sizepatch_predict_small_y) / (sizepatch_predict_small_y - widthOverlap_predict_y)+1)
numblock_x = np.around(( sizepatch_predict_x - sizepatch_predict_small_x) / (sizepatch_predict_small_x - widthOverlap_predict_x)+1)
startpointlist_split = []
for j in range(int(numblock_x)):
    for i in range(int(numblock_y)):
        startpointtemp = (i*(sizepatch_predict_small_y- widthOverlap_predict_y),j*(sizepatch_predict_small_x-widthOverlap_predict_x))
        startpointlist_split.append(startpointtemp)
if flag =='0':
    startpointlist_split = np.uint16(np.array(startpointlist_split)*2)
elif flag =='1':
    startpointlist_split = np.uint16(np.array(startpointlist_split)/0.6)
num = int(numblock_y*numblock_x)
for cz in CZ:
    filename = cz[:-6]
    
    startpointlist = np.load(pathfolder_xml + filename + '_s.npy')
    imgMTotal = np.load(pathfolder_xml + filename + '_m.npy')
    
    #推荐
    if recommend_strategy == '100':
        imgMTotal_combine = [max(imgMTotal[i*num:(i+1)*num])for i in range(len(startpointlist))]
        imgMTotal_combine_index = [np.where(imgMTotal[i*num:(i+1)*num] == max(imgMTotal[i*num:(i+1)*num]))[0] for i in range(len(startpointlist))]
    elif recommend_strategy == '111':
        imgMTotal_combine = []
        for i in range(len(startpointlist)):
            score = heapq.nlargest(3,imgMTotal[i*num:(i+1)*num]) 
            score1,score2,score3 = np.sort(score)
            if (score1-score2)>0.5:
                imgMTotal_combine.append(score1)
            elif (score1-score3)>0.5:
                imgMTotal_combine.append(np.average([score1,score2]))
            else :
                imgMTotal_combine.append(np.average(score))  
        imgMTotal_combine_index = [ map(imgMTotal[i*num:(i+1)*num].tolist().index, heapq.nlargest(3,imgMTotal[i*num:(i+1)*num])) for i in range(len(startpointlist))]
    elif recommend_strategy == '411':
        imgMTotal_combine = []
        for i in range(len(startpointlist)):
            score = heapq.nlargest(3,imgMTotal[i*num:(i+1)*num]) 
            score1,score2,score3 = np.sort(score)
            imgMTotal_combine.append((4*score1+score2+score3)/6)  
        imgMTotal_combine_index = [ map(imgMTotal[i*num:(i+1)*num].tolist().index, heapq.nlargest(3,imgMTotal[i*num:(i+1)*num])) for i in range(len(startpointlist))]
    Index = np.argsort(np.array(imgMTotal_combine),axis=0)[::-1][:num_recom]
    start_sizepatch = []
    start_sizepatch_small = []
    label_annotation = []
    label_annotation_small = []    
    for item in Index:
        item = int(item)
        start_sizepatch.append(startpointlist[item])
        label_annotation.append(imgMTotal_combine[item])
        index = imgMTotal_combine_index[item]
        for itemm in index:
            start_sizepatch_small .append((startpointlist[item][0]+startpointlist_split[itemm][0],startpointlist[item][1]+startpointlist_split[itemm][1]))
            label_annotation_small.append(imgMTotal[item*num+itemm])
    contourslist = Get_rectcontour(start_sizepatch,sizepatch)
    contourslist_small = Get_rectcontour(start_sizepatch_small,sizepatch_small)
    saveContours_xml([contourslist,contourslist_small],[label_annotation,label_annotation_small],['#00ff00','#ff0000'],pathfolder_xml +'predict_16_'+recommend_strategy+'/'+ filename + '.xml')
