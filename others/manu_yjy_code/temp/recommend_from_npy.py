# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 21:02:41 2019

@author: A-WIN10
"""
from function_set import *
import numpy as np
import os
import heapq


path = 'F:/recom_16/Shengfuyou_4th/2big/'
while(not os.path.exists(path+recommend_strategy)):
    os.mkdir(path+recommend_strategy)   
CZ = os.listdir(path)
filelist = [cz[:-6] for cz in CZ if '_s.npy' in cz]

for filename in filelist:#filename = filelist[0]
    startpointlist = np.load(path + filename + '_s.npy')
    imgMTotal = np.load(path + filename + '_p.npy')
    imgTotal = Split_into_small(imgTotal,startpointlist_split,sizepatch_predict_small)
    imgTotal = trans_Normalization(imgTotal,channal_trans_flag= True)
    #显示max(前5,)
    imgMTotal_combine_index = []
    for i in range(len(startpointlist)):
        temp  = imgMTotal[i*num:(i+1)*num].copy()
        index = np.argsort(np.array(temp),axis=0)[::-1][:max(5,np.sum(temp>0.5))]
        imgMTotal_combine_index.append(index)
     #推荐  
    if recommend_strategy == 'max':
        imgMTotal_combine = [max(imgMTotal[i*num:(i+1)*num])for i in range(len(startpointlist))]
             
    elif recommend_strategy == 'average_0.1':
        imgMTotal_combine = []  
        for i in range(len(startpointlist)):
            temp = imgMTotal[i*num:(i+1)*num]
            if max(temp) < 0.1:
                average = max(temp)
            else:
                average = np.average(temp[temp>=0.1])
            imgMTotal_combine.append(average )
    elif recommend_strategy == 'sum_0.9':
        imgMTotal_combine = []  
        for i in range(len(startpointlist)):
            temp = imgMTotal[i*num:(i+1)*num]
            if max(temp) < 0.9:
                Sum = max(temp)
            else:
                Sum = sum(temp[temp>=0.9])
            imgMTotal_combine.append(Sum)
            
    elif recommend_strategy == 'average_top3':
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
                
    elif recommend_strategy == '411':
        imgMTotal_combine = []
        for i in range(len(startpointlist)):
            score = heapq.nlargest(3,imgMTotal[i*num:(i+1)*num]) 
            score1,score2,score3 = np.sort(score)
            imgMTotal_combine.append((4*score1+score2+score3)/6)  
       
    
    Index = np.argsort(np.array(imgMTotal_combine),axis=0)[::-1][:num_recom]
    start_sizepatch = []
    start_sizepatch_small = []
    label_annotation = []
    label_annotation_small = []    
    predict_color = []
    predict_color1 = []
    for item in Index:
        item = int(item)
        start_sizepatch.append(startpointlist[item])
        label_annotation.append(imgMTotal_combine[item])
        index = imgMTotal_combine_index[item]
        predict_color.append('#00ff00')
        for itemm in index:
            itemm = itemm[0]
            start_sizepatch_small .append((startpointlist[item][0]+startpointlist_split[itemm][0],startpointlist[item][1]+startpointlist_split[itemm][1]))
            label_annotation_small.append(imgMTotal[item*num+itemm])
            predict_color1.append('#ff0000')
    contourslist = Get_rectcontour(start_sizepatch,sizepatch)
    contourslist_small = Get_rectcontour(start_sizepatch_small,sizepatch_small1)
    saveContours_xml([contourslist,contourslist_small],[label_annotation,label_annotation_small],[predict_color,predict_color1],path + recommend_strategy+'/'+ filename + '.xml')

# =============================================================================
# 
# Predict2 = []
#     for index, item in enumerate(count):
#          if item!=0:
#             a = predict2[np.sum(count[:index]):np.sum(count[:index])+item]
#             Predict2.append(a) 
#          else:
#             Predict2.append(0) 
#             a=np.array(Predict2)
# =============================================================================
