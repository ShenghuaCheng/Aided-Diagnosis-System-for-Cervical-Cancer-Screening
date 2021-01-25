# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 21:02:41 2019

@author: yujingya
"""
import sys
sys.path.append('../')
import os
import numpy as np
import cv2
from utils.parameter_set import Size_set,TCT_set
from matplotlib import pyplot as plt
from tqdm import tqdm
def Feature_map(feature,Num_x,Num_y,num_x,num_y):
    feature_map1 = []
    H = feature.shape[1]
    W = feature.shape[2]
    feature_map1 = np.zeros((Num_x*num_x*H,Num_y*num_y*W))
    count = 0
    for i in range(Num_x):
        for j in range(Num_y):
            for p in range(num_x):
                for q in range(num_y):
                    feature_map_temp = feature[...,1][count]
                    feature_map1[(i*num_x+p)*H:(i*num_x+p+1)*H,(j*num_y+q)*W:(j*num_y+q+1)*W] = feature_map_temp
                    count +=1
    return feature_map1   
          
def Predict_map1(predict1, Num_x, Num_y, num_x, num_y):
    predict_map1 = predict1.reshape((int(len(predict1)/num_y/num_x),num_x,num_y))
    for i in range(Num_x):
        temp = np.hstack(predict_map1[i*Num_y:(i+1)*Num_y])
        if i == 0:
            Predict_map1 = temp
        else:
            Predict_map1 = np.vstack((Predict_map1,temp))
    return Predict_map1

def Predict_map12(predict1, dictionary, predict2,Num_x,Num_y,num_x,num_y):
    predict12 = []
    for index, item in enumerate(dictionary):
        if item!=0:
            a = predict2[int(np.sum(dictionary[:index])):int(np.sum(dictionary[:index])+item)]
            a = a.max()
            predict12.append(a)
        else:
            predict12.append(predict1[index])
    predict12 = np.vstack(predict12)    
    predict12 = Predict_map1(predict12,Num_x,Num_y,num_x,num_y)
    return predict12
    
def Generate_heatmap(pathsave1, filename, pathsave2 = None):
    startpointlist = np.load(pathsave1 + filename + '_s.npy')
    predict1 = np.load(pathsave1 + filename + '_p.npy')
    feature = np.load(pathsave1 + filename + '_f.npy') 
    Num_y = np.sum(startpointlist[:,1]==startpointlist[0,1])
    Num_x = int(len(startpointlist)/Num_y)
    num_y = 3
    num_x = 5
    #predict1热图
    feature_map = Feature_map(feature,Num_x,Num_y,num_x,num_y)
    #predict1定位
    predict_map1 = Predict_map1(predict1, Num_x, Num_y, num_x, num_y)
    #predict12热图
    predict_map12 = []
    if pathsave2 != None:
        dictionary = np.load(pathsave2 + filename + '_dictionary.npy')  
        predict2 = np.load(pathsave2 + filename + '_p2.npy')
        predict_map12 = Predict_map12(predict1, dictionary, predict2,Num_x,Num_y,num_x,num_y)
    return feature_map,predict_map1,predict_map12



if __name__ == '__main__':
    # 1.参数设定
    pathfolder_set,  _ = TCT_set()
    for pathfolder in pathfolder_set['szsq_sdpc'][3:]:
        pathsave1 = 'F:/recom/model1_szsq540_model2_szsq990/' + pathfolder[11:]
        pathsave2 = pathsave1 + 'model2/'
        CZ = os.listdir(pathsave1)
        CZ = [cz[:-len('_s.npy')] for cz in CZ if '_s.npy' in cz]
        for filename in tqdm(CZ):
            feature_map,predict_map1,predict_map12 = Generate_heatmap(pathsave1, filename, pathsave2)
            np.save(pathsave1 + filename + '_fm.npy',feature_map)
            np.save(pathsave1 + filename + '_pm.npy',predict_map1)
            np.save(pathsave2 + filename + '_p2m.npy',predict_map12)