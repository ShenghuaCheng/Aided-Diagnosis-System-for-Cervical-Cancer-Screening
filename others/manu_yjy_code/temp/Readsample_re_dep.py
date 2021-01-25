# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 15:47:03 2018
@author: yujingya
"""
from __future__ import print_function
from glob import glob
import numpy as np
import multiprocessing.dummy as multiprocessing
import cv2
import random
from keras.utils.np_utils import to_categorical
from data_enhancement import *
import math
def img_enhance(img,img_mask):
    """对图片进行变换
    :param img: 输入需要变换的图片
    :param img_mask: 输入需要变换的图片对应的mask
    :return: 返回变换后的图片和mask
    """
    n = np.random.randint(0,2)
    if n==0:
        img = Sharp(img)
    else:
        img = Gauss(img)  
    img = HSV_trans(img)    
    img = RGB_trans(img)
    #旋转
    angle = np.random.randint(0,4)*90
    height = width = 512
    rotateMat = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
    img = cv2.warpAffine(img, rotateMat, (width, height))
    n = np.random.randint(0,3)
    if n!= 2:
        img = np.flip(img, n)   
    if img_mask !=[]:
        height =  mask_shape[0]
        width = mask_shape[1]
        rotateMat = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
        img_mask = cv2.warpAffine(img_mask, rotateMat, (width, height))
        if n!= 2:
            img_mask = np.flip(img_mask, n)  
        img_mask = to_categorical(img_mask,2)
    return  img,img_mask

def GetImg(dirList, ith,
           resize_shape = (768,768),#(2188,3484)
           crop_shape = (768,768),#(2188,3484)
           output_shape = (512,512), #(1216,1936)
           enhance_flag = False,
           path_mask = None, 
           mask_shape = None,#mask_shape = (38,61)
           local = None):
    img_mask = []
    dirlist = dirList[ith]
    img = cv2.imread(dirlist)
    size = img.shape[0]
    if size in [1228,1024,512]:
        local_x = 0
        local_y = 0
        img = cv2.resize(img,output_shape[::-1]) 
        if path_mask != None:
            img_mask = np.zeros(mask_shape,dtype = np.uint8)
    elif size in [1843,1536,768,4376,5252,2188]:
        img = cv2.resize(img,resize_shape[::-1])
        if resize_shape != crop_shape:
            x = int((resize_shape[0]-crop_shape[0])/2)
            y = int((resize_shape[1]-crop_shape[1])/2)
            img = img [x:x+crop_shape[0],y:y+crop_shape[1]]
        if local!= None:
            local_x =  local[ith][0]
            local_y =  local[ith][1]
        else:
            local_x = np.random.randint(0, resize_shape[0] - output_shape[0])
            local_y = np.random.randint(0, resize_shape[1] - output_shape[1])
        img = img[local_x: local_x + output_shape[0], local_y: local_y + output_shape[1]]
        if path_mask != None:
            if ('ASCUS' in dirlist or 'HL' in dirlist):
                img_mask = cv2.imread(path_mask[0] + dirlist.split('/')[-1],0)
                img_mask = cv2.resize(img_mask, resize_shape[::-1])
                if resize_shape != crop_shape:
                    img_mask = img_mask[x:x+crop_shape[0],y:y+crop_shape[1]]
                img_mask = img_mask[local_x: local_x + output_shape[0], local_y: local_y + output_shape[1]]
                img_mask = cv2.resize(img_mask, mask_shape[::-1])
                img_mask[img_mask!=0] = 1
            else:
                img_mask = np.zeros(mask_shape,dtype = np.uint8)
    else:
        raise ValueError('error!__imgsize = '+ str(size))
    if enhance_flag :
       img, img_mask = img_enhance(img,img_mask,path_mask = None)
    return img, img_mask,(local_x,local_y)


def GetImgMultiprocess(dirList, 
                       resize_shape = (768,768),#(2188,3484)
                       crop_shape = (768,768),#(2188,3484)
                       output_shape = (512,512), #(1216,1936)
                       enhance_flag = False,
                       path_mask = None, 
                       mask_shape = (16,16),# (38,61)
                       local = None):
    pool = multiprocessing.Pool(20)
    img_and_img_maskTotals = []
    for ith in range(len(dirList)):
        img_and_img_mask = pool.apply_async(GetImg, args=(dirList, ith,resize_shape,crop_shape,output_shape,enhance_flag,path_mask, mask_shape,local))
        img_and_img_maskTotals .append(img_and_img_mask)
    pool.close()
    pool.join()
    imgTotaltest = np.array([x.get()[0] for x in img_and_img_maskTotals])
    img_maskTotaltest  = np.array([x.get()[1] for x in img_and_img_maskTotals])
    local  = np.array([x.get()[2] for x in img_and_img_maskTotals])
    return imgTotaltest,img_maskTotaltest,local
        
def trans_Normalization(imgTotal,
                        channal_trans_flag):
    """Normalization for channel_last imgs
    :param imgTotal:
    :param channal_trans_flag: 
    :return: imgs after normalization
    """
    if channal_trans_flag is None:
        raise ValueError('The channal_trans_flag should not be "None".'
                         ' Please check wether need to trans channel older')
    if channal_trans_flag:
        imgTotal = imgTotal[...,::-1]
    imgTotal = np.float32(imgTotal) / 255.
    imgTotal = imgTotal - 0.5
    imgTotal = imgTotal * 2.  
    return imgTotal

def region_proposal(heatmap, output_shape,img=None, threshold=0.2,ratio=0.2):
    img_prop = []
    heatmap = cv2.resize(heatmap, output_shape[::-1])
    w_hm, h_hm = heatmap.shape
    prob_max = np.max(heatmap)
    _, mask = cv2.threshold(heatmap, min(threshold,prob_max * ratio), prob_max, cv2.THRESH_BINARY)
    mask[mask > 0] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))  # 定义结构元素
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 开运算 先缩后涨
    _, mask_c, _ = cv2.findContours(np.uint8(mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    local = []
    for c in mask_c:
        c_lt = np.array([np.min(c[...,1]), np.min(c[...,0])])
        c_rb = np.array([np.max(c[...,1]), np.max(c[...,0])])
        c_delta = c_rb-c_lt
        if c_delta.max()>128:
            times = np.round(c_delta/128)
            times = times.astype(np.int)+1
            for i in range(times[0]):
                for j in range(times[1]):
                    local.append(tuple(c_lt+[i*128, j*128]))
        else:
            local.append((np.mean(c[..., 1]), np.mean(c[..., 0])))

    if np.any(img != None):
        mask = np.zeros((w_hm, h_hm))
        img_prop = np.uint8((img / 2 + 0.5) * 255)
        for c in mask_c:
            hull = cv2.convexHull(c)  # 凸包
            img_prop = cv2.polylines(img_prop, [hull], True, (0, 255, 0), 2)  # 绘制凸包
            mask = cv2.fillPoly(mask, [hull], 255)
    return img_prop, mask, local

def IOU(img_maskTotaltest,imgMTotal,threshold=0.5):
    Iou = []
    for i in range(len(img_maskTotaltest)):
        answer =  img_maskTotaltest[i].copy()
        predict = imgMTotal[i][...,1].copy()
        answer = cv2.resize(answer,predict.shape[::-1])
        predict[predict<=threshold] = 0 
        predict[predict>threshold] = 1 
        mask_temp = answer+predict
        if np.sum(mask_temp) !=0:
            Iou.append(np.sum(mask_temp==2)/np.sum(mask_temp!=0)) 
        else:
            Iou.append(1)    
    return np.average(Iou)
      
class Readsample_for_list:
    def __init__(self,path,classes):
        self.path = path
        self.classes = classes

    def Get_filelist(self):
        path = self.path
        classes = self.classes 
        for i in range(len(path)):
            exec ("filelist_train%s = []"%i)
            for m in range(len(classes)):
                filelist = glob(path[i] + classes[m] + '/*.tif')
                exec ("filelist_train%s.append(filelist)"%i)
            exec ("self.filelist_train%s = filelist_train%s"%(i,i))

    def Get_dirlist(self, Vol):
        path = self.path
        classes = self.classes
        dirlist_train = []
        for n in range(len(classes)):
            vol = int(Vol[n])
            for i in range(len(path)):
                exec("dirlist_train+= random.sample(self.filelist_train%s[n], vol)"%i)
        return dirlist_train
    
    
