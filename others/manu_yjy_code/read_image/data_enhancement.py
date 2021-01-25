# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 15:58:18 2018

@author: yujingya
"""
import os
import numpy as np
from numpy import *
import random
import cv2
from skimage import  morphology
import scipy.ndimage as ndi
import math
import matplotlib.pyplot as plt
from skimage import exposure
"""
linear_trans:线性变换
"""
def linear_trans(img):
    imgflat = img.flatten()
    imgflat = imgflat[imgflat>0]
    # 线性变换
    threshold_low = 0
    threshold_high = np.argmax(np.bincount(imgflat))
    img = np.float32(img)
    if threshold_high <= threshold_low:
        print("Error:高阈值取值太小")
        return
    img_max = (img > threshold_high) * 255
    img_min = (img < threshold_low) * 0
    img_middle = (img >= threshold_low)*(img <= threshold_high)*1
    img_middle = ((255/ (threshold_high - threshold_low)) * (img - threshold_low)) * img_middle
    img = np.uint8(img_max + img_min + img_middle)
    #线性计算
    k = np.random.randint(96,104)/100
    b = np.random.randint(-4,5)
    img = np.add(np.multiply(k, img),b)
    img[img>255] = 255
    img[img < 0] = 0
    img = np.uint8(img)
    return img

"""
rotate:图片随机旋转+flip操作
"""
def rotate(img):
    angle = np.random.randint(0,4)*90
    height = img.shape[0]
    width = img.shape[1]
    if height == width:
        if angle%180 == 0:
            scale = 1
        elif angle%90 == 0:
            scale = float(max(height, width))/min(height, width)
        else:
            scale = math.sqrt(pow(height,2)+pow(width,2))/min(height, width)
        rotateMat = cv2.getRotationMatrix2D((width/2, height/2), angle, scale)
        img = cv2.warpAffine(img, rotateMat, (width, height))
    #flip操作
    n = np.random.randint(0, 3)
    if n != 2:
        img = np.flip(img, n)
    return img


def rotate_batch(imgbatch):
    """对图片和mask对进行同样的变换操做
    :param imgbatch:
    :return:
    """
    angle = np.random.randint(0,4)*90
    height = imgbatch[0].shape[0]
    width = imgbatch[0].shape[1]
    if height == width:
        if angle%180 == 0:
            scale = 1
        elif angle%90 == 0:
            scale = float(max(height, width))/min(height, width)
        else:
            scale = math.sqrt(pow(height,2)+pow(width,2))/min(height, width)
        rotateMat = cv2.getRotationMatrix2D((width/2, height/2), angle, scale)
        imgbatch = [cv2.warpAffine(img, rotateMat, (width, height)) for img in imgbatch]
    #flip操作
    n = np.random.randint(0,3)
    if n!= 2:
        imgbatch = [np.flip(img, n) for img in imgbatch]
    return imgbatch

"""
contrast:对比度变换 c 建议取值0.7-1.2
random.random()*0.5 + 0.7
"""
def contrast(img, c):   
    # 亮度就是每个像素所有通道都加上b   
    b = 0     
    rows, cols, chunnel = img.shape
    blank = np.zeros([rows, cols, chunnel], img.dtype) 
    # np.zeros(img1.shape, dtype=uint8)
    dst = cv2.addWeighted(img, c, blank, 1-c, b)
    return dst
"""
gamma_trans:gamma变换 gamma建议取值 0.6-1.6
"""
def gamma_trans(img):   
    gamma = random.random()*0.8+ 0.8
    img = exposure.adjust_gamma(img, gamma)                                          
    return img
"""
img_noise:图像加入噪声
"""
NOISE_NUMBER = 1000
def img_noise(img):
    height,weight,channel = img.shape  
    for i in range(NOISE_NUMBER):
        x = np.random.randint(0,height)
        y = np.random.randint(0,weight)
        img[x ,y ,:] = 255
    return img
"""
PCA Jittering:先计算RGB通道的均值和方差，进行归一化，然后在整个训练集上计算协方差矩阵，
进行特征分解，得到特征向量和特征值(这里其实就是PCA分解)，在分解后的特征空间上对特征值
做随机的微小扰动，根据下式计算得到需要对R,G,B扰动的值，把它加回到RGB上去，作者认为这样
的做法可以获得自然图像的一些重要属性，把top-1 error又降低了1%
建议a取值范围为(0-0.004)
random.random()*0.004)
"""
def PCA_Jittering(img,a):       
    img_size = img.size/3  
    #print(img.size,img_size)
    img1= img.reshape(int(img_size),3)
    img1 = np.transpose(img1)
    img_cov = np.cov([img1[0], img1[1], img1[2]])  #计算矩阵特征向量
    lamda, p = np.linalg.eig(img_cov)
    p = np.transpose(p)  #生成正态分布的随机数
    alpha1 = random.normalvariate(0,a)  
    alpha2 = random.normalvariate(0,a)  
    alpha3 = random.normalvariate(0,a)  
    v = np.transpose((alpha1*lamda[0], alpha2*lamda[1], alpha3*lamda[2])) #加入扰动  
    add_num = np.dot(p,v)   
    img2 = np.array([img[:,:,0]+add_num[0], img[:,:,1]+add_num[1], img[:,:,2]+add_num[2]])  
    img2 = np.swapaxes(img2,0,2)  
    img2 = np.swapaxes(img2,0,1)  
    return img2
"""
Gauss_edge:对输入的图片img的细胞边缘进行高斯滤波处理
"""
def Gauss_edge(img, ks=None): ##高斯模糊
    # 细胞边缘模糊整体图:
    img_edge = img.copy()
    sigmas = random.random()*1.5 + 0.5 #0.5-2
    img_edge_Gauss = cv2.GaussianBlur(img_edge, (int(6*np.ceil(sigmas)+1),int(6*np.ceil(sigmas)+1)), sigmas)
    # 得到细胞前景
    threColor = 8
    threVol = 1024
    wj1 = img.max(axis=2)
    wj2 = img.min(axis=2)
    wj3 = wj1 - wj2
    imgBin = wj3 > threColor
    imgBin = morphology.remove_small_objects(imgBin, min_size=threVol)
    imgBin = np.uint8(imgBin)
    # 去除孔洞
    kernel = np.ones((5,5),np.uint8)  
    imgBin = cv2.dilate(imgBin,kernel,iterations = 1)
    imgBin = cv2.erode(imgBin,kernel,iterations = 1)
    # 得到边缘部分大前景:
    kernel = np.ones((15,15),np.uint8)  
    imgBin_big = cv2.dilate(imgBin,kernel,iterations = 1)
    kernel = np.ones((40,40),np.uint8) 
    imgBin_small = cv2.erode(imgBin,kernel,iterations = 1)
    imgBin_edge_big = imgBin_big-imgBin_small
    # 得到边缘部分小前景:
    kernel = np.ones((5,5),np.uint8)  
    imgBin_big_temp = cv2.dilate(imgBin,kernel,iterations = 1)
    kernel = np.ones((5,5),np.uint8) 
    imgBin_small_temp = cv2.erode(imgBin,kernel,iterations = 1)
    imgBin_edge_small = imgBin_big_temp - imgBin_small_temp
    # 在小前景取某一小边缘:
    ind = np.flatnonzero(imgBin_edge_small.copy())
    ind = ind[np.random.randint(np.size(ind), size = random.choice([3,4,5]))]
    indRow, indCol = np.unravel_index(ind, np.shape(imgBin_edge_small))#取1-2个点
    imgmap = np.zeros(np.shape(imgBin_edge_small),np.uint8)
    for k in range(len(indRow)):
        imgmap[int(indRow[k]-40):int(indRow[k]+40),int(indCol[k]-40):int(indCol[k]+40)] = 1
        
    imgBin_edge = np.multiply(imgBin_edge_big,imgmap)
    sigmas = 5
    imgBin_edge = cv2.GaussianBlur(imgBin_edge, (int(6*np.ceil(sigmas)+1),int(6*np.ceil(sigmas)+1)), sigmas) 
    img_edge_Gauss[imgBin_edge==0] = 0
    img[imgBin_edge==1] = 0#原地操作
    img_new = img + img_edge_Gauss
    # 整体模糊让边缘不突兀
    sigmas = random.random()*0.1 + 0.1 #0.1-0.2
    img_new = cv2.GaussianBlur(img_new,(int(6*np.ceil(sigmas)+1),int(6*np.ceil(sigmas)+1)), sigmas)
    return img_new
"""
Gauss:对输入的图片img整体进行高斯滤波处理
"""
def Gauss(img): 
    # 整体模糊让边缘不突兀
    sigmas = random.random() + 0.1 #0.1-1.1
    img_new = cv2.GaussianBlur(img,(int(6*np.ceil(sigmas)+1),int(6*np.ceil(sigmas)+1)), sigmas)
    return img_new
"""
Sharp:对输入的图片img进行锐化
建议滤波器卷积核中间值：8.3-8.7
"""
def Sharp(img): 
    sigmas = random.random()*0.4 + 8.3 #8.3-8.7
    kernel = np.array([[-1, -1, -1], [-1, sigmas, -1], [-1, -1, -1]], np.float32)/(sigmas-8) #锐化
    img_new = cv2.filter2D(img, -1, kernel=kernel)       
    return img_new
"""
HSV_trans:对输入的图片img进行hsv空间的变换，h：色相，s：饱和度，v：亮度
若做GAN可以适当缩小或者关闭h的变换 
"""
def HSV_trans(img, h_change=1, s_change=1, v_change=1): # hsv变换
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.float64(hsv)
    # hsv[...,0] = 180
    if h_change != 0:  # random.random()
        k = random.random()*0.1 + 0.95 # 0.95-1.05
        b = random.random()*6 - 3 # -3/3
        hsv[...,0] = k*hsv[...,0] + b
        hsv[...,0][ hsv[...,0] <= 0] = 0
        hsv[...,0][ hsv[...,0] >= 180] = 180
    if s_change != 0:
        k =  random.random()*0.8 + 0.7  # 0.7-1.5
        b = random.random()*20 - 10  # -10/10
        hsv[...,1] = k*hsv[...,1] + b
        hsv[...,1][ hsv[...,1] <= 0] = 1
        hsv[...,1][ hsv[...,1] >= 255] = 255
    if  v_change != 0:
        k = random.random()*0.45 + 0.75#0.75-1.2
        b = random.random()*18 - 10#-10-8
        hsv[...,2] = k*hsv[...,2] + b
        hsv[...,2][ hsv[...,2] <= 0] = 1
        hsv[...,2][ hsv[...,2] >= 255] = 255
    hsv = np.uint8(hsv)
    img_new = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img_new
"""
RGB_trans:对输入的图片img进行RGB通道随机转换
"""
def RGB_trans(img):
    index = [i for i in range(3)]
    random.shuffle(index)
    img = np.stack((img[:,:,index[0]],img[:,:,index[1]],img[:,:,index[2]]),axis = 2)
    return img
"""
BinarySP:对输入的图片img取前景
level = 1时
threColor = 10
threVol = 1000
"""
def BinarySP(img, threColor = 10, threVol= 1000):
    wj1 = img.max(axis=2)
    wj2 = img.min(axis=2)
    wj3 = wj1 - wj2
    imgBin = wj3 > threColor
    imgBin = ndi.binary_fill_holes(imgBin)
    s = np.array([[0,1,0],[1,1,1],[0,1,1]], dtype=np.bool)
    imgBin = ndi.binary_opening(imgBin, s)#可去掉
    imgCon, numCon = ndi.label(imgBin)
    imgConBig = morphology.remove_small_objects(imgCon, min_size=threVol)
    imgBin = imgConBig > 0
    return imgBin


def szsq_normal(img_szsq):
    """深圳生强系统数据规范
    :param img_szsq:
    :return:
    """
    img_szsq_trans = exposure.adjust_gamma(img_szsq,0.6)
    hsv = cv2.cvtColor(img_szsq_trans, cv2.COLOR_BGR2HSV)
    hsv =  np.float64(hsv)
    h = hsv[...,0]*0.973 + 7
    s = hsv[...,1]*0.929 + 18
    v = hsv[...,2]*0.976 + 6
    h[h>255] = 255
    s[s>255] = 255
    v[v>255] = 255
    hsv[...,0] = h
    hsv[...,1] = s
    hsv[...,2] = v
    hsv = np.uint8(hsv)
    img_szsq_trans = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img_szsq_trans

##################################################################
if __name__  ==  '__main__':
    pathsrc = 'H:/yujingya/test/sample/'
    pathdst = 'H:/yujingya/test/sample_Binary/'
    listDir = os.listdir(pathsrc)
    level = 1
    for cz in listDir:
        print(cz)
        path = pathsrc+ cz 
        img = cv2.imread(path)
# =============================================================================
#         img_Sharp = Sharp(img_before.copy())
#         img_gamma = gamma_trans(img_Sharp.copy())
#         img_trans_S = HSV_trans(img_gamma.copy())
# 
#         img_Gauss = Gauss(img_before.copy())
#         img_gamma = gamma_trans(img_Gauss.copy())
#         img_trans_G = HSV_trans(img_gamma.copy())
#         
#         img_Sharp = Sharp(img_before.copy())
#         img_gamma = gamma_trans(img_Sharp.copy())
#         img_trans = HSV_trans(img_gamma.copy())
#         img_trans = Gauss(img_trans.copy())
#         
#         cv2.imwrite(pathdst + cz[:-4] + '.tif',np.concatenate((img,img_trans_S,img_trans_G,img_trans),axis = 1))
# =============================================================================
# =============================================================================
#         font=cv2.FONT_HERSHEY_SIMPLEX
#         for sigmas in range(850,870,1):
#             sigmas = sigmas/100
#             print(sigmas)
#             kernel = np.array([[-1, -1, -1], [-1, sigmas, -1], [-1, -1, -1]], np.float32)/(sigmas-8) #锐化
#             img_trans = cv2.filter2D(img_before.copy(), -1, kernel=kernel)
#             #img_trans = cv2.GaussianBlur(img_before,(int(6*np.ceil(sigmas)+1),int(6*np.ceil(sigmas)+1)), sigmas)
#             img_trans = cv2.putText(img_trans,str(sigmas),(220,50),font,2,(255,0,255),5)
#             img = np.concatenate((img,img_trans),axis = 1)
#         cv2.imwrite(pathdst + cz[:-4] + '.tif',img)
# =============================================================================
# =============================================================================
#         threColor = [6,8,10,12,14]
#         threVol = [50,600,650,700,750,800,850,900,950,1000,2000,4000]
#         font=cv2.FONT_HERSHEY_SIMPLEX
#         for Color in threColor:
#             for Vol in threVol:
#                 img_before = img.copy()
#                 if Vol == threVol[0]:
#                     img_after = img.copy()
#                 imgBin = BinarySP(img_before, Color, Vol)
#                 img_before[imgBin==0] = 0
#                 img_before = cv2.putText(img_before,str(Color)+'/'+str(Vol),(220,50),font,2,(0,255,0),5)
#                 img_after = np.concatenate((img_after,img_before),axis = 1)
#             if Color == threColor[0]:
#                 img_after1 = img_after
#             else:
#                 img_after1 = np.concatenate((img_after1,img_after),axis = 0)
#         cv2.imwrite(pathdst + cz[:-4] + '.tif',img_after1)
# =============================================================================
