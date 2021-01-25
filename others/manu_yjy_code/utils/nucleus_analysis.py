# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: Screening
@File: nucleus_analysis.py
@Date: 2019/4/17 
@Time: 11:34
@Desc:
'''
import os
import numpy as np
from numpy import linalg
import csv
import cv2
import matplotlib.pyplot as plt
from skimage import measure, morphology
import scipy.ndimage as ndi


class nucSet():
    def __init__(self, img_patch, img_preds):

        img_masks = img_preds['masks']
        img_rois = img_preds['rois']
        if len(img_masks):
            img_contours = get_contours(img_masks)
            valid_index = get_focus_area(img_rois)
        else:
            img_contours = []
            valid_index = []

        self.img_patch = img_patch
        self.img_masks = img_masks
        self.img_rois = []
        self.img_contours = []
        self.img_rois = img_rois
        self.img_contours = img_contours
        self.valid_index = valid_index

    def __del__(self):
        pass

    def density(self, diameter=240):  # 中层鳞状上皮细胞典型直径值   240为经验值
        length = len(self.img_contours)

        # 计算所有轮廓的中心点坐标——即细胞核中心
        centers = np.zeros((length, 1, 2))
        for i in range(length):
            centers[i] = self.img_contours[i].mean(axis=0).reshape(1, 2)

        # 计算每个细胞核周边的细胞核数目
        density = np.zeros(length)
        for i in range(length):
            temp = centers - centers[i]
            distance = linalg.norm(temp, axis=2).reshape(length, )
            # 保留满足距离条件的点
            distance = distance[distance <= diameter]
            distance = distance[distance >= 10]
            # 计算高斯密度
            density[i] = GaussianFuction(distance, diameter // 2).sum()
        return density

    def WeighedFeature(self, diameter=50):
        length = len(self.img_contours)

        # 计算所有轮廓的中心点坐标——即细胞核中心
        centers = np.zeros((length, 1, 2))
        for i in range(length):
            centers[i] = self.img_contours[i].mean(axis=0).reshape(1, 2)

        # 计算每个细胞核周边的细胞核数目
        WeighedArea = np.ones(length) * 1000
        WeighedStaindegree = np.ones(length) * 1000
        for i in range(length):
            temp = centers - centers[i]
            distance = linalg.norm(temp, axis=2).reshape(length, )
            # 保留满足距离条件的点
            index = np.where(distance <= 2 * diameter)  # 满足条件的下标
            index = index[0]
            index = index[index != i]
            distance = distance[distance <= 2 * diameter]
            distance = distance[distance >= 1]
            GaussianVector = np.zeros(len(distance))
            GaussianVector = GaussianFuction(distance, diameter // 2)
            AroundArea = np.zeros(len(distance))
            AroundStaindegree = np.zeros(len(distance))
            for j in range(len(distance)):  # 满足条件的细胞
                nucluei = nucleus(self.img_patch, self.img_masks[:, :, index[j]], self.img_rois[index[j]], self.img_contours[index[j]])
                area = nucluei.area()
                stainDegree = nucluei.stainDegree()
                AroundArea[j] = area
                AroundStaindegree[j] = stainDegree
            WeighedArea[i] = np.dot(GaussianVector, AroundArea)
            WeighedStaindegree[i] = np.dot(GaussianVector, AroundStaindegree)
        return WeighedArea, WeighedStaindegree

    def extractFeature(self):
        dataset = []
        if not len(self.img_contours):
            self.dataset = dataset
            return len(dataset)
        density = self.density(240)
        WeighedArea, WeighedStaindegree = self.WeighedFeature(50)
        # for i in range(len(self.ROIs)):
        for i in self.valid_index:
            nucluei = nucleus(self.img_patch, self.img_masks[:, :, i], self.img_rois[i], self.img_contours[i])
            feature = {}
            feature['area'] = nucluei.area()  # 面积
            feature['perimeter'] = nucluei.perimeter()  # 周长
            feature['convexHullArea'] = nucluei.convexHullArea()  # 凸包面积
            feature['eccentricity'] = nucluei.eccentricity()  # 离心率
            feature['stainDegree'] = nucluei.stainDegree()  # 染色程度
            feature['circularity'] = nucluei.circularity()  # 圆度
            feature['CAR'] = nucluei.CAR()  # 凸壳面积比
            feature['compactness'] = nucluei.compactness()  # 扁度
            feature['density'] = density[i]  # 密集程度
            feature['brightness'] = nucluei.brightness()  # 核亮度
            feature['halo_brightness'] = nucluei.halo_brightness()  # 核周亮度
            feature['brightness_delta'] = nucluei.halo_brightness() - nucluei.brightness()  # 核周与核内亮度差
            feature['stainness_delta'] = nucluei.stainness()- nucluei.halo_stainness()    # 核内与核周染色差

            feature['WeighedStaindegree'] = WeighedStaindegree[i]  # 加权染色程度
            feature['WeighedArea'] = WeighedArea[i]  # 加权面积

            feature['areaMstaindegree'] = nucluei.area() * nucluei.stainDegree()  # 面积和染色程度
            feature['carDcompactness'] = nucluei.CAR() / (nucluei.compactness() + 1e-6)  # 凸壳除扁度
            feature['mean_RGB'] = nucluei.Mean_RGB()
            dataset.append(feature)
        self.dataset = dataset
        return len(dataset)


class nucleus(object):
    """计算单核的特征
    """
    def __init__(self, patch, mask, ROI, contour, r=19):
        x = ROI[0]
        y = ROI[1]
        h = max(ROI[2]-x, 1)
        w = max(ROI[3]-y, 1)

        self.patch = patch[x: x+h, y: y+w]
        self.mask = mask[x: x+h, y: y+w]

        # mask = np.zeros(self.patch.shape[:2], dtype=np.uint8)
        # points = np.array(contour) - np.array((x - r, y - r))
        # cv2.fillPoly(mask, [np.stack(points)], color=255)

        self.feature = measure.regionprops(self.mask)[0]

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))
        self.loop = cv2.dilate(self.mask, kernel) - self.mask

    def __del__(self):
        pass

    def area(self):  # 面积
        return self.feature.area

    def stainDegree(self):  # 染色程度
        # 暂时用核的HSV通道的Saturation(饱和度)绝对值代替
        hsv = cv2.cvtColor(self.patch, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1] * (self.mask == 1)
        return saturation.sum() / np.flatnonzero(self.mask).size

    def brightness(self):  # 亮度
        # 平均亮度
        hsv = cv2.cvtColor(self.patch, cv2.COLOR_RGB2HSV)
        brightness = hsv[:, :, 2] * (self.mask == 1)
        return brightness.sum() / np.flatnonzero(self.mask).size
    
    def stainness(self):  # 亮度
        # 平均亮度
        hsv = cv2.cvtColor(self.patch, cv2.COLOR_RGB2HSV)
        brightness = hsv[:, :, 1] * (self.mask == 1)
        return brightness.sum() / np.flatnonzero(self.mask).size

    def perimeter(self):  # 周长
        return measure.perimeter(self.mask == 1)

    def convexHullArea(self):  # 凸壳面积
        return self.feature.convex_area

    def eccentricity(self):  # 离心率
        return self.feature.eccentricity

    def circularity(self):  # 圆度
        return 4 * np.pi * self.feature.area / (self.perimeter() ** 2)

    def CAR(self):  # 凸壳面积比
        return self.convexHullArea() / self.area()

    def Mean_RGB(self):  # 凸壳面积比
        img = self.patch
        img[...,0] = img[...,0] * (self.mask == 1)
        img[...,1] = img[...,1] * (self.mask == 1)
        img[...,2] = img[...,2] * (self.mask == 1)
        mean_RGB = (int(np.mean(img[...,0])),int(np.mean(img[...,1])),int(np.mean(img[...,2])))
        return mean_RGB
    
    def compactness(self):  # 扁度
        return self.perimeter() ** 2 / (2 * self.area())

    def halo_brightness(self):  # 核周亮度
        # 注释部分为原亮度指标，即大于220像素个数
        # 现改为核周平均亮度
        hsv = cv2.cvtColor(self.patch, cv2.COLOR_RGB2HSV)
        loopBrightness = hsv[:, :, 2] * (self.loop == 1)
        # num = loopBrightness[loopBrightness > 220].size
        # return num
        return loopBrightness.sum() / np.flatnonzero(self.loop).size
    
    def halo_stainness(self):  # 核周亮度
        # 注释部分为原亮度指标，即大于220像素个数
        # 现改为核周平均亮度
        hsv = cv2.cvtColor(self.patch, cv2.COLOR_RGB2HSV)
        loopBrightness = hsv[:, :, 1] * (self.loop == 1)
        # num = loopBrightness[loopBrightness > 220].size
        # return num
        return loopBrightness.sum() / np.flatnonzero(self.loop).size

def features_wrt_to_csv(dataset, dst_csvfile):
    if not len(dataset):
        return print('%s is empty, skip' % dst_csvfile)
    with open(dst_csvfile, 'w', newline='', encoding='utf-8') as f:
        fieldnames = list(dataset[0].keys())
        dict_writer = csv.DictWriter(f, delimiter=',', fieldnames=fieldnames)
        dict_writer.writeheader()
        for i in range(len(dataset)):
            dict_writer.writerow(dataset[i])
    return print('Done %s' % dst_csvfile)


def GaussianFuction(x, theta):       #高斯函数，用于计算密度
    return np.exp(-np.square(x) / (2*np.square(theta))) / (np.sqrt(2*np.pi)*theta)


def BinarySP(img, threColor = 15, threVol = 80):    #图片进行二值化
    wj1 = img.max(axis=2)
    wj2 = img.min(axis=2)
    wj3 = wj1 - wj2
    imgBin = wj3 > threColor
    imgBin = ndi.binary_fill_holes(imgBin)
    s = np.array([[0,1,0],[1,1,1],[0,1,1]], dtype=np.bool)
    imgBin = ndi.binary_opening(imgBin, s)
    imgCon, numCon = ndi.label(imgBin)
    imgConBig = morphology.remove_small_objects(imgCon, min_size=threVol)
    imgBin = imgConBig > 0
    return imgBin


def get_contours(img_masks):
    img_contours = []
    for i in range(img_masks.shape[-1]):
        mask = img_masks[::, ::, i]
        mask = np.expand_dims(mask, axis=2)
        _, contour, _ = cv2.findContours(mask.astype(np.uint8), 3, 2)
        if not len(contour):
            img_contours.append(np.array([]))
            continue
        contour = [p[0] for p in contour[0]]
        img_contours.append(np.stack(contour))
    return img_contours


def get_focus_nucleu(img_rois):
    focus_point = np.array([255, 255])
    roi_c_delt = np.zeros_like(img_rois[:, :2])
    for i, roi in enumerate(img_rois):
        x_c = np.mean([roi[0], roi[2]])
        y_c = np.mean([roi[1], roi[3]])
        roi_c_delt[i] = np.abs(np.array([x_c, y_c]) - focus_point)
    focus_index = np.argmin(np.sum(roi_c_delt, axis=1))
    return focus_index


def get_focus_area(img_rois):
    focus_index = []
    focus_area_rect = np.array([192, 192, 320, 320])

    x_r = np.where(img_rois[:, 0] >= focus_area_rect[0])
    y_r = np.where(img_rois[:, 1] >= focus_area_rect[1])
    x_l = np.where(img_rois[:, 2] < focus_area_rect[2])
    y_l = np.where(img_rois[:, 3] < focus_area_rect[3])

    x_set = np.intersect1d(x_r, x_l)
    y_set = np.intersect1d(y_r, y_l)

    focus_index = np.intersect1d(x_set, y_set)
    return focus_index


def cal_focus_nucleus_feature_v1(Img_preds):
    Areas = []
    Contours = []
    for index,img_preds in enumerate(Img_preds):
        img_masks = img_preds['masks']
        img_rois = img_preds['rois']
        valid_index = get_focus_area(img_rois)
        if len(valid_index) == 0 or len(img_masks) == 0:
            Areas.append([0])
            Contours.append([])
        else:
            img_contours = get_contours(img_masks)
            areas = []
            contours = []
            for i in valid_index:
                mask = img_masks[:, :, i]
                ROI = img_rois[i]
                contour = img_contours[i]
                nuc = nucleus(None, mask, ROI, contour, r=19)
                areas.append(nuc.area())
                contours.append(contour)
            Areas.append(areas)
            Contours.append(contours)
    return Areas,Contours


def cal_focus_nucleus_feature_v2(imgTotal2,predict2_core):
    Areas = []
    Valid_flag = []
    Valid_contours = []
    for img, img_preds in zip(imgTotal2.copy(), predict2_core.copy()):
        nucS = nucSet(img, img_preds)
        nucS.extractFeature()
        if len(nucS.dataset):
            valid_flag = False
            valid_contours = [nucS.img_contours[v_ind] for v_ind in nucS.valid_index]
            valid_areas = [a['area'] for a in nucS.dataset]
            feature_brightness = [a['brightness'] for a in nucS.dataset]
            feature_stain_degree = [a['stainDegree'] for a in nucS.dataset]
            feature_weighted_stain = [a['WeighedStaindegree'] for a in nucS.dataset]
            feature_brightness_delta = [a['brightness_delta'] for a in nucS.dataset]
    
            index_max_area = np.argmax(valid_areas)
            brightness = feature_brightness[index_max_area]
            stain_degree = feature_stain_degree[index_max_area]
            weighted_stain = feature_weighted_stain[index_max_area]
            brightness_delta = feature_brightness_delta[index_max_area]
            if stain_degree < 200:
                if brightness < 200 and brightness > 80:
                    if brightness_delta > 10 and weighted_stain < 3:
                        valid_flag = True
                elif brightness >= 200:
                    if weighted_stain < 3:
                        valid_flag = True
            Areas.append(valid_areas)
            Valid_flag.append(valid_flag)
            Valid_contours.append(valid_contours)
        else:
            Areas.append([0])
            Valid_flag.append(False)
            Valid_contours.append([])
    return Areas, Valid_flag, Valid_contours

def cal_focus_nucleus_feature_v3(imgTotal2_core,predict2_core):
    Areas = []
    Type_img = []
    Contours_core = []
    for img, img_preds in zip(imgTotal2_core.copy(), predict2_core.copy()):
        nucS = nucSet(img, img_preds)
        nucS.extractFeature()
        if len(nucS.dataset):
            valid_contours = [nucS.img_contours[v_ind] for v_ind in nucS.valid_index]
            valid_areas = [a['area'] for a in nucS.dataset]
            feature_brightness = [a['brightness'] for a in nucS.dataset]
            feature_stain_degree = [a['stainDegree'] for a in nucS.dataset]
            feature_weighted_stain = [a['WeighedStaindegree'] for a in nucS.dataset]
            feature_brightness_delta = [a['brightness_delta'] for a in nucS.dataset]
            feature_stainness_delta = [a['stainness_delta'] for a in nucS.dataset]
            feature_mean_RGB = [a['mean_RGB'] for a in nucS.dataset]
            
            index_max_area = np.argmax(valid_areas)
            brightness = feature_brightness[index_max_area]
            stain_degree = feature_stain_degree[index_max_area]
            weighted_stain = feature_weighted_stain[index_max_area]
            brightness_delta = feature_brightness_delta[index_max_area]
            stainness_delta = feature_stainness_delta[index_max_area]
            mean_RGB = feature_mean_RGB[index_max_area]
            # 排除拥挤细胞团 前景的平均饱和度 深染面积
            img_small = img[192:320,192:320,:]
            img_small = cv2.resize(img_small,(512,512))
            hsv = cv2.cvtColor(img_small, cv2.COLOR_RGB2HSV)
            imgBin = hsv[...,1]>50
            imgCon, numCon = ndi.label(imgBin)
            imgConBig = morphology.remove_small_objects(imgCon, min_size=1000)
            imgBin = imgConBig > 0
            imgBin_dark = np.logical_and(hsv[...,1]>110,hsv[...,2]<200)
            
            Dark_area = int(np.sum(imgBin_dark))
            if np.sum(imgBin)!=0:
                meanStain_foreground = int(np.sum(hsv[...,1]*imgBin)/np.sum(imgBin))
                if (mean_RGB[0]>mean_RGB[1] and mean_RGB[0]>mean_RGB[2]+20) or (mean_RGB[1]>mean_RGB[0] and mean_RGB[1]>mean_RGB[2]):
                    Type_img.append('Color')
                elif stain_degree > 200:
                    Type_img.append('Black_impurity')
                elif brightness < 80: 
                    Type_img.append('Dark')
                elif Dark_area>90000  or (Dark_area>70000 and meanStain_foreground>130) or weighted_stain > 5:
                    Type_img.append('Black_group')
                elif brightness < 150 and brightness_delta < 10 and stainness_delta<40:
                    Type_img.append('Boundary_blur')
                else:
                    Type_img.append('Right')#Isolated
            else:
                Type_img.append('Nothing')
            Areas.append(valid_areas)
            Contours_core.append(valid_contours)    
        else:
            Areas.append([0])
            Contours_core.append([]) 
            Type_img.append('Nothing')
    return Areas, Type_img, Contours_core