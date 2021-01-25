# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 00:43:20 2018
@author: yujingya
"""
import numpy as np
import multiprocessing.dummy as multiprocessing
import cv2
from matplotlib import pyplot as plt
from lxml import etree
from scipy import ndimage as ndi
from skimage import morphology

def EstimateRegion(ors,leveltemp=5):
    '''
    估计切片的主要区域，并返回在leveltemp下的二值图片
    :param pathImg: 图片.mrxs文件路径
    :param leveltemp: 图片操作缩放等级，default = 5
    :return: 切片主要区域的二值Mask
    '''
    level = leveltemp
    position = (0, 0)
    size = ors.level_dimensions[level]
    img = ors.read_region(position, level, size)
    img = np.array(img)
    img = img[:, :, 0 : 3] 
    threColor = 8
    threVol = 4
    wj1 = img.max(axis=2)
    wj2 = img.min(axis=2)
    wj3 = wj1 - wj2
    imgBin = wj3 > threColor
    imgBin = morphology.remove_small_objects(imgBin, min_size=threVol)
    imgBinB = imgBin
    imgBin = cv2.blur(np.float32(imgBin), (30, 30)) > 0
    imgCon, numCon = ndi.label(imgBin)
    num = np.zeros((numCon,))
    for i in range(1, 1+numCon):
        tempImg = imgCon == i
        num[i-1] = np.sum(tempImg)
    ind = np.argmax(num)
    imgBin = imgCon == ind+1
    imgBin = ndi.binary_fill_holes(imgBin)
    if np.sum(imgBin) < 6000000:
        imgBin = imgBinB
        imgBin = cv2.blur(np.float32(imgBin), (50, 50)) > 0
        imgCon, numCon = ndi.label(imgBin)
        num = np.zeros((numCon,))
        for i in range(1, 1+numCon):
            tempImg = imgCon == i
            num[i-1] = np.sum(tempImg)
        ind = np.argmax(num)
        imgBin = imgCon == ind+1
        imgBin = ndi.binary_fill_holes(imgBin)
        if np.sum(imgBin) < 5800000:
            imgBin = imgBinB
            imgBin = cv2.blur(np.float32(imgBin), (80, 80)) > 0
            imgCon, numCon = ndi.label(imgBin)
            num = np.zeros((numCon,))
            for i in range(1, 1+numCon):
                tempImg = imgCon == i
                num[i-1] = np.sum(tempImg)
            ind = np.argmax(num)
            imgBin = imgCon == ind+1
            imgBin = ndi.binary_fill_holes(imgBin)
    return imgBin


def Get_startpointlist(ors, level, levelratio, sizepatch,widthOverlap,flag):
    sizepatch_y,sizepatch_x = sizepatch
    widthOverlap_y,widthOverlap_x = widthOverlap
    ratio = levelratio ** level
    if flag =='0':
        imgmap = EstimateRegion(ors)
        start_xmin = imgmap.nonzero()[0][0] * 32
        end_xmax = imgmap.nonzero()[0][-1] * 32
        start_ymin = np.min(imgmap.nonzero()[1]) * 32
        end_ymax = np.max(imgmap.nonzero()[1]) * 32
    elif flag == '1':
        size = ors.level_dimensions[level]
        start_xmin = 0
        end_xmax = size[1]
        start_ymin = 0
        end_ymax = size[0]
    numblock_x = np.around(((end_xmax-start_xmin)/ratio - sizepatch_x) / (sizepatch_x - widthOverlap_x)+1)
    numblock_y = np.around(((end_ymax-start_ymin)/ratio - sizepatch_y) / (sizepatch_y - widthOverlap_y)+1)
    # 得到所有开始点的坐标(排除白色起始点）
    startpointlist = []
    for j in range(int(numblock_x)):
        for i in range(int(numblock_y)):
            startpointtemp = (start_ymin + i*(sizepatch_y-widthOverlap_y)*ratio,start_xmin+j*(sizepatch_x-widthOverlap_x)*ratio)
            startpointlist.append(startpointtemp)
    return startpointlist



def GetImg_color(startpointlist, k, ors, level, sizepatch, sizepatch_predict):
    img = ors.read_region(startpointlist[k], level, sizepatch)
    img = np.array(img)#RGBA
    img = img[:, :, 0 : 3]
    img = cv2.resize(img,sizepatch_predict)
    return img

def Get_predictimgMultiprocess(startpointlist, ors, level, sizepatch, sizepatch_predict):    
    # 多线程读图
    pool = multiprocessing.Pool(20)
    imgTotals = []
    for k in range(len(startpointlist)):
        imgTotal = pool.apply_async(GetImg_color,args=(startpointlist, k, ors, level, sizepatch, sizepatch_predict))
        imgTotals.append(imgTotal)
    pool.close()
    pool.join()
    imgTotal = np.array([x.get() for x in imgTotals])
    imgTotals = []
    return imgTotal

def Split_into_small(imgTotal,sizepatch_predict,sizepatch_predict_small, widthOverlap_predict):
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
   
    imgTotals = []
    for k in range(len(imgTotal)):
        for i in range(len(startpointlist)):
            startpointtemp = startpointlist[i]
            startpointtempy,startpointtempx = startpointtemp
            img = imgTotal[k][startpointtempx:startpointtempx+512,startpointtempy:startpointtempy+512]
            imgTotals.append(img)
    imgTotals = np.array(imgTotals)
    return imgTotals,int(numblock_x*numblock_y),startpointlist

def trans_Normalization(imgTotal):
    imgTotal = imgTotal[:,:,:,::-1]
    imgTotal = np.float32(imgTotal) / 255.
    imgTotal = imgTotal - 0.5
    imgTotal = imgTotal * 2.  
    return imgTotal

def Get_rectcontour(startlist_pos_temp,Sizeimgtemp):
    contourlist =[]
    for i in range(len(startlist_pos_temp)):
        if len(Sizeimgtemp)!=2:
            sizeimgtemp = Sizeimgtemp[i]
        else:
            sizeimgtemp = Sizeimgtemp
        startpointlist0 = (startlist_pos_temp[i])
        startpointlist1 = (startlist_pos_temp[i][0]+sizeimgtemp[0],startlist_pos_temp[i][1])
        startpointlist2 = (startlist_pos_temp[i][0]+sizeimgtemp[0],startlist_pos_temp[i][1]+sizeimgtemp[1])
        startpointlist3 = (startlist_pos_temp[i][0],startlist_pos_temp[i][1]+sizeimgtemp[1])
        contoursListtemp = np.vstack((startpointlist0,startpointlist1,startpointlist2,startpointlist3))
        contourlist.append(contoursListtemp)
    return contourlist

def saveContours_xml(contourslist,label_annotation,label_color,filename):#ratio-采样率之比
    #根节点
    ASAP_Annotations = etree.Element("ASAP_Annotations")
    #一级子节点，标记的集合
    Annotations = etree.SubElement(ASAP_Annotations,'Annotations')
    #一级子节点，标记的类别信息
    AnnotationGroups = etree.SubElement(ASAP_Annotations,'AnnotationGroups')
    for i in range(len(contourslist)):
        for j in range(len(contourslist[i])):
            Annotation = etree.SubElement(Annotations,'Annotation')   
            Annotation.set('Color',label_color[i])
            Annotation.set('Name','Annotation '+str(j))
            Annotation.set('PartOfGroup',str(label_annotation[i][j]))
            Annotation.set('Type','Polygon')
            contour = contourslist[i][j]
            Coordinates = etree.SubElement(Annotation,'Coordinates')
            for k in range(contour.shape[0]):
                x = str(contour[k,0])
                y = str(contour[k,1])
                Coordinate = etree.SubElement(Coordinates,'Coordinate')         
                Coordinate.set('Order',str(k))
                Coordinate.set('X',x)
                Coordinate.set('Y',y)
        Group = etree.SubElement(AnnotationGroups,'Group')
        Group.set('Color',label_color[i])
        Group.set('Name','Annotation Group'+str(i))
        Group.set('PartOfGroup',str(label_annotation[i][j]))
        Attributes = etree.SubElement(Group,'Attributes')     
    tree = etree.ElementTree(ASAP_Annotations)
    tree.write(filename, pretty_print=True, xml_declaration=True, encoding='utf-8')
    
def region_proposal_old(heatmap, img = None, threshold=0.7):
    img_prop = []
    heatmap = cv2.resize(heatmap,(512,512))
    w_hm, h_hm = heatmap.shape
    prob_max = np.max(heatmap)
    _, mask = cv2.threshold(heatmap, prob_max*threshold, prob_max, cv2.THRESH_BINARY)
    mask[mask > 0] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))  # 定义结构元素
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 开运算 先缩后涨
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    mask = cv2.dilate(mask, kernel)  
    _, mask_c, _ = cv2.findContours(np.uint8(mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros((w_hm, h_hm))  
    local = []
    
    for c in mask_c:
       local.append((np.mean(c[...,1]),np.mean(c[...,0])))
    
    if img!= None:
        img_prop = np.uint8((img / 2 + 0.5) * 255)
        for c in mask_c:
            hull = cv2.convexHull(c)  # 凸包
            img_prop = cv2.polylines(img_prop, [hull], True, (0, 255, 0), 2)  # 绘制凸包
            mask = cv2.fillPoly(mask, [hull], 255)
    return img_prop, mask,local 

#heatmap = item[...,1]
def region_proposal(heatmap, img=None, threshold=0.7):
    img_prop = []
    heatmap = cv2.resize(heatmap, (512, 512))
    w_hm, h_hm = heatmap.shape
    prob_max = np.max(heatmap)
    _, mask = cv2.threshold(heatmap, prob_max * threshold, prob_max, cv2.THRESH_BINARY)
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

    if img != None:
        mask = np.zeros((w_hm, h_hm))
        img_prop = np.uint8((img / 2 + 0.5) * 255)
        for c in mask_c:
            hull = cv2.convexHull(c)  # 凸包
            img_prop = cv2.polylines(img_prop, [hull], True, (0, 255, 0), 2)  # 绘制凸包
            mask = cv2.fillPoly(mask, [hull], 255)
    return img_prop, mask, local