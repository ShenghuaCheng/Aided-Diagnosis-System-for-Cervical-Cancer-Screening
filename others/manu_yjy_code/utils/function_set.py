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
from skimage import morphology,measure
import time
from multiprocessing import Manager, Lock
from utils.sdpc_python import sdpc
from openslide import OpenSlide

def EstimateRegion(ors,leveltemp = 5,threColor = 8):
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
    wj1 = img.max(axis=2)
    wj2 = img.min(axis=2)
    wj3 = wj1 - wj2
    imgBin = wj3 > threColor
    threVol = 4
    imgBin = morphology.remove_small_objects(imgBin, min_size = threVol)
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


def Get_startpointlist(pathfile, level, levelratio, sizepatch_read, widthOverlap, threColor = 8):
    sizepatch_y,sizepatch_x = sizepatch_read
    widthOverlap_y,widthOverlap_x = widthOverlap
    ratio = levelratio ** level
    filetype = '.'+pathfile.split('.')[-1]
    if filetype == '.mrxs':
        ors = OpenSlide(pathfile)
        imgmap = EstimateRegion(ors,threColor = threColor)
        start_xmin = imgmap.nonzero()[0][0] * 32
        end_xmax = imgmap.nonzero()[0][-1] * 32
        start_ymin = np.min(imgmap.nonzero()[1]) * 32
        end_ymax = np.max(imgmap.nonzero()[1]) * 32
    else:
        if filetype == '.sdpc':
            ors = sdpc.Sdpc()
            ors.open(pathfile)
            attr = ors.getAttrs()
            size = (attr['width'], attr['height'])
        elif filetype == '.svs':
            ors = OpenSlide(pathfile)
            size = ors.level_dimensions[level]
        start_xmin = 0
        end_xmax = size[1]
        start_ymin = 0
        end_ymax = size[0]
        
    numblock_x = np.ceil(((end_xmax-start_xmin)/ratio - sizepatch_x) / (sizepatch_x - widthOverlap_x)+1)
    numblock_y = np.ceil(((end_ymax-start_ymin)/ratio - sizepatch_y) / (sizepatch_y - widthOverlap_y)+1)
    # 得到所有开始点的坐标(排除白色起始点）
    startpointlist = []
    for j in range(int(numblock_x)):
        for i in range(int(numblock_y)):
            startpointtemp = (start_ymin + i*(sizepatch_y-widthOverlap_y)*ratio,start_xmin+j*(sizepatch_x-widthOverlap_x)*ratio)
            startpointlist.append(startpointtemp)
    return startpointlist

def GetImg_whole(startpointlist, k, ors, level, sizepatch, pathsave = None):
    """ 返回RGB通道 sizepatch大小图片
    """
    img = ors.read_region(startpointlist[k], level, sizepatch)
    img = np.array(img)#RGBA
    img = img[:, :, 0 : 3]
    print(k)
    if pathsave!=None:
        cv2.imwrite(pathsave + '{}_{}.tif'.format(k, startpointlist[k]),img[...,::-1])
    return img


def GetImg_whole_sdpc(startpointlist, k, ors, level, sizepatch, lock,pathsave=None):
    """ 返回RGB通道 sizepatch大小图片
    """
    print(k)
    lock.acquire()
    img = ors.getTile(level, startpointlist[k][1], startpointlist[k][0], int(sizepatch[0]), int(sizepatch[1]))
    lock.release()

    img = np.ctypeslib.as_array(img)
    img.dtype = np.uint8
    img = img.reshape((int(sizepatch[1]), int(sizepatch[0]), 3))
    img = img[...,::-1]
    if pathsave!=None:
        cv2.imwrite(pathsave + '{}_{}.tif'.format(k, startpointlist[k]), img[...,::-1])
    return img


def Get_predictimgMultiprocess(pathfile, startpointlist, level, sizepatch_read, pathsave=None):
    """ 添加参数lock，保证多线程读图
    """
    filetype = '.'+pathfile.split('.')[-1]
    if filetype == '.sdpc':
        ors = sdpc.Sdpc()
        ors.open(pathfile)
    else:
        ors = OpenSlide(pathfile)
    start = time.time()
    pool = multiprocessing.Pool(16)
    imgTotals = []
    manager = Manager()
    lock = manager.Lock()
    for k in range(len(startpointlist)):
        if filetype == '.sdpc':
            imgTotal = pool.apply_async(GetImg_whole_sdpc, args=(startpointlist, k, ors, level, sizepatch_read, lock, pathsave))
        else:
            imgTotal = pool.apply_async(GetImg_whole, args=(startpointlist, k, ors, level, sizepatch_read, pathsave))
        imgTotals.append(imgTotal)
    pool.close()
    pool.join()
    imgTotals = np.array([x.get() for x in imgTotals])
    end = time.time()
    print('多线程读图耗时{}\n{}张\nlevel = {}读入大小{}'.format(end-start,len(startpointlist),level,sizepatch_read))
    return imgTotals

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
    return int(numblock_y*numblock_x),startpointlist

def Split_into_small(imgTotal,startpointlist_split,sizepatch_predict_small):
    start = time.time()
    imgTotals = []
    for k in range(len(imgTotal)):
        for i in range(len(startpointlist_split)):
            startpointtemp = startpointlist_split[i]
            startpointtempy,startpointtempx = startpointtemp
            img = imgTotal[k][startpointtempx:startpointtempx+sizepatch_predict_small[1],startpointtempy:startpointtempy+sizepatch_predict_small[0]]
            imgTotals.append(img)
    imgTotals = np.array(imgTotals)
    end = time.time()
    print('Split_into_small耗时' + str(end-start)+'/'+ str(len(imgTotals))+'张')
    return imgTotals

def trans_Normalization(imgTotal,
                        channal_trans_flag):
    """Normalization for channel_last imgs
    :param imgTotal:
    :param channal_trans_flag: 
    :return: imgs after normalization
    """
    start = time.time()
    if channal_trans_flag is None:
        raise ValueError('The channal_trans_flag should not be "None".'
                         ' Please check wether need to trans channel older')
    if channal_trans_flag:
        print('———通道顺序改变———')
        imgTotal = imgTotal[...,::-1]
    imgTotal = np.float32(imgTotal) / 255.
    imgTotal = imgTotal - 0.5
    imgTotal = imgTotal * 2.  
    end = time.time()
    print('Normalization耗时' + str(end-start)+'/'+ str(len(imgTotal))+'张')
    return imgTotal

def region_proposal(heatmap, output_shape, img=None, threshold=0.7):
    prob_max = np.max(heatmap)
    heatmap[heatmap < prob_max *threshold] = 0
    mask = (heatmap!=0).astype(np.uint8)
    mask_label = measure.label(mask.copy())
    local = []
    for c in range(mask_label.max()):
        heatmap_temp = heatmap*(mask_label == (c+1))
        a = np.where(heatmap_temp == heatmap_temp.max())
        local_y = np.around((a[1][0]+0.5)*output_shape[1]/heatmap.shape[1]).astype(int)
        local_x = np.around((a[0][0]+0.5)*output_shape[0]/heatmap.shape[0]).astype(int)
        local.append((local_y, local_x))
        
    if np.any(img != None):
        img_prop = []
        heatmap = cv2.resize(heatmap, output_shape[::-1])
        prob_max = np.max(heatmap)
        heatmap[heatmap < prob_max *threshold] = 0
        mask = (heatmap!=0).astype(np.uint8)
        _, mask_c, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(output_shape)
        img_prop = np.uint8((img / 2 + 0.5) * 255)
        for c in mask_c:
            hull = cv2.convexHull(c)  # 凸包
            img_prop = cv2.polylines(img_prop, [hull], True, (0, 255, 0), 2)  # 绘制凸包
            mask = cv2.fillPoly(mask, [hull], 255)
        return img_prop, mask, local

    else:
        return local
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
            Annotation.set('Color',label_color[i][j])
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
        Group.set('Color',label_color[i][j])
        Group.set('Name','Annotation Group'+str(i))
        Group.set('PartOfGroup',str(label_annotation[i][j]))
        Attributes = etree.SubElement(Group,'Attributes')     
    tree = etree.ElementTree(ASAP_Annotations)
    tree.write(filename, pretty_print=True, xml_declaration=True, encoding='utf-8')