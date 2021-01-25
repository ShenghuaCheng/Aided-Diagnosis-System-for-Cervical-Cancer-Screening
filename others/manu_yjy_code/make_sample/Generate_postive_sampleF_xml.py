# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 23:24:06 2018

@author: yujingya
"""
import sys
sys.path.append('../')
import xml.etree.cElementTree as et
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import scipy.ndimage as ndi
from openslide import OpenSlide
from multiprocessing import Manager, Lock
from utils.sdpc_python import sdpc
from tqdm import tqdm
from utils.function_set import Get_predictimgMultiprocess
import multiprocessing.dummy as multiprocessing
import time

def GetImg_whole(original_index, startpointlist, k, ors, level, sizepatch_read, pathsave,filename,group):
    """ 返回RGB通道 sizepatch大小图片
    """
    img = ors.read_region(startpointlist[k], level, (sizepatch_read,sizepatch_read))
    img = np.array(img)#RGBA
    img = img[:, :, 0 : 3]
    print(k)
    pathsave1 = pathsave + group[k]
    if not os.path.exists(pathsave1):
        os.makedirs(pathsave1)
    cv2.imwrite(pathsave1 + '/{}_{}_{}_{}.tif'.format(filename, original_index[k], startpointlist[k],group[k]), img[...,::-1])
    return None

def GetImg_whole_sdpc(original_index, startpointlist, k, ors, level, sizepatch_read, lock,pathsave,filename,group):
    """ 返回RGB通道 sizepatch大小图片
    """
#    print(k)
    lock.acquire()
    img = ors.getTile(level, startpointlist[k][0], startpointlist[k][1], int(sizepatch_read), int(sizepatch_read))
    lock.release()

    img = np.ctypeslib.as_array(img)
    img.dtype = np.uint8
    img = img.reshape((int(sizepatch_read), int(sizepatch_read), 3))
    img = img[...,::-1]
    
    pathsave1 = pathsave + group[k]
    if not os.path.exists(pathsave1):
        os.makedirs(pathsave1)
    cv2.imwrite(pathsave1 + '/{}_{}_{}_{}.tif'.format(filename, original_index[k], startpointlist[k],group[k]), img[...,::-1])
    return None

def Generate_postive_sample_xml(pathfolder_xml,pathfolder_mrxs,pathsave,filename ,filetype,level,sizepatch_read):
    pathxml = pathfolder_xml + filename + '.xml'
    pathfile = pathfolder_mrxs + filename + filetype
    if not os.path.exists(pathfile):
        print(pathfile)
    tree = et.parse(pathxml)
    root = tree.getroot()
    group = []
    startpointlist = []
    original_index = []
    root1 = root.findall('Annotations')[0]
    for annotation in root1.findall('Annotation'):
        group.append(annotation.get("PartOfGroup").split('_')[0])
        original_index.append(annotation.get("Name").split(' ')[-1])
        root2 = annotation.find('Coordinates')
        points = []
        for root3 in root2.findall('Coordinate'):
            Y = int(float(root3.get("Y")))
            X = int(float(root3.get("X")))
            points.append((Y,X))
        points = np.array(points)
        start_pointY = (points[:,0].min()+points[:,0].max())//2-sizepatch_read//2
        start_pointX = (points[:,1].min()+points[:,1].max())//2-sizepatch_read//2
        startpointlist.append((start_pointY,start_pointX))
#    return group 
    
    filetype = '.'+pathfile.split('.')[-1]
    if filetype == '.sdpc':
        ors = sdpc.Sdpc()
        ors.open(pathfile)
    else:
        ors = OpenSlide(pathfile)
    pool = multiprocessing.Pool(15)
    manager = Manager()
    lock = manager.Lock()
    start = time.time()
    for k in range(len(startpointlist)):
        if filetype == '.sdpc':
            pool.apply_async(GetImg_whole_sdpc, args=(original_index, startpointlist, k, ors, level, sizepatch_read, lock, pathsave,filename,group))
        else:
            pool.apply_async(GetImg_whole, args=(original_index, startpointlist, k, ors, level, sizepatch_read, pathsave,filename,group))
    pool.close()
    pool.join()
    end = time.time()
    print('多线程读图耗时{}\n{}张\nlevel = {}读入大小{}'.format(end-start,len(startpointlist),level,sizepatch_read))
 
    return None
        
        
if __name__=='__main__':
    pathfolder_xml = r'H:\TCTDATA\SZSQ_originaldata\Labelfiles\xml_Shengfuyou_1th'+'/'
    pathfolder_mrxs =  r'H:\TCTDATA\SZSQ_originaldata\Shengfuyou_1th'+'/'
    pathsave = r'H:\AdaptDATA\train\szsq_sfy1_75_'
    sizepatch_read = int(1024*1.5*0.293/0.18)
    filelist = os.listdir(pathfolder_xml)
    filetype = '.sdpc'
    filelist = [cz[:-len('.xml')] for cz in filelist if '.xml' in cz]
    Group = []
    level = 0
    for filename in tqdm(filelist[40:]):#len(filelist)
        Generate_postive_sample_xml(pathfolder_xml,pathfolder_mrxs,pathsave,filename ,filetype,level,sizepatch_read)
    
    
    