# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 23:24:06 2018

@author: YuJingya
"""
import sys
sys.path.append('../')
from utils.parameter_set import TCT_set,Size_set
import os
from matplotlib import pyplot as plt
import time
import numpy as np
import cv2
from openslide import OpenSlide
import os
import heapq
from tqdm import tqdm 
from utils.function_set import saveContours_xml
from utils.Recommend_class import Recommend
import xml.etree.cElementTree as et
import scipy.ndimage as ndi
from openslide import OpenSlide
from multiprocessing import Manager, Lock
from utils.sdpc_python import sdpc
  
def Generate_sample_xml_22():
    ratio = input_str['model1_resolution']/input_str['input_resolution'] 
    level = input_str['level']
    font=cv2.FONT_HERSHEY_SIMPLEX
    pathxml = pathfolder_xml + filename + '.xml'
    pathfile = pathfolder + filename + input_str['filetype'] 
    
    if input_str['filetype']  == '.sdpc':
        ors = sdpc.Sdpc()
        ors.open(pathfile)
    else:
        ors = OpenSlide(pathfile)
        
    tree = et.parse(pathxml)
    root = tree.getroot()
    root = root.findall('Annotations')[0]
    for num_picture in range(2):
        img0 = []
        img1 = []
        img2 = []
        for index,annotation in enumerate(root.findall('Annotation')[num_picture*11:(num_picture+1)*11]):
#            break
            root2 = annotation.find('Coordinates')
            root_start = root2.findall('Coordinate')[0]
            root_end = root2.findall('Coordinate')[2]
            if index in range(0,2):
                size_recom  = [int(128*ratio*16/3/2),int(128*ratio)]
            elif index in range(2,2+4):
                size_recom  = [int(128*ratio*16/3/4),int(128*ratio)]
            elif index in range(2+4,2+4+5):
                size_recom  = [int(128*ratio*16/3/5),int(128*ratio)]
                root2.findall('Coordinate')[3].get("Y")
            X_start = int(float(root_start.get("X")))
            Y_start = int(float(root_start.get("Y")))
            X_end = int(float(root_end.get("X")))
            Y_end = int(float(root_end.get("Y")))
            X = int(float(root_start.get("X"))) - int((size_recom[0] - (X_end-X_start))/2)
            Y = int(float(root_start.get("Y"))) - int((size_recom[1] - (Y_end-Y_start))/2)
            
            
            if input_str['filetype']  == '.sdpc':
                manager = Manager()
                lock = manager.Lock()
                lock.acquire()
                imgSub = ors.getTile(level, Y, X, size_recom[0], size_recom[1])
                lock.release()
                imgSub = np.ctypeslib.as_array(imgSub)
                imgSub.dtype = np.uint8
                imgSub = imgSub.reshape((int(size_recom[1]), int(size_recom[0]), 3))
                imgSub = imgSub[...,::-1]
            
            else:
                imgSub = ors.read_region((X,Y), level, (size_recom[0], size_recom[1]))
                imgSub = np.array(imgSub)
                imgSub = imgSub[:, :, 0: 3]
                
                
            image = np.ascontiguousarray(imgSub, dtype=np.uint8)
            predicit2 = annotation.get("PartOfGroup").split('_')[1][:6]
            imgSub = cv2.putText(image,str(predicit2),(15,70),font,1,(0,0,255),3)#10x :(15,15) 0.5 1 
            if index in range(0,2):
                img0.append(imgSub)
            elif index in range(2,2+4):
                img1.append(imgSub)
            elif index in range(2+4,2+4+5):
                img2.append(imgSub)
        img0 = np.hstack(img0)
        img1 = np.hstack(img1)
        img2 = np.hstack(img2)
        w = min([img0.shape[1],img1.shape[1],img2.shape[1]])
        img = [img0[:,:w,:]]+[img1[:,:w,:]]+[img2[:,:w,:]]
        img = np.vstack(img)
        cv2.imwrite(pathfolder_save + filename + '_%s.tif'%num_picture, img[:, :, ::-1])          
        
        lock = manager.Lock()
        lock.acquire()
        imgSub = ors.getTile(level, Y, X, img.shape[1], img.shape[0])
        lock.release()
        imgSub = np.ctypeslib.as_array(imgSub)
        imgSub.dtype = np.uint8
        imgSub = imgSub.reshape((int(img.shape[0]), int(img.shape[1]), 3))
        imgSub = imgSub[...,::-1] 
        cv2.imwrite(pathfolder_save + filename + '_center%s.tif'%num_picture, imgSub[:, :, ::-1])          
    return None
    
    
def Generate_sample_xml_2_4_5():
    ratio = input_str['model1_resolution']/input_str['input_resolution'] 
    level = input_str['level']
    size_predict  = [int(128*ratio),int(128*ratio)]
    font=cv2.FONT_HERSHEY_SIMPLEX
    pathxml = pathfolder_xml + filename + '.xml'
    pathsvs = pathfolder + filename + '.svs'
    ors = OpenSlide(pathsvs)
    tree = et.parse(pathxml)
    root = tree.getroot()
    root = root.findall('Annotations')[0]
    for num_picture in range(2):
        img0 = []
        img1 = []
        img2 = []
        for index,annotation in enumerate(root.findall('Annotation')[num_picture*11:(num_picture+1)*11]):
    #        break
            root2 = annotation.find('Coordinates')
            root2 = root2.findall('Coordinate')[0]
            if index in range(0,2):
                size_recom  = [int(256*16/3/2*ratio),int(256*ratio)]
            elif index in range(2,2+4):
                size_recom  = [int(256*16/3/4*ratio),int(256*ratio)]
            elif index in range(2+4,2+4+5):
                size_recom  = [int(256*16/3/5*ratio),int(256*ratio)]
            X = int(float(root2.get("X"))) - int((size_recom[0] - size_predict[0])/2)
            Y = int(float(root2.get("Y"))) - int((size_recom[1] - size_predict[1])/2)
            imgSub = ors.read_region((X,Y), level, (size_recom[0], size_recom[1]))
            imgSub = np.array(imgSub)
            imgSub = imgSub[:, :, 0: 3]
            image = np.ascontiguousarray(imgSub, dtype=np.uint8)
            Type = annotation.get("PartOfGroup").split('_')[0]
            predicit2 = annotation.get("PartOfGroup").split('_')[1][:6]
            morphology = {'0':'single_hs',
                          '1':'cells',
                          '2':'single',
                          '3':'ngec',
                          '4':'yz',
                          '5':'nothing',
                          '6':'cells_hs'}
            imgSub = cv2.putText(image,str(morphology[Type]),(15,100),font,1,(0,255,0),3)#10x :(15,15) 0.5 1 
            imgSub = cv2.putText(image,str(predicit2),(15,70),font,1,(0,0,255),3)#10x :(15,15) 0.5 1 
            if annotation.get("Color") == '#ff0000':
                imgSub = cv2.putText(image,str('union'),(15,30),font,1,(255,0,0),3)
            elif annotation.get("Color") == '#ffff00':
                imgSub = cv2.putText(image,str('model2'),(15,30),font,1,(255,0,0),3)#10x :(15,15) 0.5 1 
            elif annotation.get("Color") == '#0000ff':
                imgSub = cv2.putText(image,str('nucleus'),(15,30),font,1,(255,0,0),3)#10x :(15,15) 0.5 1 
            elif annotation.get("Color") == '#00ffff':
                imgSub = cv2.putText(image,str('model2'),(15,30),font,1,(255,0,0),3)#10x :(15,15) 0.5 1 
            if index in range(0,2):
                img0.append(imgSub)
            elif index in range(2,2+4):
                img1.append(imgSub)
            elif index in range(2+4,2+4+5):
                img2.append(imgSub)
        img0 = np.hstack(img0)
        img1 = np.hstack(img1)
        img2 = np.hstack(img2)
        w = min([img0.shape[1],img1.shape[1],img2.shape[1]])
        img = [img0[:,:w,:]]+[img1[:,:w,:]]+[img2[:,:w,:]]
        img = np.vstack(img)
        cv2.imwrite(pathfolder_save + filename + '_%s.tif'%num_picture, img[:, :, ::-1])          
        return None
if __name__=='__main__':
    input_str = {'filetype': '.sdpc',# .svs .sdpc .mrxs
                 'level': 0,
                 'read_size_model1_resolution': (1216, 1936), #1936 1216 横 model1_resolution下
                 'scan_over': (120, 120),
                 'model1_input': (512, 512, 3),
                 'model2_input': (256, 256, 3),
                 'model1_resolution':0.586,
                 'model2_resolution':0.293,
                 'input_resolution':0.18,#0.179
                 'cut':'3*5'}
    size_set = Size_set(input_str)
    num_recom = 22
    level = 0
    pathfolder_set, _ = TCT_set()
    for pathfolder in pathfolder_set['szsq_sdpc'][-2:-1]:
#        break
        pathsave1 = r'F:\recom\model1_szsq615_model2_szsq1050\SZSQ_originaldata\Tongji_4th\positive'+'/'
        pathfolder_xml = pathsave1 + 'xml_22/'
        pathfolder_save = pathsave1 + 'img_22/'
        while not os.path.exists(pathfolder_save):
            os.makedirs(pathfolder_save)  
        CZ = os.listdir(pathfolder_xml)
        CZ = [cz[:-len('.xml')] for cz in CZ if '.xml' in cz]
        for filename in tqdm(CZ):
            print(filename)
            Generate_sample_xml_22()

# =============================================================================
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = cv2.GaussianBlur(gray, (3, 3), 0)
# canny1 = cv2.Canny(gray, 50, 120)
# =============================================================================

