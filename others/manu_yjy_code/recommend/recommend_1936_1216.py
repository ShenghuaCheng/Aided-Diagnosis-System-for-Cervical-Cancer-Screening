# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 21:02:41 2019

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
import numpy as np
import os
import heapq
from tqdm import tqdm 
from utils.function_set import saveContours_xml
from utils.Recommend_class import Recommend
from lxml import etree
from multiprocessing import Manager, Lock
from utils.sdpc_python import sdpc

if __name__ == '__main__':
    input_str = {'filetype': '.sdpc',# .svs .sdpc .mrxs
                 'level': 0,
                 'read_size_model1_resolution': (1216, 1936), #1936 1216 横 model1_resolution下
                 'scan_over': (120, 120),
                 'model1_input': (512, 512, 3),
                 'model2_input': (256, 256, 3),
                 'model1_resolution':0.586,
                 'model2_resolution':0.293,
                 'input_resolution':0.18,#0.
                 'cut':'3*5'}
    num_recom = 16
    size_set = Size_set(input_str)
    
    ratio = input_str['model1_resolution'] /input_str['input_resolution']
    w = int(input_str['read_size_model1_resolution'][0]*ratio)
    h = int(input_str['read_size_model1_resolution'][1]*ratio)
    
    pathfolder = r'Z:\LSB\tj5\positive'+'/'
    pathsave1 = r'F:\recom\model1_szsq646_model2_szsq1084\LSB\tj5\positive'+'/'
    pathsave2 = pathsave1 + 'model2/'
    pathsave2_core = None
   
    
    CZ = os.listdir(pathsave2)
    CZ = [cz[:-len('_p2.npy')] for cz in CZ if '_p2.npy' in cz]
    
    
    for filename in tqdm(CZ):
        print(filename)
        pathsave = pathsave1 + filename
        while not os.path.exists(pathsave):
            os.makedirs(pathsave)  
        startpointlist_split = np.load(pathsave1 + 'startpointlist_split.npy')   
        startpointlist_split = np.int16(np.array(startpointlist_split)*ratio)  
        startpointlist_big = np.load(pathsave1 + filename + '_s.npy')
        predict12 = np.load(pathsave2 + filename + '_p12.npy')
        num = len(predict12)//len(startpointlist_big)
        predict_big = [predict12[num*i:num*(i+1)].max() for i in range(len(startpointlist_big))]
        predict_index = [np.argmax(predict12[num*i:num*(i+1)]) for i in range(len(startpointlist_big))]
        
        
        
        Index = np.argsort(np.array(predict_big),axis=0)[::-1][:num_recom]
        pathfile = pathfolder + filename +input_str['filetype']
        ors = sdpc.Sdpc()
        ors.open(pathfile)
        attr = ors.getAttrs()
        size = (attr['width'], attr['height'])
        for di,index in enumerate(Index):
#            break
            img = ors.getTile(0, startpointlist_big[index][1],startpointlist_big[index][0], w, h)
            img = np.ctypeslib.as_array(img)
            img.dtype = np.uint8
            img = img.reshape((h, w, 3))
            img = img[...,::-1]
            row = startpointlist_split[predict_index[index]][1]
            col = startpointlist_split[predict_index[index]][0]
            img = cv2.rectangle(img.copy(), (col, row), (col+1664, row+1664), (255, 0, 0), 15) 
            cv2.imwrite(pathsave + '/{}_{}_{}_{}.tif'.format(di,predict_index[index],predict_big[index], startpointlist_big[index]), img[...,::-1])
            
