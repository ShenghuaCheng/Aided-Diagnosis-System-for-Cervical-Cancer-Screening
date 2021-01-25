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
    num_recom_min = 50
    num_recom_max = 200
    size_set = Size_set(input_str)
    pathfolder_set, _ = TCT_set()
    for pathfolder in pathfolder_set['szsq_sdpc'][-2:-1]:
#        break
        pathsave1 = r'F:\recom\model1_szsq728_model2_szsq1142\SZSQ_originaldata\Tongji_4th\positive'+'/'
        pathsave2 = pathsave1 + 'model2/'
        pathsave2_core = None
        pathsave_xml = pathsave1 + 'xml_200/'
        while not os.path.exists(pathsave_xml):
            os.makedirs(pathsave_xml)  
        CZ = os.listdir(pathsave2)
        CZ = [cz[:-len('_p2.npy')] for cz in CZ if '_p2.npy' in cz]
        
        
        for filename in tqdm(CZ):
            print(filename)
            object_recommend = Recommend(pathsave1, pathsave_xml,
                                     filename, input_str, num_recom_min, num_recom_max,
                                     pathsave2, pathsave2_core)
            startpointlist, predict, Sizepatch_small = object_recommend.get_startpointlist12()
            contourslist_small_recom, predict_recom, color = object_recommend.recommend_region(startpointlist, predict, Sizepatch_small)
            
            # 补充
            # original_xml_dir = os.path.join(pathsave_xml.split('xml_19_100/')[0], filename+'.xml')
            # tree = etree.parse(original_xml_dir)
            # tree_root = tree.getroot()
            # saveContours_xml([contourslist_small_recom[len(tree_root[0]):]],[predict_recom[len(tree_root[0]):]],
            #                  [color[len(tree_root[0]):]],pathsave_xml + filename + '.xml')
            saveContours_xml([contourslist_small_recom],[predict_recom],
                             [color],pathsave_xml + filename + '.xml')
           
   
