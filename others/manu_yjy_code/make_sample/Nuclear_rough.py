# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 09:53:21 2019

@author: A-WIN10
"""
import sys
sys.path.append('../')
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import multiprocessing.dummy as multiprocessing
from tqdm import tqdm 
from utils.parameter_set import Size_set,TCT_set
from glob import glob
from openslide import OpenSlide

def Save_img(ors, startpointlist, k, pathsave,pathsave_name):
    img = ors.read_region(startpointlist[k], 0, (1024,1024))
    img = np.array(img)
    img = img[:, :, 0 : 3]
    cv2.imwrite(pathsave+pathsave_name[k] ,img[...,::-1])
  
pathfolder_svs_set, pathsave_set, _ = TCT_set()
for pathfolder_svs,pathsave1 in zip(pathfolder_svs_set['20x'],pathsave_set['20x']):
    pathsave_sample = 'H:/morphology/morphology_by_core/train/our_20x_Nothing/'
    pathsave = 'H:/morphology/morphology_by_core/train/our_20x_Nothing_origin/'
    CZ = os.listdir(pathsave1)
    CZ = [cz[:-len('_p.npy')] for cz in CZ if '_p.npy' in cz]
    for filename in tqdm(CZ):
        file = glob(pathsave_sample + pathsave1.split('/')[-2] +'_'+ filename +'*.tif')
        startpointlist = []
        pathsave_name = []
        for item in file:
            Y = int(item.split('_')[-2])
            X = int(item.split('_')[-1][:-4])
            startpointlist.append((Y,X))
            pathsave_name.append(item.split('\\')[-1])
        ors = OpenSlide(pathfolder_svs + filename + '.svs')
        pool = multiprocessing.Pool(20)
        for k in range(len(startpointlist)):
            pool.apply_async(Save_img,args=(ors, startpointlist, k, pathsave,pathsave_name))
        pool.close()
        pool.join()