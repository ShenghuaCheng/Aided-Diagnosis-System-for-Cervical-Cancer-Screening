# -*- coding: utf-8 -*-
"""
Created on Thu May 16 21:23:56 2019

@author: A-WIN10
"""
import sys 
sys.path.append('../')
from utils.parameter_set import Size_set,TCT_set
import numpy as np
import os
pathfolder_set, _ = TCT_set()
for pathfolder in pathfolder_set['10x'][4:5]:
#    break
    pathsave1 = 'F:/recom/340_adapt/our/10x/Positive/unsure/'
    pathsave2 = pathsave1 + 'model2/'
    CZ = os.listdir(pathsave1)
    CZ = [cz[:-len('_p.npy')] for cz in CZ if '_p.npy' in cz]
    for filename in CZ:
        predict1 = np.load(pathsave1 + filename + '_p.npy')
        predict2 = np.load(pathsave2 + filename + '_p2.npy',)
        dictionary = np.load(pathsave2 + filename + '_dictionary.npy',)
        
        predict12 = []
        for index, item in enumerate(dictionary):
            if item!=0:
                print('yes')
                a = predict2[int(np.sum(dictionary[:index])):int(np.sum(dictionary[:index])+item)]
                a = a.max()
                predict12.append(a)
            else:
                predict12.append(predict1[index])
        predict12 = np.vstack(predict12)
        np.save(pathsave2 + filename + '_p12.npy',predict12)