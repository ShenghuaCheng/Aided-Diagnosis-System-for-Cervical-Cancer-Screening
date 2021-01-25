# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 11:30:55 2019

@author: A-WIN10
"""
import os
from glob import glob
import pandas as pd
import numpy as np
import cv2
 
DF = pd.DataFrame(columns=['1th','2th','3th','4th','size','num'])
path_1 = r'H:\AdaptDATA\train'
path_2 = ['3d', 'our', 'szsq']
Num = []
Size = []
for path_22 in path_2:
    path_3 = os.listdir(path_1 +'/'+ path_22)
    for path_33 in path_3:
        path_4 = os.listdir(path_1 +'/'+ path_22+'/'+ path_33)
        if 'origin' in path_4:
            path_33 = path_33 + '/origin'
            path_4 = os.listdir(path_1 +'/'+ path_22+'/'+ path_33)
        for path_44 in path_4:
            if 'n' == path_44 and len(os.listdir(path_1 +'/'+ path_22+'/'+ path_33 +'/'+ path_44))==6:
                path_4.remove('n')
                path_4.append('n/n_0')
                path_4.append('n/n_1')
                path_4.append('n/n_2')
                path_4.append('n/n_5')
                path_4.append('n/n_8')
                path_4.append('n/n_9')
        for path_44 in path_4:
            pathimg = glob(path_1 +'/'+ path_22+'/'+ path_33 +'/'+ path_44 +'/*.tif')
            try:
                img = cv2.imread(pathimg[0])
                shape = img.shape[0]
            except:
                shape = 0
            num = len(pathimg)
            
            print('shape_{}/num_{}'.format(shape,num))
            Num.append(num)
            Size.append(shape)
DF['size'] = Size
DF['num'] = Num
index = 0
for path_22 in path_2:
    path_3 = os.listdir(path_1 +'/'+ path_22)
    for path_33 in path_3:
        path_4 = os.listdir(path_1 +'/'+ path_22+'/'+ path_33)
        if 'origin' in path_4:
            path_33 = path_33 + '/origin'
            path_4 = os.listdir(path_1 +'/'+ path_22+'/'+ path_33)
        for path_44 in path_4:
            if 'n' == path_44 and len(os.listdir(path_1 +'/'+ path_22+'/'+ path_33 +'/'+ path_44))==6:
                path_4.remove('n')
                path_4.append('n/n_0')
                path_4.append('n/n_1')
                path_4.append('n/n_2')
                path_4.append('n/n_5')
                path_4.append('n/n_8')
                path_4.append('n/n_9')
        for path_44 in path_4:
            DF['1th'][index] = path_1
            DF['2th'][index] = path_22
            DF['3th'][index] = path_33
            DF['4th'][index] = path_44
            index+=1
DF.to_excel(path_1+'/num.xlsx',index = False)
# =============================================================================
# data_root = r'H:\AdaptDATA\test\szsq\sdpc_sfy3\origin\n'
# path_list = os.listdir(data_root)
# for path in path_list:
# #    break
#     path_ = os.path.join(data_root, path)
#     if len(glob(path_+'/*.tif'))==len(glob(path_.replace('origin','gamma')+'/*.tif')):
#         print('yes_{}_{}'.format(len(glob(path_+'/*.tif')),path))
#     else:
#         print(path_.replace('origin','gamma'))
# =============================================================================
