# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 10:54:44 2019
@author: A-WIN10
"""
import sys
sys.path.append('../')
import os
from glob import glob
import numpy as np
from utils.parameter_set import test_sample
# =============================================================================
# path = r'H:\AdaptDATA\train\szsq\szsq_sdpc_tongji3\gamma\190705'
# 
# filelist = glob(path+'/*.tif')
# filetype = [item[:-4].split('_')[-1] for item in filelist]
# filetype = np.unique(filetype)
# 
# for type_ in filetype:
#     if not os.path.exists(path.replace('190705',type_)):
#         os.makedirs(path.replace('190705',type_))
#     file = glob(path+'/*'+ type_ + '.tif')
#     for cz in file:
#         os.rename(cz,cz.replace('190705',type_))
# =============================================================================
path = r'H:\AdaptDATA\train\szsq\sdpc_sfy1\gamma\ASCUS'
testlist = test_sample['test_p_sfy1']
for file in testlist:
#    break
    filelist = glob(path +'/'+ file+'_*.tif')
    for cz in filelist:
#        break
        if not os.path.exists(path.replace('train','test')):
            os.makedirs(path.replace('train','test'))
        os.rename(cz,cz.replace('train','test'))
        