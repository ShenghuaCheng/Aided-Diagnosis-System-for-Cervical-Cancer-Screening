# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:13:27 2019

@author: yujingya
"""
import sys
sys.path.append('../')
import json
from glob import glob
import numpy as np
from utils.function_set import region_proposal
from utils.parameter_set import Size_set,TCT_set
import os 
from tqdm import tqdm

# 转换路径
pathsave1 = r'F:\recom\model1_szsq646_model2_szsq1084\SZSQ_originaldata\Tongji_3th\positive\tongji_3th_positive_40x'+'/'
pathsave2 = pathsave1+ 'model2/'
pathsave_json = pathsave1 + 'all_results_json/'
if not os.path.exists(pathsave_json):
    os.makedirs(pathsave_json)  
    
# 参数设定
input_str = {'filetype': '.sdpc',# .svs .sdpc .mrxs
             'level': 0,
             'read_size_model1_resolution': (1216, 1936), #1936 1216 横 model1_resolution下
             'scan_over': (120, 120),
             'model1_input': (512, 512, 3),
             'model2_input': (256, 256, 3),
             'model1_resolution':0.586,
             'model2_resolution':0.293,
             'input_resolution':0.18,#
             'cut':'3*5'}
size_set = Size_set(input_str)
sizepatch_predict_small1 = size_set['sizepatch_predict_small1']
sizepatch_predict_small2 = size_set['sizepatch_predict_small2']

# 
filename_list = [x[len(pathsave1):-6] for x in glob(pathsave1 + '*_s.npy')]
for filename in tqdm(filename_list):
    # 读入npy数据
    startpoints_big_col_row = np.load(pathsave1 + filename + '_s.npy')
    predict_model1 = np.load(pathsave1 + filename + '_p.npy')
    startpointlist_split = np.load(pathsave1 + 'startpointlist_split.npy')
    feature = np.load(pathsave1 + filename + '_f.npy')
    dictionary = np.load(pathsave2 + filename + '_dictionary.npy')
    predict2 = np.load(pathsave2 + filename + '_p2.npy')
    # 转换成便于json存储的格式
    startpoints_big_col_row = startpoints_big_col_row.tolist()
    predict_model1 = [float(item[0]) for item in predict_model1]
    predict_model2 = []
    predict2 = [float(item[0]) for item in predict2]
    for index, item in enumerate(dictionary):
        a = predict2[int(np.sum(dictionary[:index])):int(np.sum(dictionary[:index])+item)]
        predict_model2.append(a)
    # 计算model1起始点 model1定位点 的绝对坐标
    startpoints_model1_col_row = []
    anchor_model1_col_row = []
    ratio = input_str['model1_resolution']/input_str['input_resolution']
    sizepatch_temp = (int(sizepatch_predict_small1[0]*ratio),int(sizepatch_predict_small1[1]*ratio))
    num_split = len(startpointlist_split)
    for index, item in enumerate(feature):
        startpoints_model1_row = startpoints_big_col_row[index//num_split][0] + int((startpointlist_split[index%num_split][0])*ratio)
        startpoints_model1_col = startpoints_big_col_row[index//num_split][1] + int((startpointlist_split[index%num_split][1])*ratio)
        startpoints_model1_col_row.append([startpoints_model1_row,startpoints_model1_col])
        anchor = region_proposal(item[...,1], sizepatch_temp, img=None, threshold=0.7)
        anchor = [[int(startpoints_model1_row + local[0]),int(startpoints_model1_col + local[1])] for local in anchor]
        anchor_model1_col_row.append(anchor)
    # 保存json文件
    content = {'startpoints_big_col_row': startpoints_big_col_row,
               'startpoints_model1_col_row': startpoints_model1_col_row,
               'anchor_model1_col_row': anchor_model1_col_row,
               'predict_model1': predict_model1,
               'predict_model2': predict_model2}
    with open(pathsave_json + '{}.json'.format(filename), "w") as f:
        json.dump(content,f)
        
# =============================================================================
#     with open(pathsave_json + '{}.json'.format(filename)) as f:
#         x = json.load(f)
# =============================================================================
