# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 00:43:20 2018
@author: yujingya
"""
import os
import numpy as np
from skimage import measure
from tqdm import tqdm
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
    return local

def Model1_location(online = False):
    startpointlist0 = np.load(path + filename + '_s.npy')
    feature = np.load(path + filename + '_f.npy') 
    startpointlist_split = np.load(path + 'startpointlist_split.npy') 
    startpointlist_split = np.int16(np.array(startpointlist_split)*ratio) 
    # 计算model2定位坐标起始点并读图
    startpointlist2 = []
    for index, item in enumerate(feature):
        Local_3 = region_proposal(item[...,1], (int(512*ratio),int(512*ratio)), img=None, threshold=0.7)
        startpointlist2_temp = []
        for local in Local_3:
            startpointlist_2_x = startpointlist0[index//num][1] + startpointlist_split[index%num][1] + local[1]
            startpointlist_2_y = startpointlist0[index//num][0] + startpointlist_split[index%num][0] + local[0]
            startpointlist2_temp.append((startpointlist_2_y,startpointlist_2_x))
        startpointlist2.append(startpointlist2_temp)
    np.save(path + filename + '_l.npy',startpointlist2)    
#x 行 Y 列  
    return None
if __name__ == '__main__':
    path = 'F:/recom/340_adapt/our/20x/Positive/Shengfuyou_5th/'
    ratio = 2# our_10x:1 our_20x:2 3d:2.4
    num = 15
    filelist = os.listdir(path)
    filelist = [item[:-len('_p.npy')] for item in filelist if '_p.npy' in item]
    for filename in tqdm(filelist):
        Model1_location()