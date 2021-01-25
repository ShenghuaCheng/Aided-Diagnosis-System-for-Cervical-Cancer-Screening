# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: Screening
@File: test_dataset.py
@Date: 2019/4/1 
@Time: 20:58
@Desc:
'''

import os
import random
from glob import glob
import math
import numpy as np

from read_in import read_in, multiprocess_read_in
import pandas as pd
class DataSet(object):
    def __init__(self,
                 data_root,
                 crop_num=7,
                 norm_flag=False,
                 enhance_flag=False,
                 scale_flag=False,
                 include_center=False,
                 mask_flag=False
                 ):
        self.data_root = data_root
        self.crop_num = crop_num
        self.norm_flag = norm_flag
        self.enhance_flag = enhance_flag
        self.scale_flag = scale_flag
        self.include_center = include_center
        self.__get_img_dir()
        
    def __get_img_dir(self):
        self.img_dir = {}
        self.class_label = {}
        self.each_class_num = {}
        self.all_size = {}
        key = ['0-20/n_0','20-50/n_1','50-100/n_2','100/n_5','n_8','n_9']
        df = pd.read_excel(self.data_root)  
        
        for i in range(len(df)):
            if df['flag'][i]==1:
                if pd.isnull(df['3th'][i]):
                    class_path = os.path.join(df['1th'][i],df['2th'][i])
                    img_dirs =  glob(os.path.join(class_path,'*.tif'))
                    self.img_dir[class_path] = img_dirs
                    self.class_label[class_path] = df['label'][i]
                    self.each_class_num[class_path] = int(df[key[0]][i]*df['inter_ratio'][i])
                    self.all_size[class_path] = {}
                    self.all_size[class_path]['img_size'] = df['img_size'][i]
                    self.all_size[class_path]['crop'] = df['crop'][i]
                    self.all_size[class_path]['resize'] = df['resize'][i]
                    self.all_size[class_path]['final'] = df['final'][i]
                else:
                    file3th = df['3th'][i].split(',')
                    for j in range(len(file3th)):
                        class_path = os.path.join(df['1th'][i],df['2th'][i])
                        class_path = os.path.join(class_path,file3th[j])
                        img_dirs =  glob(os.path.join(class_path,'*.tif'))
                        self.img_dir[class_path] = img_dirs
                        self.class_label[class_path] = df['label'][i]
                        self.each_class_num[class_path] = int(df[key[j]][i]*df['inter_ratio'][i])
                        self.all_size[class_path] = {}
                        self.all_size[class_path]['img_size'] = df['img_size'][i]
                        self.all_size[class_path]['crop'] = df['crop'][i]
                        self.all_size[class_path]['resize'] = df['resize'][i]
                        self.all_size[class_path]['final'] = df['final'][i]

    def get_read_in_img_dir(self):
        self.read_in_img_dir = {}
        classes = self.img_dir.keys()
        for c in classes:
            if len(self.img_dir[c]) > self.each_class_num[c]:
                self.read_in_img_dir[c] = random.sample(self.img_dir[c], self.each_class_num[c])
            else:
                self.read_in_img_dir[c] = self.img_dir[c]
        classes = self.read_in_img_dir.keys()
        return classes
    
    def get_img_and_label(self,c):
        img_dir_list = self.read_in_img_dir[c]
        init_size =  (int(self.all_size[c]['resize']),int(self.all_size[c]['resize']))
        enlarge_size = (int(self.all_size[c]['crop']),int(self.all_size[c]['crop']))
        train_size =  (int(self.all_size[c]['final']),int(self.all_size[c]['final']))
        img_batchs = multiprocess_read_in(16, dirImg=img_dir_list, init_size=init_size,
                                          enlarge_size=enlarge_size, train_size=train_size,
                                          crop_num=self.crop_num, norm_flag=self.norm_flag,
                                          enhance_flag=self.enhance_flag, scale_flag=self.scale_flag,
                                          include_center=self.include_center, mask_flag=False,dirMask=None)
        img_label = [float(self.class_label[c])]*len(self.read_in_img_dir[c])*self.crop_num
        
        img_total = []
        for img_batch in img_batchs:
            img_total += img_batch
        img_total = np.array(img_total)
        return img_total,img_label,img_dir_list




