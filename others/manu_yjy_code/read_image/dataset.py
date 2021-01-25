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
from others.manu_yjy_code.read_image.read_in import read_in, multiprocess_read_in
import pandas as pd


class DataSet(object):
    """未完成的数据集类，后续添加mask等读入
    """
    def __init__(self,
                 data_root,
                 xlsx_name,
                 category,
                 crop_num=1,
                 norm_flag=False,
                 enhance_flag=False,
                 scale_flag=False,
                 include_center=False,
                 mask_flag=False,
                 ):
        self.data_root = data_root
        self.xlsx_name = xlsx_name
        self.category = category
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
        print(self.data_root+'/'+self.xlsx_name)
        df = pd.read_excel(self.data_root+'/'+self.xlsx_name,sheetname = self.category)
        #df = pd.read_excel(config,sheet_name = 'test')
        
        for i in range(len(df)):
            class_path = df['txt_name'][i]
            print(self.data_root+'/'+class_path+'.txt')
            with open(self.data_root+'/'+class_path+'.txt', 'r') as f:
                img_dirs = f.read()
            img_dirs=img_dirs.split('\n')[:-1]
            self.img_dir[class_path] = img_dirs
            self.class_label[class_path] = df['label'][i]
            self.each_class_num[class_path] = int(np.ceil(df['num_block'][i]*df['inter_ratio'][i]))
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
        return classes
    
    def get_img(self,classes=None):
        img_total = []
        if classes==None:
            classes = self.img_dir.keys()
        
        for c in classes:
            img_dir_list = self.read_in_img_dir[c]
            init_size = (int(self.all_size[c]['resize']),int(self.all_size[c]['resize']))
            enlarge_size =  (int(self.all_size[c]['crop']),int(self.all_size[c]['crop']))
            train_size =  (int(self.all_size[c]['final']),int(self.all_size[c]['final']))
            img_batchs = multiprocess_read_in(16, dirImg=img_dir_list, init_size=init_size,
                                              enlarge_size=enlarge_size, train_size=train_size,
                                              crop_num=self.crop_num, norm_flag=self.norm_flag,
                                              enhance_flag=self.enhance_flag, scale_flag=self.scale_flag,
                                              include_center=self.include_center, mask_flag=False,
                                              dirMask=None)
            for img_batch in img_batchs:
                img_total += img_batch
                
        return img_total
    

    def get_label(self,classes=None):
        img_label = []
        if classes==None:
            classes = self.img_dir.keys()
            
        total = 0
        for c in classes:
            img_label += [float(self.class_label[c])]*len(self.read_in_img_dir[c])*self.crop_num
            print(c + ':' + str(len(self.read_in_img_dir[c])) + '/张')
            total += len(self.read_in_img_dir[c])
        print('共计:' + str(total) + '/张')
        return img_label




