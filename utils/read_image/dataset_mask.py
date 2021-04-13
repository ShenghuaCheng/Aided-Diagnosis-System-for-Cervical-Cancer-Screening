# -*- coding:utf-8 -*-

import os
import random
from glob import glob

import numpy as np
import pandas as pd
import cv2

from keras.utils import to_categorical

from .read_in import multiprocess_read_in

class DataSet(object):
    def __init__(self,
                 data_root,
                 crop_num=1,
                 norm_flag=False,
                 enhance_flag=False,
                 scale_flag=False,
                 include_center=False,
                 mask_flag=False,
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
        self.mask_root = {}
        key = ['0-20/n_0','20-50/n_1','50-100/n_2','100/n_5','n_8','n_9']
        print(self.data_root)
        df = pd.read_excel(self.data_root)
        for i in range(len(df)):
            if df['flag'][i]==1:
                if pd.isnull(df['3th'][i]):
                    class_path = os.path.join(df['1th'][i],df['2th'][i])
                    img_dirs = glob(os.path.join(class_path, '*.*'))

                    self.mask_root[class_path] = df['mask_root'][i]

                    self.img_dir[class_path] = [img_d for img_d in img_dirs if '.tif' in img_d or '.png' in img_d or '.jpg' in img_d]
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

                        self.mask_root[class_path] = df['mask_root'][i]

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

    def get_img(self):
        img_total = []
        classes = self.img_dir.keys()
        for c in classes:
            img_dir_list = self.read_in_img_dir[c]
            init_size = (int(self.all_size[c]['resize']),int(self.all_size[c]['resize']))
            enlarge_size = (int(self.all_size[c]['crop']),int(self.all_size[c]['crop']))
            train_size = (int(self.all_size[c]['final']),int(self.all_size[c]['final']))
            img_batchs = multiprocess_read_in(16, dirImg=img_dir_list, init_size=init_size,
                                              enlarge_size=enlarge_size, train_size=train_size,
                                              crop_num=self.crop_num, norm_flag=self.norm_flag,
                                              enhance_flag=self.enhance_flag, scale_flag=self.scale_flag,
                                              include_center=self.include_center, mask_flag=False,
                                              dirMask=None)
            for img_batch in img_batchs:
                img_total += img_batch
        return img_total

    def __search_mask_name(self, img_dir, c):
        img_name = os.path.split(img_dir)[-1].split('.')[0]
        img_name = img_name.split('_')
        counter = 0
        name = ''
        while True:
            if counter == len(img_name):
                print('Can\'t find the mask of %s' % img_dir)
                self.read_in_img_dir[c].remove(img_dir)
                print('remove %d %d' % (len(self.read_in_img_dir[c]), len(self.read_in_mask_dir[c])))
                return 1
            else:
                name += img_name[counter] + '_'
                try:
                    img_mask_dir = glob(os.path.join(self.mask_root[c], name) + '.*')
                except:
                    print(img_dir, c)
                    raise TypeError
                counter += 1
                if len(img_mask_dir) == 1:
                    self.read_in_mask_dir[c].append(img_mask_dir[0])
                    print('append %d %d' % (len(self.read_in_img_dir[c]), len(self.read_in_mask_dir[c])))
                    return 0

    def __get_read_in_mask_dir(self):
        self.read_in_mask_dir = {}
        for c in self.img_dir.keys():
            if self.class_label[c]:
                self.read_in_mask_dir[c] = []
                search_list = self.read_in_img_dir[c].copy()
                for img_dir in search_list:
                    mask_flag = self.__search_mask_name(img_dir, c)
                print(len(self.read_in_mask_dir[c]), len(self.read_in_img_dir[c]))

    def get_img_with_mask(self):
        img_total = []
        imgmask_total = []
        self.__get_read_in_mask_dir()
        classes = self.img_dir.keys()
        for c in classes:
            img_dir_list = self.read_in_img_dir[c]
            init_size = (int(self.all_size[c]['resize']),int(self.all_size[c]['resize']))
            enlarge_size = (int(self.all_size[c]['crop']),int(self.all_size[c]['crop']))
            train_size = (int(self.all_size[c]['final']),int(self.all_size[c]['final']))
            if self.class_label[c]:
                img_batchs = multiprocess_read_in(16, dirImg=img_dir_list, init_size=init_size,
                                                  enlarge_size=enlarge_size, train_size=train_size,
                                                  crop_num=self.crop_num, norm_flag=self.norm_flag,
                                                  enhance_flag=self.enhance_flag, scale_flag=self.scale_flag,
                                                  include_center=self.include_center, mask_flag=True,
                                                  dirMask=self.read_in_mask_dir[c])
                # img_mask_batchs = [np.expand_dims(cv2.resize((img[0][1][:, :, 0] / 255).astype(np.float32), (64, 64)), axis=-1)
                #                    for img in img_batchs]
                img_mask_batchs = [to_categorical(cv2.resize((img[0][1][:, :, 0] / 255).astype(np.uint8), (16, 16)), 2)
                                   for img in img_batchs]

                img_batchs = [img[0][0] for img in img_batchs]
            else:
                img_batchs = multiprocess_read_in(16, dirImg=img_dir_list, init_size=init_size,
                                                  enlarge_size=enlarge_size, train_size=train_size,
                                                  crop_num=self.crop_num, norm_flag=self.norm_flag,
                                                  enhance_flag=self.enhance_flag, scale_flag=self.scale_flag,
                                                  include_center=self.include_center, mask_flag=False,
                                                  dirMask=None)
                # img_mask_batchs = list(np.zeros((len(img_batchs), 64, 64, 1), dtype=np.float32))
                img_mask_batchs = list(np.zeros((len(img_batchs), 16, 16), dtype=np.uint8))
                img_mask_batchs = [to_categorical(img, 2) for img in img_mask_batchs]

                img_batchs = [img[0] for img in img_batchs]

            img_total += img_batchs
            imgmask_total += img_mask_batchs

        return img_total, imgmask_total



    def get_label(self):
        img_label = []
        classes = self.img_dir.keys()
        total = 0
        for c in classes:
            img_label += [float(self.class_label[c])]*len(self.read_in_img_dir[c])*self.crop_num
            print(c + ':' + str(len(self.read_in_img_dir[c])) + '/张')
            total += len(self.read_in_img_dir[c])
        print('共计:' + str(total) + '/张')
        return img_label




