# -*- coding: utf-8 -*-
"""
Created on Tue May 21 10:15:15 2019

@author: 03
"""
import os 
from glob import glob
import numpy as np
import pandas as pd
type_ = 'test'
weights_folder = r'H:\weights\w_szsq_tongji34_update_190703\model2_pred\eval'
weights_filelist = glob(weights_folder  + '/*.h5')
data_root = weights_folder + '\%s.xlsx'%type_

DF = pd.DataFrame()
key = ['0-20/n_0','20-50/n_1','50-100/n_2','100/n_5','n_8','n_9']
for weights_path in weights_filelist:
    print(weights_path)
#    break
    save_path = os.path.join(weights_folder, weights_path.split('\\')[-1][:-3])
    path = os.path.join(save_path,'%s_acc.txt'%type_)
    acc = []
    for line in open(path,"r") :
        if '7 crops acc:' in line:
            acc.append(line.split(':')[-1])
    df = pd.read_excel(data_root)  
    for i in range(len(df)):
        try:
            if df['flag'][i]==1:
                if pd.isnull(df['3th'][i]):
                    class_path = os.path.join(df['1th'][i],df['2th'][i])
                    img_dirs =  glob(os.path.join(class_path,'*.tif'))
                    if len(img_dirs)!=0:
                        df[key[0]][i] = acc[0]
                        del acc[0]
                else:
                    file3th = df['3th'][i].split(',')
                    for j in range(len(file3th)):
                        class_path = os.path.join(df['1th'][i],df['2th'][i])
                        class_path = os.path.join(class_path,file3th[j])
                        img_dirs =  glob(os.path.join(class_path,'*.tif'))
                        if len(img_dirs)!=0:
                            df[key[j]][i] = acc[0]
                            del acc[0]
        except:
            break
    df_temp = df[key]
    insertRow = pd.DataFrame([[weights_path.split('\\')[-1][:-3],0.,0.,0.,0.,0.]],columns=key)
    df_temp = pd.concat([insertRow, df[key]]) 
    DF = pd.concat([DF, df_temp],axis = 1,ignore_index=True) 
    
DF.to_excel(os.path.join(weights_folder,'result_%s.xlsx'%type_),index = False)