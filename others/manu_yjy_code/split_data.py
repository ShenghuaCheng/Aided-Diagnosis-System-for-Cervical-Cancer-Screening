# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 22:26:40 2020

@author: A-WIN10
"""
import os
import numpy as np
import random 
from glob import glob
import cv2
#确认从训练集中抽取的验证集合理
path_root = r'H:\AdaptDATA\train\szsq\sdpc_tongji7\origin'
path_save = r'H:\yujingya\lsb_yjy\experience\2\config'
path_folder = ['n/n']#['ASCUS', 'HSIL', 'LSIL']#['n/n']#['p']#
filelist = []
for item in path_folder:
    train_label = len(os.listdir(os.path.join(path_root,item)))
    filelist +=os.listdir(os.path.join(path_root,item))
filelist = [item for item in filelist if '.db' not in item]
filelist = [item.split('_')[0] for item in filelist]
slicenamelist = np.unique(filelist).tolist()
train_slice = len(slicenamelist)

#for a in slicenamelist :
#    print(a)
#vailslice = []
#vailslice = vailslice[0].split('\n')
#vailslice = [a.split('\r')[0] for a in vailslice 
#vailslice =vailslice[:-1]
test_slice = 8
vailslice = random.sample(slicenamelist,test_slice)
Num={}
for x in path_folder:
    print(x)
    num = 0
    for y in vailslice:
        print(y)
    for y in vailslice:   
        num+=len(glob(path_root+'\\'+x+'\\'+y+'_*'))
        print(len(glob(path_root+'\\'+x+'\\'+y+'_*')))
    Num[x]=num
    print(num)
print(Num)
print(train_label-num)
print(train_slice-test_slice)
print(num)
print(test_slice)

#用txt文档形式存放数据
trainslice = [item for item in slicenamelist if item not in vailslice]
with open(path_save+'/D_train_tongji7_n.txt', 'w') as f:
    for x in path_folder:
        for y in trainslice:
            context=glob(path_root+'\\'+x+'\\'+y+'_*.tif')
            for item in context:
                f.write(item+'\n')

with open(path_save+'/D_vail_tongji7_n.txt', 'w') as f:
    for x in path_folder:
        for y in vailslice:
            context=glob(path_root+'\\'+x+'\\'+y+'_*.tif')
            for item in context:
                f.write(item+'\n')
                
                

# test
path_root = r'H:\AdaptDATA\test\szsq\sdpc_xyw1\origin'
path_save = r'H:\yujingya\lsb_yjy\experience\2\config'
path_folder =  ['n/n']#['ASCUS', 'HSIL', 'LSIL']#['n/n']
filelist = []
for item in path_folder:
    print(len(os.listdir(os.path.join(path_root,item))))
    filelist +=os.listdir(os.path.join(path_root,item))
filelist = [item for item in filelist if '.db' not in item]
filelist = [item.split('_')[0] for item in filelist]
slicenamelist = np.unique(filelist).tolist()
print(len(slicenamelist))

with open(path_save+'/F_test_xyw1_n.txt', 'w') as f:
    context=[]
    for x in path_folder:
         context+=glob(path_root+'\\'+x+'\\*.tif') 
    for item in context:
        f.write(item+'\n')
        

#for a in slicenamelist :
#    print(a)

a = r'H:\yujingya\lsb_yjy\experience\2\config\vail'
filee = os.listdir(a)
for b in filee:
    print('vail/'+b[:-4])
for b in filee:
    with open(a+'/'+b, 'r') as f:
        context=f.read()
    context= context.split('\n')[:-1]
    print(len(context))
for b in filee:
    with open(a+'/'+b, 'r') as f:
        context=f.read()
    context= context.split('\n')[:-1]
for b in filee:
    with open(a+'/'+b, 'r') as f:
        context=f.read()
    context= context.split('\n')[:-1]
    print(cv2.imread(context[0]).shape[0])
for b in filee:
    with open(a+'/'+b, 'r') as f:
        context=f.read()
    context= context.split('\n')[:-1]
    print(context[0][13:-4])