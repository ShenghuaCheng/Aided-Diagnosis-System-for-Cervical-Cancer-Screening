# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: Screening
@File: eval_2classes_model2.py
@Date: 2019/4/16 
@Time: 9:57
@Desc:
'''
import sys
sys.path.append('../')
import os
import numpy as np
import pandas as pd
from keras.utils import multi_gpu_model
import keras.backend as K
import tensorflow as tf
from others.manu_yjy_code.read_image.dataset import DataSet
from others.manu_yjy_code.networks.resnet50_2classes import ResNet
from glob import glob
import shutil


weights_filelist = [
    r'F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\manu_yjy\model1\new_E\m1_pred_Block_333.h5',
    r'F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\manu_yjy\model1\weights\szsq_model1_700_pred.h5',
    r'F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\manu_yjy\model1\weights\Block_330.h5'
]
# weights_filelist = [r'F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\manu_yjy\model1\new_E\m1_pred_Block_333.h5']
# weights_filelist = glob(r'F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\manu_yjy\model1\weights\Block_333.*')
path_result = r'F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\manu_yjy\model1\new_E\result_512'
os.makedirs(path_result, exist_ok=True)

gpu_num = 1
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

config_path = r'F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\manu_yjy\model1\config'
DataSet = {x: DataSet(config_path,
                      'eval_E_512.xlsx',
                      x,
                      crop_num=1,
                      norm_flag=True,
                      enhance_flag=False,
                      scale_flag=False,
                      include_center=False,
                      mask_flag=False) for x in ['Itest']}

Net = ResNet(input_shape=(512, 512, 3))

t = 'Itest'
classes = DataSet[t].get_read_in_img_dir()
for c in classes:
    label_data = np.stack(DataSet[t].get_label(classes=[c]))
    img_data = np.stack(DataSet[t].get_img(classes=[c]))
    for weights_path in weights_filelist:
        print(c)
        print('Set model %s' % weights_path)
        Net.load_weights(weights_path)
         # 预测及统计
        preds_prob = Net.predict(img_data, batch_size=16*gpu_num, verbose=1)
#        np.save(path_result+'/'+weights_path.split('\\')[-2]+c.split('/')[1]+'.npy',np.array(preds_prob))

        if label_data[0]==1:
            acc=np.sum(preds_prob>0.5)/len(preds_prob)
        elif label_data[0]==0:
            acc=np.sum(preds_prob<0.5)/len(preds_prob)
        else:
            print('error!')
        print(acc)
        with open(path_result+'/'+c.split('/')[1]+'.txt', 'a') as f:
            f.write(str(acc)+'\n')
# 
# # =============================================================================
# # 
# # filist = glob(r'H:\AdaptDATA\test\szsq\sdpc_xyw1\origin\n\n\052800031_*.tif')
# # 
# # for a in filist:
# #     os.remove(a)
# # =============================================================================
# 
# =============================================================================
record_col=[]
for c in classes:
    with open(path_result+'/'+c.split('/')[1]+'.txt', 'r') as f:
        acc = f.read()
    record_col.append(acc.split('\n')[:-1])
    
record_col= np.vstack(record_col)
record_df = pd.DataFrame(record_col)
# change the index and column name
record_df.index = classes
record_df.columns = [a.split('\\')[-1][:-3] for a in weights_filelist]
writer = pd.ExcelWriter(path_result+'/result.xlsx')
record_df.to_excel(writer, 'Sheet1', float_format='%.5f')  # float_format 控制精度
writer.save()    