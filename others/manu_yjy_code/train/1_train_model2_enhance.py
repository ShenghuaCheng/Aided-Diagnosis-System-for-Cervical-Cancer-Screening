# -*- coding:utf-8 -*-
'''
@Author: Yujingya
@Project: Screening
@File: train.py
@Date: 2019/4/4 
@Time: 16:32
@Desc: 此脚本训练实验一、数据增强的重要性Model2
'''

import os
import time
import random
import pandas as pd
import numpy as np
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras.callbacks import TensorBoard
import tensorflow as tf
import sys

from others.manu_yjy_code.networks.resnet50_2classes import ResNet
from others.manu_yjy_code.read_image.dataset import DataSet

# =====================================================================================================================
# DataSet Setting
# ================================== ===================================================================================
num_gpu = 1
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

learning_rate = 1e-4
config_path = r'F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\manu_yjy\exp1\config'
path_checkpoints = r'F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\manu_yjy\exp1\weights\1_stage'

file_pretrain_weights = r'F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\manu_yjy\exp1\origin_flip_Block_200.h5'
start = 0
end = 800 #1epoch=20blocks
Frozen_layers = 37

enhance_flag = {'train': True, 'vail': False}
scale_flag = {'train': False, 'vail': False}

DataSet = {x: DataSet(config_path,
                      '1_stage_37.xlsx',
                      x,
                      crop_num=1,
                      norm_flag=True,
                      enhance_flag=enhance_flag[x],
                      scale_flag=scale_flag[x],
                      include_center=False,
                      mask_flag=False) for x in ['train', 'vail']}
# =====================================================================================================================
# Model Setting
# =====================================================================================================================

path_log = os.path.join(path_checkpoints, 'logs')
hist_file = 'hist.txt'
if not os.path.exists(path_checkpoints):
    os.makedirs(path_checkpoints)
if not os.path.exists(path_log):
    os.makedirs(path_log)

print('Setting Model')
if num_gpu > 1:
    with tf.device('/cpu:0'):
        model = ResNet(input_shape=(256, 256, 3),frozen_layers=Frozen_layers)  # 37
        model.compile(optimizer=Adam(lr=learning_rate), loss=["binary_crossentropy"], metrics=["binary_accuracy"])
        if file_pretrain_weights is not None:
            print('Load weights %s' % file_pretrain_weights)
            model.load_weights(file_pretrain_weights)
    parallel_model = multi_gpu_model(model, gpus=num_gpu)
    parallel_model.compile(optimizer=Adam(lr=learning_rate), loss=["binary_crossentropy"], metrics=["binary_accuracy"])
else:
    parallel_model = ResNet(input_shape=(256, 256, 3), frozen_layers=37)
    parallel_model.compile(optimizer=Adam(lr=learning_rate), loss=["binary_crossentropy"], metrics=["binary_accuracy"])
    if file_pretrain_weights is not None:
        print('Load weights %s' % file_pretrain_weights)
        parallel_model.load_weights(file_pretrain_weights)
# parallel_model.save(os.path.join(path_checkpoints, 'Block_990_szsq_model2_0_37_model.h5'))
# =====================================================================================================================
# Training
# =====================================================================================================================

for block in range(start, end):
    print('Training with block %d' % block)
    print('Reading ...')
    since = time.time()
    img_data = {}
    label_data = {}
    try:
        for data in ['train', 'vail']:
            DataSet[data].get_read_in_img_dir()
            label_data[data] = np.stack(DataSet[data].get_label())
            img_data[data] = np.stack(DataSet[data].get_img())
    except:
        continue
    print("\nTotal train images: %s" % (len(img_data['train'])),
          "\nTotal vail images: %s" % (len(img_data['vail'])),
          "\n%.2f sec per 1000 image" % ((time.time() - since) * 1000 / (len(img_data['train']) + len(img_data['vail']))))

    tensorboard = TensorBoard(log_dir=path_log)

    hist = parallel_model.fit(img_data['train'], label_data['train'], batch_size=32*num_gpu, epochs=1, verbose=1,
                              validation_data=(img_data['vail'], label_data['vail']), callbacks=[tensorboard])

    with open(os.path.join(path_log, hist_file), 'a') as f:
        f.write('Block' + str(block) + '\n')
        f.write(str(hist.history) + '\n')

    if block % 5 == 0:
        if num_gpu > 1:
            model.save_weights(os.path.join(path_checkpoints, 'Block_%s.h5' % block))
        else:
            parallel_model.save_weights(os.path.join(path_checkpoints, 'Block_%s.h5' % block))
