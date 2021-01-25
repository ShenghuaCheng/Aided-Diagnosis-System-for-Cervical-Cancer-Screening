# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: get_sample_score.py
@Date: 2020/2/18 
@Time: 12:10
@Desc: get all data scores
'''
import os
from functools import partial
from multiprocessing.dummy import Pool
import numpy as np
from utils.networks.wsi_related import resnet_clf
from utils.auxfunc.readin_val import readin_val

if __name__ == '__main__':
    label = 1
    # get data list
    config_root = r'F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\manu_yjy\model2\config\Itest'
    config_file = 'B_test_P'
    post_fix = '.txt'
    with open(os.path.join(config_root, config_file + post_fix), 'r') as f:
        data_list = f.readlines()
    data_list = [it.strip() for it in data_list if it]
    # model setting
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    weight_file = r'F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\manu_yjy\model2\exp3_re_270.h5'
    model = resnet_clf((256, 256, 3))
    model.load_weights(weight_file)
    # get save dir
    save_file = os.path.join(weight_file, weight_file.split('.h5')[0] + config_file + '.npy')
    # read in test img
    batchsize = 500
    re_size = (1843, 1843)
    crop_size = (256, 256)
    readinfunc = partial(readin_val, re_size=re_size, crop_size=crop_size)
    ind = 0
    scores = []
    while ind*batchsize < len(data_list):
        tmp_list = data_list[ind*batchsize: (ind+1)*batchsize]
        pool = Pool(16)
        img_batch = pool.map(readinfunc, tmp_list)
        pool.close()
        pool.join()
        preds = model.predict(np.stack(img_batch))
        preds.ravel()
        scores.append(list(preds))
        print('finish %d/%d' % ((ind+1)*batchsize, len(data_list)))
        ind += 1
    np.save(save_file, np.array(preds))




