# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem_tf20
@File: verify_model.py
@Date: 2019/12/4 
@Time: 11:14
@Desc: 脚本用于验证转移到tf2.0后的模型是否与原模型的预测结果一致
'''
import time
import os
import cv2
import numpy as np
from utils.networks.wsi_related import resnet_encoder
from utils.networks.model1 import Resnet_location, Resnet_location_and_predict

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # config
    save_file = r'E:\LiuSibo\Project\AidedDiagnosisSystem_tf20\doc\verified_160_RN50M1.txt'
    w_local_file = r'H:\weights\000000_ValidModels\190920_1010_model1_szsq700_model2_szsq658\szsq_model1_700_36_local.h5'  # model1
    w_file = r'H:\weights\000000_ValidModels\190920_1010_model1_szsq700_model2_szsq658\szsq_model1_700_pred.h5'  # model1
    # w_file = r'H:\weights\000000_ValidModels\190920_1010_model1_szsq700_model2_szsq658\szsq_model2_658_pred.h5'  # model2
    in_size = 512
    # read in imgs
    img_root = r'H:\AdaptDATA\test\3d\sfy1\ASCUS'
    img_dirs = [os.path.join(img_root, n) for n in os.listdir(img_root) if '.tif' in n]
    imgs = []
    for d in img_dirs[:1000]:
        img = cv2.imread(d)
        img = cv2.resize(img, (768, 768))
        img = img[128:640, 128:640, :]
        # img = cv2.resize(img, (1536, 1536))
        # img = img[640:896, 640:896, :]
        img = (img/255.-0.5)*2
        imgs.append(img)
    # create model instance
    # model = resnet_encoder((in_size, in_size, 3))
    model = Resnet_location_and_predict((in_size, in_size, 3))
    model.load_weights(w_local_file, by_name=True)
    model.load_weights(w_file, by_name=True)
    # inference
    since = time.time()
    preds = model.predict(np.stack(imgs))
    print(time.time()-since)
    # # write results
    # with open(save_file, 'a') as f:
    #     for (n, p) in zip(img_dirs[:50], preds[0].ravel()):
    #         f.write('%f, %s\n' % (p, n))
    # write results
    np.save(r'F:\LiuSibo\Project\AidedDiagnosisSystem_tf20\doc\160_preds_withlocal.npy', preds[1])

