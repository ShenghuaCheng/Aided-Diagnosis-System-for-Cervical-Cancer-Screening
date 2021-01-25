# -*- coding: utf-8 -*-
'''
@Author: LiuSibo
@Project: Screening
@File: pb_forward_verify.py
@Date: 2019/4/25
@Time: 10:52
@Desc:  调用pb进行前向的python代码
'''
import os
import cv2
import numpy as np

from core.h5_to_pb import pb_forward
from core.networks import ResNet

if __name__ == "__main__":
    # =============== .pb Config ===============
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    pb_root = r'F:\LiuSibo\Models\tongji34_update_190703\pbs'
    pb_file = 'szsq_model2_1084_pred.pb'

    pb_dir = os.path.join(pb_root, pb_file)

    input_name = 'input_1:0'
    output_name_preds = 'dense_2/Sigmoid:0'
    # output_name_local = 'conv2d_1/truediv:0'

    # =============== test data ===============
    data_root = r'H:\weights\w_szsq\model1_local\pb\imgs'
    data_list = os.listdir(data_root)
    data = []
    for img_name in data_list:
        img = cv2.imread(os.path.join(data_root, img_name))
        img = (img/255.-0.5)*2
        img = cv2.resize(img, (256, 256))
        data.append(img)
    data = np.stack(data)

    # =============== pb forward ===============
    preds = pb_forward(pb_dir, input_name, output_name_preds, data)
    # locals = pb_forward(pb_dir, input_name, output_name_local, data)
    # import cv2
    for i, p in enumerate(preds):
    #     cv2.imwrite(r'H:\weights\w_szsq\model1_local\pb\locals\%r_%r.tif' %(i, 0), p[...,0]*255)
    #     cv2.imwrite(r'H:\weights\w_szsq\model1_local\pb\locals\%r_%r.tif' %(i, 1), p[...,1])
        print(p.ravel()[0])

