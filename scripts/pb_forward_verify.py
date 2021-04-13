# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np

from core.h5_to_pb import pb_forward
from core.networks import ResNet

if __name__ == "__main__":
    # =============== .pb Config ===============
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    pb_root = r'pbs'
    pb_file = '.pb'

    pb_dir = os.path.join(pb_root, pb_file)

    input_name = 'input_1:0'
    output_name_preds = 'dense_2/Sigmoid:0'

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
    for i, p in enumerate(preds):
        print(p.ravel()[0])

