# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: inferrence_EF.py
@Date: 2020/5/14 
@Time: 10:47
@Desc: 前向推理EF所选测试图片，同时也是人工check的图片，记录推理结果，作图
'''
import os
import json
import numpy as np
import cv2
from others.manu_yjy_code.read_image.dataset import DataSet
from others.manu_yjy_code.networks.resnet50_2classes import ResNet
from utils.auxfunc.plot import plot_roc

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    # input_shape = (512, 512, 3)
    # weight = r"F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\manu_yjy\select_w\m1_pred_Block_333.h5"  # model1
    input_shape = (256, 256, 3)
    weight = r"F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\manu_yjy\exp2\new_E\AB_Block_339.h5"  # model2AB
    # weight = r"F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\manu_yjy\select_w\m2_Block_102.h5"  # model2

    src = 'F'
    md = '2'
    save_root = r"I:\20200929_EXP4\Tiles\models\model{}AB\{}".format(md, src)
    img_fld = r"I:\20200929_EXP4\RAWDATA\exp_4_yjy4000\sample\model{}\{}".format(md, src)
    img_list = os.listdir(img_fld)
    label_file = r"I:\20200929_EXP4\RAWDATA\exp_4_yjy4000\shuffle_list_{}_{}.json".format(src, md)
    with open(label_file, 'r') as f:
        shuffle_list = json.load(f)
    label_origin = shuffle_list["label"]

    print("read in ...")
    img_data = []
    labels = []
    img_dirs = []
    for idx in range(len(img_list)):
        img_dir = os.path.join(img_fld, "%d.png" % idx)
        if not os.path.exists(img_dir):
            print("%s does not exist" % img_dir)
            continue

        img = cv2.imread(img_dir)
        img = cv2.resize(img, input_shape[:2])
        img = (img/255-0.5)*2
        img_data.append(img)
        labels.append(label_origin[idx])
        img_dirs.append(img_dir)

    print("set model with weight: %s" %weight)
    Net = ResNet(input_shape=input_shape)
    Net.load_weights(weight)
    scores = Net.predict(np.stack(img_data), batch_size=32, verbose=1)

    os.makedirs(save_root, exist_ok=True)
    plot_roc(src, save_root, labels, scores.ravel())
    with open(os.path.join(save_root, "results.txt"), 'w') as f:
        for i in range(len(img_dirs)):
            f.write("%.5f, %d, %s\n" % (scores[i], labels[i], img_dirs[i]))

