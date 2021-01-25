# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: sample_adapt_data.py
@Date: 2020/6/19 
@Time: 17:02
@Desc:
'''
""" 未完成的抽取standard数据集代码，已经由fql完成
"""

import os
import random
import numpy as np
import glob2
import pandas as pd
import cv2

if __name__ == '__main__':
    nb_file = r"F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\sample_AdaptData\select_num.xlsx"
    train_file = r"F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\sample_AdaptData\train_num.xlsx"
    test_file = r"F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\sample_AdaptData\test_num.xlsx"
    # 读入excel
    roots = {
        "train": pd.read_excel(train_file, sheet_name="path", index_col="source"),
        "test": pd.read_excel(test_file, sheet_name="path", index_col="source")
    }
    nb_imgs = pd.read_excel(nb_file, index_col="source")
    # 获取抽样列表
    for item in nb_imgs.index.tolist()[:-1]:
        print(item)
        for type in nb_imgs.columns.tolist():
            nb = nb_imgs[type][item]  # 读入总数
            index = roots[type].index == item
            paths = roots[type][index]  # 读入该类所有路径
            if "pos" in item:
                # 阳性随机抽样
                img_list = []
                for path_se in zip(paths["1th"], paths["2th"]):
                    img_list += [img for img in glob2.glob(os.path.join(path_se[0], path_se[1], "*.*")) if ".tif" in img or ".jpg" in img or ".png" in img]
                sample_list = random.sample(img_list, np.minimum(nb, len(img_list)))  # 样本数量不足则全取
            elif "neg" in item:
                # 阴性困难样本按比例抽样 1：9
                img_list = []
                img_list_nplus = []
                for path_se in zip(paths["1th"], paths["2th"]):
                    if "nplus" in path_se[1] or "n_5" in path_se[1] or "n_8" in path_se[1] or "n_9" in path_se[1]:
                        # nplus类关键字
                        img_list_nplus += [img for img in glob2.glob(os.path.join(path_se[0], path_se[1], "*.*")) if ".tif" in img or ".jpg" in img or ".png" in img]
                    else:
                        img_list += [img for img in glob2.glob(os.path.join(path_se[0], path_se[1], "*.*")) if ".tif" in img or ".jpg" in img or ".png" in img]

                    # 处理没有普通阴性或者困难阴性的情况
                    if len(img_list)==0:
                        sample_list = random.sample(img_list_nplus, np.minimum(nb, len(img_list_nplus)))
                    elif len(img_list_nplus==0):
                        sample_list = random.sample(img_list, np.minimum(nb, len(img_list)))
                    else:
                        sample_list = random.sample(img_list_nplus, np.minimum(int(nb*0.1), len(img_list_nplus))) + random.sample(img_list, np.minimum(int(nb*0.9), len(img_list)))
