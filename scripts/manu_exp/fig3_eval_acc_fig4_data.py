# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: fig3_eval_acc_fig4_data.py
@Date: 2020/12/21 
@Time: 14:46
@Desc: 根据fig4做t-SNE的数据来出消融实验的结果
'''
import numpy as np
import pandas as pd
import glob2
import os
from sklearn import metrics

MODEL_NAMES = [
    'Origin', 'Enhanced', 'Mined', 'Baseline', 'HR_model', 'LR_model'
]


def parse_res(res_dir):
    res = np.load(res_dir)



if __name__ == '__main__':
    root = r'I:\20201218_Manu_Fig4\ABCDEF_features'
    for mn in MODEL_NAMES:
        res_list = glob2.glob(os.path.join(root, mn, 'feature_data_*.npz'))
        for res_d in res_list:


