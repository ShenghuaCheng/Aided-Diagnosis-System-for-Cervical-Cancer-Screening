# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: fig3_get_roc_point.py
@Date: 2020/12/31 
@Time: 10:14
@Desc:  本脚本用于获取FIG3作图所需的ROC散点 存入Excel
'''
import os
import pandas as pd
import numpy as np
from glob2 import glob
from sklearn import metrics

def parse_txt_results(txt_dir):
    with open(txt_dir, 'r') as f:
        lines = f.readlines()
    lines = [l.strip().split(', ')[:2] for l in lines]
    scores = [float(l[0]) for l in lines]
    labels = [int(l[1]) for l in lines]
    return scores, labels


if __name__ == '__main__':
    # save excel
    save_excel = r'I:\20201221_Manu_Fig3\fig3_roc_scatter.xlsx'
    wrtr = pd.ExcelWriter(save_excel)

    # # 经过查证，已有现成数据，故略过
    # # 先处理HR和BASELINE
    # res_root = {
    #     'HR_model': r'I:\20200929_EXP4\Tiles\models\model2',
    #     'Baseline': r'I:\20200929_EXP4\Tiles\models\model2AB',
    # }
    # for mn, rt in res_root.items():
    #     all_scores = []
    #     all_labels = []
    #     for bat in ['E', 'F']:
    #         scores, labels = parse_txt_results(os.path.join(rt, bat, 'results.txt'))
    #         all_scores += scores
    #         all_labels += labels
    #     all_scores = np.array(all_scores)
    #     all_labels = np.array(all_labels)
    #     fpr, tpr, thre = metrics.roc_curve(all_labels, all_scores)
    #     auc = metrics.roc_auc_score(all_labels, all_scores)


    npy_fld = r'F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\manu_yjy\model2'
    model_types = ['origin_B', 'enhance_B', 're_hard_B']
    src = 'B'

    labels = []
    scores = []
    for model_type in model_types:
        # parse npy to get score ande label
        p_s = []
        n_s = []
        for d in glob(os.path.join(npy_fld, '%s*_p.npy' % model_type)):
            p_s += list(np.load(d).ravel())
        for d in glob(os.path.join(npy_fld, '%s*_n.npy' % model_type)):
            n_s += list(np.load(d).ravel())
        print('pos num: {}\nneg num: {}'.format(len(p_s), len(n_s)))
        s = p_s + n_s
        label = list(np.ones_like(p_s)) + list(np.zeros_like(n_s))
        labels.append(label)
        scores.append(s)

