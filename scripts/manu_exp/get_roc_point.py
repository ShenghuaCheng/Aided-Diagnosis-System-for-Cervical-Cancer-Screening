# -*- coding:utf-8 -*-
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

    npy_fld = r'\model2'
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

