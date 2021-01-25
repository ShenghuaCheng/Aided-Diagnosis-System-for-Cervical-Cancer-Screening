# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: plot_roi.py
@Date: 2020/3/3 
@Time: 15:10
@Desc:
'''
import os
from glob2 import glob
import numpy as np
from utils.auxfunc.plot import plot_roc, plot_roc_muti

if __name__ == '__main__':
    # # single
    # npy_fld = r'F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\manu_yjy\exp1\results\roc'
    # model_type = 'origin_flip'
    # # parse npy to get score ande label
    # p_s = []
    # n_s = []
    # for d in glob(os.path.join(npy_fld, '%s*_p.npy' % model_type)):
    #     p_s += list(np.load(d).ravel())
    # for d in glob(os.path.join(npy_fld, '%s*_n.npy' % model_type)):
    #     n_s += list(np.load(d).ravel())
    # s = p_s + n_s
    # label = list(np.ones_like(p_s)) + list(np.zeros_like(n_s))
    # # plot
    # src = model_type
    # file_root = npy_fld
    # plot_roc(src, file_root, np.array(label), np.array(s))

    # multi
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
        s = p_s + n_s
        label = list(np.ones_like(p_s)) + list(np.zeros_like(n_s))
        labels.append(label)
        scores.append(s)
    # plot
    file_root = npy_fld
    names = model_types
    plot_roc_muti(src, file_root, labels, scores, names)

