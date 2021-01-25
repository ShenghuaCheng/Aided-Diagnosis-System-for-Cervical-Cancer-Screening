# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: xlsx_roc.py
@Date: 2020/1/4 
@Time: 9:54
@Desc:
'''
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def plot_roc(src, file_root, labels, scores):
    fpr, tpr, thre = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6.18))
    lw = 2  # 线宽
    font = {'family': 'serif',
            'style': 'normal',
            'weight': 'bold',
            'size': 15
            }
    t_tmp = max(thre)
    for i, t in enumerate(thre):
        if t < (t_tmp - 0.1):
            t_tmp = t
            plt.text(fpr[i], tpr[i], '%.2f' % t, color='b', alpha=0.6, verticalalignment='bottom', fontdict=font)

    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (AUC = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim([0.0, 1.0], )
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.title('Receiver operating characteristic %s' % src, fontsize=20)
    plt.legend(loc="lower right", fontsize=20)
    plt.savefig(os.path.join(file_root, '%s.png' % src))
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file_root")
    args = parser.parse_args()
    file_root = args.file_root
    all_label = []
    all_score = []
    src_list = [s for s in os.listdir(file_root) if '.xlsx' in s]
    for src in src_list:
        src = src.split('.')[0]
        file_xlsx = os.path.join(file_root, '%s.xlsx' % src)
        single_df = pd.read_excel(file_xlsx)
        single_np = np.array(single_df)
        labels, scores = single_np[:, 0].astype(int), single_np[:, 2].astype(float)
        all_label += list(labels)
        all_score += list(scores)
        plot_roc(src, file_root, labels, scores)
    plot_roc('all', file_root, np.array(all_label), np.array(all_score))
