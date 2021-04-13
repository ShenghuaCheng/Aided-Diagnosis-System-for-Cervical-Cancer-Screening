# -*- coding:utf-8 -*-
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def plot_roc(src, file_root, labels, scores):
    """
    :param src:
    :param file_root:
    :param labels:
    :param scores:
    :return:
    """
    fpr, tpr, thre = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6.18))
    lw = 2  # 线宽
    # font = {'family': 'serif',
    #         'style': 'normal',
    #         'weight': 'bold',
    #         'size': 15
    #         }
    # t_tmp = max(thre)
    # for i, t in enumerate(thre):
    #     if t < (t_tmp - 0.1):
    #         t_tmp = t
    #         plt.text(fpr[i], tpr[i], '%.2f' % t, color='b', alpha=0.6, verticalalignment='bottom', fontdict=font)

    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (AUC = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim([0.0, 1.0], )
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate (1-Specificity)', fontsize=20)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=20)
    plt.title('Receiver operating characteristic %s' % src, fontsize=20)
    plt.legend(loc="lower right", fontsize=20)
    plt.savefig(os.path.join(file_root, '%s.png' % src))
    plt.close()


def plot_roc_muti(src, file_root, labels, scores, names):
    """类别不大于5，否则颜色不够用
    :param src:
    :param file_root:
    :param labels:
    :param scores:
    :return:
    """
    COLOR = {0:'r', 1:'g', 2:'b', 3:'orange', 4:'navy'}
    plt.figure(figsize=(10, 6.18))
    lw = 2  # 线宽
    font = {'family': 'serif',
            'style': 'normal',
            'weight': 'bold',
            'size': 15
            }
    for idx, name in enumerate(names):
        fpr, tpr, thre = roc_curve(labels[idx], scores[idx])
        roc_auc = auc(fpr, tpr)

        # t_tmp = max(thre)
        # for i, t in enumerate(thre):
        #     if t < (t_tmp - 0.1):
        #         t_tmp = t
        #         plt.text(fpr[i], tpr[i], '%.2f' % t, color='b', alpha=0.6, verticalalignment='bottom', fontdict=font)

        plt.plot(fpr, tpr, color=COLOR[idx],
                 lw=lw, label='%s (AUC = %0.4f)' % (name, roc_auc))
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


def plot_PRcurve(src, file_root, labels, scores, names):
    COLOR = {0:'r', 1:'g', 2:'b', 3:'orange', 4:'navy'}
    plt.figure(figsize=(10, 6.18))
    lw = 2  # 线宽
    for idx, name in enumerate(names):
        precision, recall, thre = precision_recall_curve(labels[idx], scores[idx])
        plt.plot(recall, precision, color=COLOR[idx],
                 lw=lw, label='%s' % name)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim([0.0, 1.0], )
    plt.ylim([0.0, 1.05])

    plt.xlabel('Recall', fontsize=20)
    plt.ylabel('Precision', fontsize=20)
    plt.title('PR Curve of %s' % src, fontsize=20)
    plt.legend(loc="lower right", fontsize=20)
    plt.savefig(os.path.join(file_root, '%s.png' % src))
    plt.close()
