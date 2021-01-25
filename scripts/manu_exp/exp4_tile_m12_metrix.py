# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: exp4_tile_m12_metrix.py
@Date: 2020/12/14 
@Time: 15:15
@Desc: 用于统计m12分别在EF上的表现以及画出临床判读的结果
'''
import os
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

CLINICAL_RESULTS = {
    'pa1': {
        'E': [0.032, 0.938],
        'F': [0.013, 0.824],
        None: [0.023, 0.886],
    },
    'pa2': {
        'E': [0.722, 0.980],
        'F': [0.570, 0.976],
        None: [0.660, 0.978],
    },
    'pa3': {
        'E': [0.047, 0.855],
        'F': [0.025, 0.811],
        None: [0.038, 0.837],
    },
    'pa4': {
        'E': [0.018, 0.753],
        'F': [0.021, 0.730],
        None: [0.019, 0.744],
    },
}

RNN_RESULTS = {
    'm1': {
        'E': r'I:\20200929_EXP4\Tiles\models\model1\E\results.txt',
        'F': r'I:\20200929_EXP4\Tiles\models\model1\F\results.txt'
    },
    'm2': {
        'E': r'I:\20200929_EXP4\Tiles\models\model2\E\results.txt',
        'F': r'I:\20200929_EXP4\Tiles\models\model2\F\results.txt'
    },
    'm2ab': {
        'E': r'I:\20200929_EXP4\Tiles\models\model2AB\E\results.txt',
        'F': r'I:\20200929_EXP4\Tiles\models\model2AB\F\results.txt'
    },

}


def parse_rnn_results(model, use_data=None):
    results_dir = RNN_RESULTS[model]
    results = []
    if use_data is None:
        for k,v in results_dir.items():
            with open(v, 'r') as f:
                results += [l.strip().split(', ') for l in f.readlines()]
    else:
        with open(results_dir[use_data], 'r') as f:
            results += [l.strip().split(', ') for l in f.readlines()]
    return results


def cal_roc(labels, scores):
    roc = {}
    roc_auc = metrics.roc_auc_score(labels, scores)
    roc['fpr'], roc['tpr'], roc['thresholds'] = metrics.roc_curve(labels, scores)
    return roc, roc_auc


def save_metrix(results, excel_wrtr, model, use_data=None, ifplot=None):
    results = np.array(results)
    scores = results[:, 0].astype(float)
    labels = results[:, 1].astype(int)
    img_names = results[:, 2]

    # save raw data
    results_df = {
        "Names": img_names.tolist(),
        "Labels": labels.tolist(),
        "Scores": scores.tolist()
    }
    pd.DataFrame(results_df).to_excel(excel_wrtr, sheet_name='{}{}_raw'.format(model, ('_'+use_data) if use_data is not None else ''), index=False)

    # save roc curve
    roc, roc_auc = cal_roc(labels, scores)
    pd.DataFrame(roc).to_excel(excel_wrtr, sheet_name='{}{}_roc'.format(model, ('_'+use_data) if use_data is not None else ''))  # save roc point

    best_dist = 2.
    best_idx = 0
    for idx, (x, y) in enumerate(zip(roc['fpr'], roc['tpr'])):
        dist = pow((x-0), 2) + pow((y-1), 2)
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    print("{}{} best point: ({:.3f},{:.3f}) {:.3f}".format(
        model, ('_'+use_data) if use_data is not None else '',
        roc['fpr'][best_idx], roc['tpr'][best_idx], roc['thresholds'][best_idx])
    )

    def plot_roc():
        # plot model
        fig = plt.figure()
        lw = 2
        plt.scatter(roc['fpr'][best_idx], roc['tpr'][best_idx], s=30, c='darkred', marker='*')
        plt.text(roc['fpr'][best_idx]+0.005, roc['tpr'][best_idx]-0.025, "({:.3f},{:.3f})".format(roc['fpr'][best_idx], roc['tpr'][best_idx]))

        plt.plot(roc['fpr'], roc['tpr'], color='red',
                 lw=lw, label='ROC curve (AUC = %0.3f)' % roc_auc, zorder=0)

        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')  # split line

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic of groups {}{}'.format(model, ('_'+use_data) if use_data is not None else ''))

        # plot pathologists
        mkr = ['^', 'v', '<', '>']
        for pi, k in enumerate(CLINICAL_RESULTS.keys()):
            plt.scatter(CLINICAL_RESULTS[k][use_data][0], CLINICAL_RESULTS[k][use_data][1], s=30, c='darkorange', marker=mkr[pi], label='Pathologist {} ({:.3f},{:.3f})'.format(pi, CLINICAL_RESULTS[k][use_data][0], CLINICAL_RESULTS[k][use_data][1]))
        plt.legend(loc="lower right")
        return fig

    if ifplot:
        plot_roc().savefig(ifplot)


if __name__ == '__main__':
    save_file = r'I:\20200929_EXP4\Tiles\models\tiles_AB.xlsx'
    writer = pd.ExcelWriter(save_file)
    for m in ['m1', 'm2', 'm2ab']:
        for ud in ['E', 'F', None]:
            save_metrix(parse_rnn_results(m, ud), writer, m, ud, os.path.join(save_file.split('.')[0] + '_{}{}.png'.format(m, ('_'+ud) if ud is not None else '')))
    writer.close()



