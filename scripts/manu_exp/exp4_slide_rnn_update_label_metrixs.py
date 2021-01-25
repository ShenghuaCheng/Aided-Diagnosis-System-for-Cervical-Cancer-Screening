# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: exp4_slide_rnn_update_label_metrixs.py
@Date: 2020/11/24 
@Time: 11:08
@Desc: 该脚本用于按照更新后的标注更新rnn的metrix
'''
import os
import pandas as pd
import json
import sklearn.metrics as metrics

if __name__ == '__main__':
    exclude = json.load(open(r'I:\20200929_EXP4\Slides\exclude.json', 'r'))
    exclude_names = set([exsld['SlideName'] for exsld in exclude])

    pos_check = json.load(open(r'I:\20200929_EXP4\Slides\posCheck_label.json', 'r'))
    pos_check_names = set([pcsld for pcsld in pos_check])

    raw_data = pd.read_excel(r'I:\20200929_EXP4\Slides\RNN_EF_analysis.xlsx', sheetname='raw_data').to_dict('split')
    labels = json.load(open(r'I:\20200929_EXP4\Slides\EF_sld_labels_updateposcheck5.json', 'r'))
    labels = dict(zip(labels['name'], labels['label']))
    dst_dict = {'SlideName': [], 'Labels': [], 'RNNScore': []}
    for sldn, lab, scr in raw_data['data']:
        sldn = os.path.split(sldn)[-1]
        # 排除部分切片来调整
        # if sldn in exclude_names: continue  # ==================================================================

        dst_dict['SlideName'].append(sldn)
        dst_dict['Labels'].append(labels[sldn])
        dst_dict['RNNScore'].append(scr)
    dst_df = pd.DataFrame(dst_dict)
    wrtr = pd.ExcelWriter(r'I:\20200929_EXP4\RNN_EF_poscheck5_1170.xlsx')
    dst_df.to_excel(wrtr, sheet_name='raw_data')

    # cal confusion metrics
    roc = {}
    roc_auc = metrics.roc_auc_score(dst_dict['Labels'], dst_dict['RNNScore'])
    roc['fpr'], roc['tpr'], roc['thresholds'] = metrics.roc_curve(dst_dict['Labels'], dst_dict['RNNScore'])
    pd.DataFrame(roc).to_excel(wrtr, sheet_name='roc')  # save roc point
    best_dist = 2.
    best_idx = 0
    for idx, (x, y) in enumerate(zip(roc['fpr'], roc['tpr'])):
        dist = pow((x-0), 2) + pow((y-1), 2)
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    print("best point: ({:.3f},{:.3f}) {:.3f}".format(roc['fpr'][best_idx], roc['tpr'][best_idx], roc['thresholds'][best_idx]))

    # roc plot
    use_data = 0
    import matplotlib.pyplot as plt
    posCheck1 = [[0.074, 1.], [0.979, 1.]]
    posCheck4 = [[0.042, 0.994], [0.468, 1.]]
    posCheck5 = [[0.041, 0.875], [0.064, 0.919]]

    plt.figure()
    lw = 2
    plt.scatter(roc['fpr'][best_idx], roc['tpr'][best_idx], s=30, c='darkred', marker='^')
    plt.text(roc['fpr'][best_idx]+0.005, roc['tpr'][best_idx]-0.025, "({:.3f},{:.3f})".format(roc['fpr'][best_idx], roc['tpr'][best_idx]))

    plt.plot(roc['fpr'], roc['tpr'], color='red',
             lw=lw, label='ROC curve (AUC = %0.3f)' % roc_auc, zorder=0)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')  # split line

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic of groups E&F')
    # plot pathologists
    plt.scatter(posCheck1[use_data][0], posCheck1[use_data][1], s=30, c='darkorange', marker='*', label='Pathologist 1 ({:.3f},{:.3f})'.format(posCheck1[use_data][0], posCheck1[use_data][1]))
    plt.scatter(posCheck4[use_data][0], posCheck4[use_data][1], s=30, c='darkorange', marker='*', label='Pathologist 2 ({:.3f},{:.3f})'.format(posCheck4[use_data][0], posCheck4[use_data][1]))
    plt.scatter(posCheck5[use_data][0], posCheck5[use_data][1], s=30, c='darkorange', marker='*', label='Pathologist 3 ({:.3f},{:.3f})'.format(posCheck5[use_data][0], posCheck5[use_data][1]))
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(r'I:\20200929_EXP4\RNN_EF_preview.png')
    # # plot pathologists
    # use_data = 1
    # plt.scatter(posCheck1[use_data][0], posCheck1[use_data][1], s=15, c='darkorange', marker='*', label='Pathologist 1 (fpr,tpr)=({},{})'.format(posCheck1[use_data][0], posCheck1[use_data][1]))
    # plt.scatter(posCheck4[use_data][0], posCheck4[use_data][1], s=15, c='darkorange', marker='*', label='Pathologist 2 (fpr,tpr)=({},{})'.format(posCheck4[use_data][0], posCheck4[use_data][1]))
    # plt.scatter(posCheck5[use_data][0], posCheck5[use_data][1], s=15, c='darkorange', marker='*', label='Pathologist 3 (fpr,tpr)=({},{})'.format(posCheck5[use_data][0], posCheck5[use_data][1]))
    # plt.legend(loc="lower right")
    # plt.show()
    # plt.savefig(r'I:\20200929_EXP4\RNN_EF_preview.png')
    wrtr.close()
