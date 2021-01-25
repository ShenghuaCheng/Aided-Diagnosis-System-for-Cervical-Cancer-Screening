# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: analyse_top_scores.py
@Date: 2019/12/23 
@Time: 11:23
@Desc: 分析数据集切片top分数的分布情况，寻找规则方案
'''
import numpy as np
from utils.grading.dataset import WSIDataSet
from utils.grading.visualizingfunc import draw_distribution, draw_hist

if __name__ == '__main__':
    read_in_dicts = {
        '3D': [
            # r'Shengfuyou_3th',
        ],
        'our': [
            # r'Shengfuyou_3th',
        ],
        'SZSQ_originaldata': [
            # r'Shengfuyou_3th\positive\Shengfuyou_3th_positive_40X',
        ],
        'SrpData': [
            r'out\xyw1\positive',
            r'out\xyw1\negative',
            r'out\xyw2\positive',
            r'out\xyw2\negative',
            r'out\xyw3\positive',
            r'out\xyw3\negative',

            r'out_csh\xyw1\positive',
            r'out_csh\xyw1\negative',
            r'out_csh\xyw2\positive',
            r'out_csh\xyw2\negative',
            r'out_csh\xyw3\positive',
            r'out_csh\xyw3\negative',

            r'out2\xyw1\positive',
            r'out2\xyw1\negative',
            r'out2\xyw2\positive',
            r'out2\xyw2\negative',
            r'out2\xyw3\positive',
            r'out2\xyw3\negative',

            r'out2_bak\xyw1\positive',
            r'out2_bak\xyw1\negative',
            r'out2_bak\xyw2\positive',
            r'out2_bak\xyw2\negative',
            r'out2_bak\xyw3\positive',
            r'out2_bak\xyw3\negative',

            r'out2_bak2\xyw1\positive',
            r'out2_bak2\xyw1\negative',
            r'out2_bak2\xyw2\positive',
            r'out2_bak2\xyw2\negative',
            r'out2_bak2\xyw3\positive',
            r'out2_bak2\xyw3\negative',

            r'out-2019-11-17\xyw1\positive',
            r'out-2019-11-17\xyw1\negative',
            r'out-2019-11-17\xyw2\positive',
            r'out-2019-11-17\xyw2\negative',
            r'out-2019-11-17\xyw3\positive',
            r'out-2019-11-17\xyw3\negative',

            r'out-2019-11-19\xyw1\positive',
            r'out-2019-11-19\xyw1\negative',
            r'out-2019-11-19\xyw2\positive',
            r'out-2019-11-19\xyw2\negative',
            r'out-2019-11-19\xyw3\positive',
            r'out-2019-11-19\xyw3\negative',
        ],
    }
    slds_root = r'H:\TCTDATA'
    preds_root = r'H:\fql\rnnResult\rnn1000'

    analysis_save = r'F:\LiuSibo\Exps\191223_rnn_explore\rulebase'
    topk = 5

    wsi_data = WSIDataSet(slds_root, preds_root, read_in_dicts)

    labels, tops = wsi_data.analyse_top_scores(topk)
    labels = np.array(labels)
    tops = np.stack(tops)
    means = np.mean(tops, axis=1)
    std = np.std(tops, axis=1)

    # draw_hist(means, labels, save_root=analysis_save)
    # draw_distribution(means, labels, save_root=analysis_save)
    # draw_hist(means[means<0.85], labels[means<0.85])

    draw_hist(std, labels, save_root=analysis_save)
    draw_distribution(std, labels, save_root=analysis_save)
    # draw_hist(std[std<0.85], labels[std<0.85])
