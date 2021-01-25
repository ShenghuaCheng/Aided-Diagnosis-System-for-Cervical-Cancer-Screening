# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: exp4_exclude_slides_between_0108.py
@Date: 2020/12/11
@Time: 16:54
@Desc: 按照比例剔除rnn排名0.1-0.8之间的切片，提出切片不包括posCheck的121张
'''
import json
import random
import pandas as pd

if __name__ == '__main__':
    save_exclude = r'I:\20200929_EXP4\Slides\exclude.json'
    rnn_results = pd.read_excel(r'I:\20200929_EXP4\RNN_EF.xlsx', sheetname="adjust_hist", usecols=[0,1,2])
    rnn_results = rnn_results.to_dict('records')
    posCheck121 = pd.read_excel(r'I:\20200929_EXP4\RNN_EF.xlsx', sheetname="121posCheck", header=None)
    posCheck121 = set(posCheck121.to_dict('list')[0])
    exclude_slides = []
    for sld in rnn_results:
        if sld['RNNScore'] < 0.1 or sld['RNNScore'] > 0.8:
            continue
        if sld['SlideName'] in posCheck121:
            continue
        if random.random() > 0.8:
            continue
        exclude_slides.append(sld)
        with open(save_exclude, 'w+') as f:
            json.dump(exclude_slides, f, indent=2)
