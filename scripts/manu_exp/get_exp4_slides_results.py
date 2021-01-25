# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: get_exp4_slides_results.py.py
@Date: 2020/10/27 
@Time: 10:55
@Desc: 根据hw提供的tbs.txt文件，获取每个账号对于全切片的判读情况，刻画混淆矩阵，并给出判读的对比
'''
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

label_name = {'0110': 0, '1111': 0, '2130': 1, '2111': 1}

if __name__ == '__main__':
    txt_dir = r'I:\20200929_EXP4\Slides\tbs_slides_checked_tag.txt'
    labels_dir = r'I:\20200929_EXP4\Slides\EF_sld_labels_updateposcheck5.json'
    xlsx_dir = r'I:\20200929_EXP4\Slides\posCheck145_labelPosCheck5_1170.xlsx'

    labels = json.load(open(labels_dir, 'r'))
    labels = dict(zip(labels['name'], labels['label']))

    with open(txt_dir, 'r') as f:
         txt = [l.strip().split('|') for l in f.readlines()]

    checked = {i[1]: [] for i in txt}
    checked_df = {i[1]: {} for i in txt}
    for i in txt:
        checked[i[1]].append([i[0], labels[i[0]], label_name[i[2]]])
        checked_df[i[1]][i[0]] = label_name[i[2]]
    writer = pd.ExcelWriter(xlsx_dir)
    # 这个脚本只看了posCheck1 4 5账号的全切片结果
    for usr in ['posCheck1','posCheck4','posCheck5']:
        results = np.array(checked[usr])
        results_df = {}
        keys = [items.replace('.xml', '.sdpc') for items in os.listdir(r'I:\20200929_EXP4\Slides\local_xml_split_all145_pos145\posCheck5') if '.xml' in items]
        for k in keys:
            try:
                results_df[k] = {'label': labels[k], 'check': checked_df[usr][k]}
            except KeyError:
                results_df[k] = {'label': labels[k], 'check': 2}  # 2 for uncheck
        print(confusion_matrix(results[:, 1], results[:, 2]))
        results_df = pd.DataFrame(results_df).T
        results_df.to_excel(writer, sheet_name=usr)
    writer.close()