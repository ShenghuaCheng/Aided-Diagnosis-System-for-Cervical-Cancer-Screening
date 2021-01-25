# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: exp4_update_EFslidelabel.py
@Date: 2020/11/20 
@Time: 16:35
@Desc: 根据poscheck45的结果来更新EF的标注
'''
import json
import pandas as pd
if __name__ == '__main__':
    src_labels_dir = r'I:\20200929_EXP4\Slides\EF_sld_labels.json'
    dst_labels_dir = r'I:\20200929_EXP4\Slides\EF_sld_labels_updateposcheck5.json'
    update_dict = r'I:\20200929_EXP4\posCheck145_record.xlsx'  # sheet name is newlabel
    # get new labels
    update_dict = pd.read_excel(update_dict, sheetname='newlabel')
    update_dict = update_dict.to_dict()
    labels = json.load(open(src_labels_dir, 'r'))
    labels = dict(zip(labels['name'], labels['label']))
    for k, v in update_dict['new label'].items():
        labels[k] = v
    dst_label = {'name': [], 'label': []}
    for k, v in labels.items():
        dst_label['name'].append(k)
        dst_label['label'].append(int(v))
    json.dump(dst_label, open(dst_labels_dir, 'w+'), indent=2)

