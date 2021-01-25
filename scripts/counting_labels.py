# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: counting_labels.py
@Date: 2020/1/16 
@Time: 19:41
@Desc:
'''
import os
import pandas as pd
from utils.auxfunc.data_summary import SubSet

if __name__ == '__main__':
    cnt_save = r'F:\LiuSibo\Exps\200108_paper\data_summary\F_xyw2.xlsx'
    sld_nb_cnt = r'F:\LiuSibo\Exps\200108_paper\data_summary\sld_num.txt'

    slide_fld = r'H:\TCTDATA\SZSQ_originaldata\Xiaoyuwei2\positive'
    label_fld = r'H:\TCTDATA\SZSQ_originaldata\Labelfiles\xml_xyw2'
    label_mode = 2  # 1 是csv格式；2 是xml格式

    subset = SubSet(slide_fld, label_fld, label_mode)
    with open(sld_nb_cnt, 'a') as f:
        f.write('%s, %d\n' % (slide_fld, len(subset.name_list)))
    cnt_df = subset.counter
    cnt_df = pd.DataFrame(cnt_df)
    writer = pd.ExcelWriter(cnt_save)
    cnt_df.to_excel(writer, index=False)
    writer.close()