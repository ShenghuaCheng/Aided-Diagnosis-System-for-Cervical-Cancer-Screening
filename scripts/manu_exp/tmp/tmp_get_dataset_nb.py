# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: tmp_get_dataset_nb.py
@Date: 2020/3/12 
@Time: 15:17
@Desc:
'''
import os
import pandas as pd
if __name__ == '__main__':
    root = r'F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\manu_yjy\dataset\Itest'
    wt = pd.ExcelWriter(root+".xlsx")
    files = [n for n in os.listdir(root) if '.txt' in n]
    summary = {}
    for fn in files:
        with open(os.path.join(root, fn), 'r') as f:
            img_list = f.readlines()
        nb = len(img_list)
        summary[fn.rstrip(".txt")] = nb
    sm_df = pd.DataFrame.from_dict(summary, orient="index")
    sm_df.to_excel(wt, header=None)
    wt.close()