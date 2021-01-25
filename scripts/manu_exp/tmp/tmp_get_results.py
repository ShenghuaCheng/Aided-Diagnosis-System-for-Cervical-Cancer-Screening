# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: tmp_get_results.py
@Date: 2020/3/13 
@Time: 16:42
@Desc:
'''
import os
import pandas as pd

if __name__ == '__main__':
    result_root = r'F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\manu_yjy\exp2\yjy_train_hist\3_stage_ABCD\test_model'
    txt_list = [f for f in os.listdir(result_root) if '.txt' in f]
    result = {}
    for file in txt_list:
        file_dir = os.path.join(result_root, file)
        with open(file_dir, 'r') as f:
            scores = f.readlines()
        scores = {i: float(s.strip('\n')) for i, s in enumerate(scores)}
        result[file.strip('.txt')] = scores
    results_df = pd.DataFrame.from_dict(result)
    wt = pd.ExcelWriter(os.path.join(result_root, 'result.xlsx'))
    results_df.to_excel(wt)
    wt.close()
