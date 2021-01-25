# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: tmp_trans_txt.py
@Date: 2020/2/28 
@Time: 23:20
@Desc:
'''
import os

if __name__ == '__main__':
    fld = r'F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\manu_yjy\exp1\config\test'
    src = 'A_vail_n_preds.txt'
    dst = 'A_vail_n_hard.txt'

    with open(os.path.join(fld, src), 'r') as f:
        contains = f.readlines()
    with open(os.path.join(fld, dst), 'a') as f:
        for item in contains:
            f.write(item[8:])


