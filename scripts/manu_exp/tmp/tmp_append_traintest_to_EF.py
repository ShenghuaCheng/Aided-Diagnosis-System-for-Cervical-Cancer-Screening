# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: tmp_append_xyw12_to_F.py
@Date: 2020/12/22 
@Time: 16:17
@Desc:  本脚本将szsq sfy5678 xyw12中的训练集阳性样本加入EF
'''
import os
import random

if __name__ == '__main__':
    root = r"I:\20201221_Manu_Fig3\TEST_DATASET"
    txt_list = [f for f in os.listdir(root) if "E_" in f or "F_" in f]

    for txt_n in txt_list:
        if '_sfy1_' in txt_n or '_sfy3_' in txt_n:
            continue
        f = open(os.path.join(root, txt_n))
        legacy = f.readlines()
        f.close()
        if len(legacy) >= 5000:
            print("{}: greater than 5000".format(txt_n))
            continue  # if contains greater than 5000, there is no need for appending
        fld = os.path.split(legacy[0].strip())[0]
        # check the fold existed in I: or H:
        if os.path.exists(fld):
            fld = fld
        elif os.path.exists("H" + fld[1:]):
            fld = "H" + fld[1:]
        else:
            raise ValueError("{} is not exist".format(fld))
        # appending from test or train
        if 'train' in fld:
            fld = fld.replace(r'\train', r'\test')
        elif 'test' in fld:
            fld = fld.replace(r'\test', r'\train')
        if not os.path.exists(fld):
            print("{}: no relative train or test fld {}".format(txt_n, fld))
            continue
        appd_lines = [os.path.join(fld, l+'\n') for l in os.listdir(fld) if '.tif' in l or '.jpg' in l or '.png' in l]
        appd_lines = appd_lines if len(appd_lines) <= (5000 - len(legacy)) else random.sample(appd_lines, 5000-len(legacy))

        new = legacy+appd_lines
        f = open(os.path.join(root, txt_n), 'w+')
        f.writelines(new)
        f.close()


