# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: tmp_ensure_test_file_eixted.py
@Date: 2020/12/22 
@Time: 15:26
@Desc: 该脚本用于确定TEST_DATASET中的数据均存在，同时在H盘寻找原来置于I盘的数据，并记录每个文件中显示的数据数量
'''
import os
from functools import partial
import pandas as pd


TEST_DATASET_ROOT=r'I:\20201221_Manu_Fig3\TEST_DATASET'
VALID_DATASET_ROOT=r'I:\20201221_Manu_Fig3\VALID_TEST_DATASET'
INVALID_DATASET_ROOT=r'I:\20201221_Manu_Fig3\INVALID_TEST_DATASET'
list(map(partial(os.makedirs, exist_ok=True), [VALID_DATASET_ROOT, INVALID_DATASET_ROOT]))


def parse_txt(txt_file):
    """ 解析存放数据样本路径的txt文件，返回有效和无效的路径 """
    f = open(txt_file, 'r')
    valid_lines = []
    invalid_line = []
    for l in f.readlines():
        l = l.strip()
        if os.path.exists(l):
            valid_lines.append(l)
        elif os.path.exists("H" + l[1:]):
            valid_lines.append("H" + l[1:])
        else:
            invalid_line.append(l)
    f.close()
    return valid_lines, invalid_line


def save_txt(dirs_ls, dst_txt):
    f = open(dst_txt, 'w+')
    for d in dirs_ls:
        f.write('{}\n'. format(d))
    f.close()
    print("write {} lines to {}.".format(len(dirs_ls), dst_txt))


if __name__ == '__main__':
    recorder = pd.ExcelWriter(r'I:\20201221_Manu_Fig3\test_dataset_sum.xlsx')
    txt_list = [f for f in os.listdir(TEST_DATASET_ROOT) if '.txt' in f]
    valid_set = {}
    invalid_set = {}
    for txt_n in txt_list:
        val, inval = parse_txt(os.path.join(TEST_DATASET_ROOT, txt_n))
        valid_set[txt_n] = len(val)
        invalid_set[txt_n] = len(inval)
        save_txt(val, os.path.join(VALID_DATASET_ROOT, txt_n))
        save_txt(inval, os.path.join(INVALID_DATASET_ROOT, txt_n))
    valid_set = pd.DataFrame(list(valid_set.items()), columns=['txt_name', 'num'])
    invalid_set = pd.DataFrame(list(invalid_set.items()), columns=['txt_name', 'num'])
    valid_set.to_excel(recorder, sheet_name="valid sum")
    invalid_set.to_excel(recorder, sheet_name="invalid sum")
    recorder.close()
