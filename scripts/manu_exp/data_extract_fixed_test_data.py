# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: data_extract_fixed_test_data.py
@Date: 2020/12/22 
@Time: 10:32
@Desc:  本脚本用以选定ABCDEF的测试数据，并且预处理成m12输入所需
'''
import os
import numpy as np
import pandas as pd
import random
from utils.manu.aux_func import rd_imgs, IN_SHAPE


SEED=42
random.seed(SEED)
np.random.seed(SEED)

TEST_DATASET_XLSX = r'I:\20201221_Manu_Fig3\all_test_data.xlsx'
TEST_DATASET_ROOT = r'I:\20201221_Manu_Fig3\VALID_TEST_DATASET'


def parse_group(group):
    # 获取group对应的样本列表文件列表
    sheet = pd.read_excel(TEST_DATASET_XLSX, sheetname=group)
    file_list = sheet['txt_name'].tolist()
    sample_num = sheet['sample_num'].tolist()
    label = sheet['label'].tolist()
    return file_list, label, sample_num


def sample_data(file_dir, sample_n):
    lines = open(file_dir, 'r').readlines()
    lines = random.sample(lines, sample_n)
    return lines


if __name__ == '__main__':
    model_type = 'm2'
    SAMPLED_TEST_DATASET_ROOT = r'K:\liusibo\20201221_Manu_Fig3\SAMPLED_TEST_DATASET\{}'.format(model_type)
    os.makedirs(SAMPLED_TEST_DATASET_ROOT, exist_ok=True)

    for gp in ['A', 'B', 'C', 'D', 'E', 'F']:
        file_list, label, sample_num = parse_group(gp)
        for idx, f_n in enumerate(file_list):
            lb = label[idx]
            smp_nb = sample_num[idx]
            samples = sample_data(os.path.join(TEST_DATASET_ROOT, f_n), smp_nb)
            open(os.path.join(SAMPLED_TEST_DATASET_ROOT, f_n), 'w+').writelines(samples)
            samples = [s.strip() for s in samples]
            if lb:
                print('reading {}: {}'.format(gp, f_n))
                data = rd_imgs(samples, IN_SHAPE[model_type])
                np.savez(os.path.join(SAMPLED_TEST_DATASET_ROOT, f_n.replace(".txt", ".npz")), data=data, label=np.array([lb]*len(samples)))
                print("done")



