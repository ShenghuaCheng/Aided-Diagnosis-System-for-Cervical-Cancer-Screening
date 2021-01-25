# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: data_sum_labels.py
@Date: 2020/12/17 
@Time: 9:32
@Desc:  用于统计实验进行时所有的标注数量，根据不同的group划分，统计依据是F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\manu_yjy\dataset
'''
import os
LABEL_ROOTS = {
    'train': r'F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\manu_yjy\dataset\train',
    'test': r'F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\manu_yjy\dataset\test',
    'vail': r'F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\manu_yjy\dataset\vail',
    'Itest': r'F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\manu_yjy\dataset\Itest',
}


# Group = A B C D E F
def count_labels(group, dataset):
    files = [f for f in os.listdir(LABEL_ROOTS[dataset]) if group+'_' in f and '_n' not in f]
    cnts = 0
    for file in files:
        cnts += len(open(os.path.join(LABEL_ROOTS[dataset], file)).readlines())
    print('group {} {} num: {}'.format(group, dataset, cnts))
    return cnts


if __name__ == '__main__':
    for group in ['A', 'B', 'C', 'D', 'E', 'F']:
        for dataset in ['train', 'vail', 'test', 'Itest']:
            count_labels(group, dataset)
