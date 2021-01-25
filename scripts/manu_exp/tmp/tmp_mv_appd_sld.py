# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: tmp_mv_appd_sld.py
@Date: 2020/12/26 
@Time: 12:16
@Desc: 将挑选的60张切片文件移动到新的文件夹下
'''
import os
from glob2 import glob
import shutil
import pandas as pd

if __name__ == '__main__':
    ind_file = r'I:\20201225_Manu_AppdSlide\Appd_list.xlsx'
    for shn in ['3D_TJ', 'BD_SZL', 'BD_TJ', 'BD_XH', 'SZSQ_SFY']:
        sheet = pd.read_excel(ind_file, sheetname=shn).to_dict('list')
        os.makedirs(r'K:\liusibo\20201225_Manu_AppdSlide\Slides\{}'.format(shn), exist_ok=True)
        for d in sheet['dir']:
            if os.path.exists(os.path.join(r'K:\liusibo\20201225_Manu_AppdSlide\Slides\{}'.format(shn), os.path.split(d)[-1])):
                continue
            shutil.copy(d, os.path.join(r'K:\liusibo\20201225_Manu_AppdSlide\Slides\{}'.format(shn), os.path.split(d)[-1]))



