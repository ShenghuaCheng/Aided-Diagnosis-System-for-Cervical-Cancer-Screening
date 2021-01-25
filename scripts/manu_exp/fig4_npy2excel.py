# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: fig4_npy2excel.py
@Date: 2020/12/31 
@Time: 11:27
@Desc: 将降维之后的点npy格式转存为excel
'''
import os
import pandas as pd
import numpy as np
from utils.manu.aux_func import MODEL_NAMES, GROUPS

if __name__ == '__main__':
    save_root = r'I:\20201218_Manu_Fig4\t-SNE_12'
    data_root = r'I:\20201218_Manu_Fig4\t-SNE_12'

    for m_n in MODEL_NAMES['m2']:
        save_dir = os.path.join(save_root, m_n + '.xlsx')
        wrt = pd.ExcelWriter(save_dir)

        f = np.load(os.path.join(data_root, m_n, 'feature_data.npz'))
        f_em = np.load(os.path.join(data_root, m_n, 'features_embedded.npy'))

        for idx, grp in enumerate(GROUPS):
            res = {}
            res['dirs'] = f['dirs'][idx*2000: (idx+1)*2000]
            res['labels'] = f['labels'][idx*2000: (idx+1)*2000] % 10
            res['scores'] = f['scores'].ravel()[idx*2000: (idx+1)*2000]
            res['embd_x'] = f_em[idx*2000: (idx+1)*2000, 0]
            res['embd_y'] = f_em[idx*2000: (idx+1)*2000, 1]
            res_df = pd.DataFrame(res)
            res_df.to_excel(wrt, sheet_name=grp)
        wrt.close()