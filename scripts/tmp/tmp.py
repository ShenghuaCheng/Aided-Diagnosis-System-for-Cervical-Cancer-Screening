# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: tmp.py
@Date: 2020/8/26 
@Time: 14:17
@Desc:
'''
import json
import os

if __name__ == '__main__':
    roots = [r'N:\recommend_test\11th_v4_ls-0.1_hard-0.02_nca_ir2_fix_cornified\988.h5\50_slides\BD',
    r'N:\recommend_test\11th_v4_ls-0.1_hard-0.02_nca_ir2_fix_cornified\988.h5\50_slides\Others']
    handle=open(r'N:\recommend_test\11th_v4_ls-0.1_hard-0.02_nca_ir2_fix_cornified\988.h5\50_slides\scores_m2.json','w+')
    info={}
    for root in roots:
        title=os.path.split(root)[-1]
        slides=[line for line in os.listdir(root) if '.txt' not in line and '.xml' not in line]
        info[title]={}
        for slide in slides:
            imgs = os.listdir(os.path.join(root,slide,'model2'))
            info[title][slide]=[]
            for name in imgs:
                rank,w,h,score=name.rstrip('.tif').split('_')
                info[title][slide].append((rank,w,h,score))
    json.dump(info,handle)
    handle.close()