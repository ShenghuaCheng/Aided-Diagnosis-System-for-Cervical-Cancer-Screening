# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: fig4_choose_tsne_data_Appd.py
@Date: 2020/12/29 
@Time: 16:20
@Desc: 从增加的6批共60张切片的运算结果当中分别抽取1000张阳性，1000张阴性tile
'''
import os
from glob2 import glob
import random

TEST_DICT = {
    'G': r'H:\liusibo\manuscripts_top_tiles_for_rnn\origin\Appd\3D_TJ',
    'H': r'H:\liusibo\manuscripts_top_tiles_for_rnn\origin\Appd\BD_SZL',
    'I': r'H:\liusibo\manuscripts_top_tiles_for_rnn\origin\Appd\BD_TJ',
    'J': r'H:\liusibo\manuscripts_top_tiles_for_rnn\origin\Appd\BD_XH',
    'K': r'H:\liusibo\manuscripts_top_tiles_for_rnn\origin\Appd\JY',
    'L': r'H:\liusibo\manuscripts_top_tiles_for_rnn\origin\Appd\SZSQ_SFY',
}

if __name__ == '__main__':
    save_root = r'I:\20201218_Manu_Fig4\GHIJKL_samples_most'
    os.makedirs(save_root, exist_ok=True)
    for grp, grp_dir in TEST_DICT.items():
        for cls in ['P', 'N']:
            all_slds = [item for item in os.walk(os.path.join(grp_dir, cls)) if len(item[-1])]
            all_imgs = []
            for sld in all_slds:
                all_imgs += glob(os.path.join(sld[0], '*.png'))

            thre_p=0.0
            thre_n=1.0
            for dec in range(100):
                if cls == 'P':
                    tmp = [item for item in all_imgs if float(os.path.splitext(os.path.split(item)[-1])[0].split('_')[-1]) > thre_p]
                    if len(tmp) > 1000: thre_p += 0.01
                    else:
                        thre_p -= 0.01
                        all_imgs = [item for item in all_imgs if
                               float(os.path.splitext(os.path.split(item)[-1])[0].split('_')[-1]) > thre_p]
                        break

                elif cls == 'N':
                    tmp = [item for item in all_imgs if float(os.path.splitext(os.path.split(item)[-1])[0].split('_')[-1]) < thre_n]
                    if len(tmp) > 1000: thre_n -= 0.01
                    else:
                        thre_n += 0.01
                        all_imgs = [item for item in all_imgs if
                               float(os.path.splitext(os.path.split(item)[-1])[0].split('_')[-1]) < thre_n]
                        break

            print("{} {} total: {}".format(grp, cls, len(all_imgs)))
            open(os.path.join(save_root, '{}_{}.txt'.format(grp, cls)), 'w+').writelines([l+'\n' for l in random.sample(all_imgs, 1000)])



