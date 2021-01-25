# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: get_img_list_txt.py
@Date: 2020/7/21 
@Time: 10:07
@Desc:
'''
import os
import random
import glob2

if __name__ == '__main__':
    nb = 5000
    bat_name = 'sdpc_sfy8'
    bat_type = 'n\\n_9'
    img_fld = r'H:\AdaptDATA\train\szsq\%s\gamma_simple\%s' % (bat_name, bat_type)
    txt_root = r'F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\manu_yjy\exp2\config\Itest'
    txt_name = 'E_szsq_%s_gamma_simple_%s.txt' % (bat_name, bat_type.replace('\\', '_'))
    img_list = glob2.glob(os.path.join(img_fld, '*.tif'))
    img_list = random.sample(img_list, min(len(img_list), nb))
    with open(os.path.join(txt_root, txt_name), 'a') as f:
        for img_d in img_list:
            f.write(img_d+'\n')


