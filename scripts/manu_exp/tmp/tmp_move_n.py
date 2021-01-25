# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: tmp_move_n.py
@Date: 2020/3/15 
@Time: 13:50
@Desc:
'''
import os
import functools
import shutil
import random
from multiprocessing.dummy import Pool


def copy_img(src_fld, dst_fld, img_name):
    shutil.copy(os.path.join(src_fld, img_name), os.path.join(dst_fld, img_name))


if __name__ == '__main__':
    src_roots = [
        r'H:\AdaptDATA\test\3d\sfy1',
        r'H:\AdaptDATA\test\3d\sfy2',
        r'H:\AdaptDATA\test\szsq\sdpc_sfy1\gamma_simple',
        r'H:\AdaptDATA\test\szsq\sdpc_sfy3\gamma_simple',
        r'H:\AdaptDATA\test\szsq\sdpc_tongji3\gamma_simple',
        r'H:\AdaptDATA\test\szsq\sdpc_tongji4\gamma_simple',
        r'H:\AdaptDATA\test\szsq\sdpc_tongji5\gamma_simple',
        r'H:\AdaptDATA\test\szsq\sdpc_tongji6\gamma_simple',
        r'H:\AdaptDATA\test\szsq\sdpc_tongji7\gamma_simple',
        r'H:\AdaptDATA\test\szsq\sdpc_xyw1\gamma_simple',
        r'H:\AdaptDATA\test\szsq\sdpc_xyw2\gamma_simple',
    ]
    sub_flds = ['nplus', 'n/n', 'n/n_w', 'n/n_0', 'n/n_1', 'n/n_2', 'n/n_5', 'n/n_8', 'n/n_9']
    for src_root in src_roots:
        for sub_fld in sub_flds:
            src_fld = os.path.join(src_root, sub_fld)
            if not os.path.exists(src_fld):
                continue
            dst_fld = src_fld.replace(r"H:", r"I:")
            os.makedirs(dst_fld, exist_ok=True)
            img_list = os.listdir(src_fld)
            if len(img_list) > 5000:
                img_list = random.sample(img_list, 5000)
            print('%s to %s, nb: %d' % (src_fld, dst_fld, len(img_list)))
            # src_img = [os.path.join(src_fld, n) for n in img_list]
            # dst_img = [os.path.join(dst_fld, n) for n in img_list]
            cp_func = functools.partial(copy_img, src_fld, dst_fld)
            pool = Pool(16)
            pool.map(cp_func, img_list)
            pool.close()
            pool.join()


