# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: tmp_ex_wrongimg.py
@Date: 2020/3/8 
@Time: 21:31
@Desc:
'''
import os
import shutil
import cv2

if __name__ == '__main__':
    root_list = [
        r"H:\AdaptDATA\train\szsq\sdpc_xyw1\origin\n",
        r"H:\AdaptDATA\train\szsq\sdpc_xyw1\gamma_simple\n",
        r"H:\AdaptDATA\train\szsq\sdpc_xyw2\origin\n",
        r"H:\AdaptDATA\train\szsq\sdpc_xyw2\gamma_simple\n",

        r"H:\AdaptDATA\test\szsq\sdpc_xyw1\origin\n",
        r"H:\AdaptDATA\test\szsq\sdpc_xyw1\gamma_simple\n",
        r"H:\AdaptDATA\test\szsq\sdpc_xyw2\origin\n",
        r"H:\AdaptDATA\test\szsq\sdpc_xyw2\gamma_simple\n",
        ]

    sub_list = ["n", "n_w", "n_0", "n_1", "n_2", "n_5", "n_8", "n_9"]

    for root in root_list[7:8]:
        for sub in sub_list:
            fld_dir = os.path.join(root, sub)
            img_names = [n for n in os.listdir(fld_dir) if '.tif' in n]
            nb_imgs = len(img_names)
            for i, name in enumerate(img_names):
                print("%s %d/%d" % (fld_dir, i+1, nb_imgs))
                if cv2.imread(os.path.join(fld_dir, name)) is None:
                    dst_fld = os.path.join(fld_dir, 'wrong')
                    os.makedirs(dst_fld, exist_ok=True)
                    print("move %s to wrong" % name)
                    shutil.move(os.path.join(fld_dir, name), os.path.join(dst_fld, name))
