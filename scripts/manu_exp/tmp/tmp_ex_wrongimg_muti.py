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
import json
import shutil
import cv2
from multiprocessing.dummy import Pool


def check_img(img_dir):
    try:
        img = cv2.imread(img_dir)
        if img is None:
            print(img_dir)
            return [img_dir, 1]
        return [img_dir, 0]
    except:
        print(img_dir)
        return [img_dir, 1]


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

    for root in root_list[1:2]:
        for sub in sub_list:
            fld_dir = os.path.join(root, sub)
            print(fld_dir)
            img_names = [n for n in os.listdir(fld_dir) if '.tif' in n]
            img_dirs = [os.path.join(fld_dir, n) for n in img_names]

            nb_imgs = len(img_names)

            pool = Pool(16)
            check_list = pool.map(check_img, img_dirs)
            pool.close()
            pool.join()
            with open(os.path.join(fld_dir, 'check_list.json'), 'a') as f:
                json.dump(check_list, f)

            for i, item in enumerate(check_list):
                name = img_names[i]
                os.path.join(fld_dir, name)
                if item[1]:
                    dst_fld = os.path.join(fld_dir, 'wrong')
                    os.makedirs(dst_fld, exist_ok=True)
                    print("error, move %s to wrong" % name)
                    shutil.move(os.path.join(fld_dir, name), os.path.join(dst_fld, name))
                else:
                    print("ok, %s %d/%d" % (fld_dir, i + 1, nb_imgs))
