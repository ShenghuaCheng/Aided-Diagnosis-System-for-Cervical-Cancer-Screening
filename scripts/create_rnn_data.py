# -*- coding:utf-8 -*-
import os
import time
from utils.grading.dataset import WSIDataSet

if __name__ == '__main__':
    read_in_dicts = {
        '': [
            r'A\Positive',
            r'A\Negative',
            r'B\Positive',
            r'B\Negative',
        ],
        'C': [
        ]
        # ......
    }
    slds_root = r'root to slide'
    preds_root = r"root to predict results"
    save_root = r"root to cropped images"
    save_gamma = r'root to cropped images with gamma crct'

    wsi_data = WSIDataSet(slds_root, preds_root, read_in_dicts)
    # main loop
    since = time.time()
    for i, wsi_item in enumerate(wsi_data):
        print('[%d/%d]' % (i, len(wsi_data)))
        # skip already exist file
        original_save = os.path.join(save_root, wsi_item[1].slide_dir.lstrip(slds_root))
        gamma_save = os.path.join(save_gamma, wsi_item[1].slide_dir.lstrip(slds_root))
        if os.path.exists(original_save) and os.path.exists(gamma_save):
            print('already exist %s' % wsi_item[1].slide_dir)
            continue
        # crop and save
        wsi_item[1].crop_imgs_data(100, 100, 0, 384, slds_root, save_root, save_gamma)
        print('avg time: %.3f/slide' % ((time.time()-since)/(i+1)))
