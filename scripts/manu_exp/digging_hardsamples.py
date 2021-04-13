# -*- coding:utf-8 -*-
import os
from functools import partial
from multiprocessing.dummy import Pool
import numpy as np
from utils.networks.wsi_related import resnet_clf
from utils.auxfunc.readin_val import readin_val

if __name__ == '__main__':
    label = 1
    # get data list
    config_root = r'\train'
    config_file = ''
    post_fix = '.txt'
    with open(os.path.join(config_root, config_file + post_fix), 'r') as f:
        data_list = f.readlines()
    data_list = [it.strip() for it in data_list if it]
    # get save dir
    origin_save = os.path.join(config_root, config_file + '_preds_hard' + post_fix)
    # model setting
    os.environ["CUDA_VISIBLE_DEVICES"] = '4'
    weight_file = r'.h5'
    model = resnet_clf((256, 256, 3))
    model.load_weights(weight_file)
    # read in test img
    batchsize = 500
    re_size = (1843, 1843)
    crop_size = (256, 256)
    readinfunc = partial(readin_val, re_size=re_size, crop_size=crop_size)
    ind = 0
    while ind*batchsize < len(data_list):
        tmp_list = data_list[ind*batchsize: (ind+1)*batchsize]
        pool = Pool(16)
        img_batch = pool.map(readinfunc, tmp_list)
        pool.close()
        pool.join()
        preds = model.predict(np.stack(img_batch))
        preds.ravel()
        dist = np.abs(preds-label)
        wrong_ind = np.where(dist > 0.5)[0]
        print('finish %d/%d' % ((ind+1)*batchsize, len(data_list)))
        with open(origin_save, 'a') as f:
            for w_i in wrong_ind:
                f.write('%.5f %s\n' % (preds[w_i], tmp_list[w_i]))
        ind += 1




