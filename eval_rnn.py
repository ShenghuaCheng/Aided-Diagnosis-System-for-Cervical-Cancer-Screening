# -*- coding:utf-8 -*-
import os
import time
import json
import numpy as np
import pandas as pd
from glob2 import glob
from multiprocessing.dummy import Pool
import keras.backend as K
from utils.networks.wsi_related import resnet_encoder, simple_rnn
from utils.grading.dataset import WSIDataSet
from utils.grading.preprocess import trans_img


if __name__ == '__main__':
    """dataset setting"""
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

#    origin_root = r"root to cropped images"
    gamma_root = r'root to cropped images with gamma crct'

    wsi_data = WSIDataSet(slds_root, preds_root, read_in_dicts)
    """gpu config"""
    gpu_id = '0'
    nb_gpu = len(gpu_id.split(','))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    """encoder weights"""
    resnet_weight = r'model2.h5'

    """rnn weights"""
    rnn_weight_list = [
        r'F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\manu_yjy\select_w\rnn\10_0_6_02-0.89-0.92.h5',
        r'F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\manu_yjy\select_w\rnn\10_5_0_06-0.91-0.95.h5',
    ]
    """set encoder"""
    print('setting resnet encoder')
    encoder = resnet_encoder((256, 256, 3))
    encoder.trainable = False
    encoder.load_weights(resnet_weight)
    print('load encoder %s' % resnet_weight)

    # 根据patch test抽样列表获取test切片
    with open(r"F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\totalTestSlide.json", "r") as f:
        all_test = json.load(f)
    test_sld_name = []
    for k in all_test:
        test_sld_name += (all_test[k])

    """encode imgs"""
    start = 0
    nb_rnn = 10
    save_name = 'inference_top%d' %nb_rnn
    rng = [start, start+nb_rnn]

    names = []
    labels = []
    gamma_encoded = []

    since = time.time()
    for i, wsi_item in enumerate(wsi_data):
        print('[%d/%d]' % (i+1, len(wsi_data)))

        name = wsi_item[1].slide_dir.rsplit(slds_root, 1)[-1]
        gamma_save = os.path.join(gamma_root, os.path.splitext(name)[0])

        if os.path.split(name)[-1].split('.')[0] not in test_sld_name:
            continue

        try:
            gamma_dir_list = [glob(os.path.join(gamma_save, '%.2d_*' % n))[0] for n in range(rng[0], rng[1])]
        except:
            print("no images %s" % wsi_item[1].slide_dir)
            continue

        pool = Pool(16)
        imgs = pool.map(trans_img, gamma_dir_list)
        pool.close()
        pool.join()
        labels.append(wsi_item[0])
        names.append(name)
        gamma_encoded.append(encoder.predict(np.stack(imgs))[1])
        print('avg time: %.3f/slide' % ((time.time()-since)/(i+1)))
    K.clear_session()

    """set rnn and predict"""
    rnn = simple_rnn((nb_rnn, 2048), 512)
    rnn_result = []
    for rnn_weight in rnn_weight_list:
        rnn.load_weights(rnn_weight)
        print('load rnn: %s' % rnn_weight)
        rnn_result.append(list(rnn.predict(np.stack(gamma_encoded)).ravel()))
    K.clear_session()

    """analysis of results"""
    results_save = r'.\%s.xlsx' %save_name
    writer = pd.ExcelWriter(results_save)
    result_df = rnn_result
    result_df.insert(0, labels)
    result_df.insert(0, names)
    result_df = pd.DataFrame(np.transpose(result_df))
    result_df.to_excel(writer, header=False, index=False)
    writer.close()

