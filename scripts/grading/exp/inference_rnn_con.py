# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: inference_rnn.py
@Date: 2019/12/24 
@Time: 8:55
@Desc: 多个rnn权重的输入图片的改变，测算得分的一致性并前向推理
'''
import os
import time
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
            r'Shengfuyou_1th\Positive',
            r'Shengfuyou_1th\Negative',
            r'Shengfuyou_2th\Positive',
            r'Shengfuyou_2th\Negative',
            r'3D_Shengfuyou_3th\Positive',
            r'3D_Shengfuyou_3th\Negative',
        ],
        'our': [
            r'Positive\Shengfuyou_3th',
            r'Negative\ShengFY-N-L240(origin date)',
            r'Positive\Shengfuyou_4th',
            r'Positive\Shengfuyou_5th\svs-20',
            # r'Positive\Tongji_4th',
            # r'Negative\Tongji_4th_neg',
            # r'Positive\Tongji_5th',
        ],
        'SZSQ_originaldata': [
            r'Shengfuyou_1th',
            r'Shengfuyou_3th\positive\Shengfuyou_3th_positive_40X',
            r'Shengfuyou_3th\negative\Shengfuyou_3th_negative_40X',
            r'Shengfuyou_5th\positive\Shengfuyou_5th_positive_40X',
            r'Shengfuyou_6th\Shengfuyou_6th_negtive_40X',
            r'Shengfuyou_7th\positive\Shengfuyou_7th_positive_40x',
            r'Shengfuyou_7th\negative\Shengfuyou_7th_negative_40x',
            r'Shengfuyou_8th\positive\pos_ascus',
            r'Shengfuyou_8th\positive\pos_hsil',
            r'Shengfuyou_8th\positive\pos_lsil',
            r'Shengfuyou_8th\negative',
            # r'Tongji_3th\positive\tongji_3th_positive_40x',
            # r'Tongji_3th\negative\tongji_3th_negtive_40x',
            # r'Tongji_4th\positive',
            # r'Tongji_4th\negative',
            # r'Tongji_5th\tongji_5th_positive\tongji_5th_positive_7us',
            # r'Tongji_5th\tongji_5th_negative\tongji_5th_negative_7us',
            # r'Tongji_6th\positive',
            # r'Tongji_6th\negative',
            # r'Tongji_7th\positive',
            # r'Tongji_7th\negative',
            # r'Tongji_8th\positive',
            # r'Tongji_8th\negative',
            # r'Tongji_9th\positive',
            # r'Tongji_9th\negative',
            # r'XiaoYuwei\positive',
            # r'XiaoYuwei\negative',
            # r'XiaoYuwei2\positive',
            # r'XiaoYuwei2\negative',
        ],
        'SrpData': [
            # r'out\xyw1\positive',
            # r'out\xyw1\negative',
            # r'out\xyw2\positive',
            # r'out\xyw2\negative',
            # r'out\xyw3\positive',
            # r'out\xyw3\negative',
            #
            # r'out_csh\xyw1\positive',
            # r'out_csh\xyw1\negative',
            # r'out_csh\xyw2\positive',
            # r'out_csh\xyw2\negative',
            # r'out_csh\xyw3\positive',
            # r'out_csh\xyw3\negative',
            #
            # r'out2\xyw1\positive',
            # r'out2\xyw1\negative',
            # r'out2\xyw2\positive',
            # r'out2\xyw2\negative',
            # r'out2\xyw3\positive',
            # r'out2\xyw3\negative',
            #
            # r'out2_bak\xyw1\positive',
            # r'out2_bak\xyw1\negative',
            # r'out2_bak\xyw2\positive',
            # r'out2_bak\xyw2\negative',
            # r'out2_bak\xyw3\positive',
            # r'out2_bak\xyw3\negative',
            #
            # r'out2_bak2\xyw1\positive',
            # r'out2_bak2\xyw1\negative',
            # r'out2_bak2\xyw2\positive',
            # r'out2_bak2\xyw2\negative',
            # r'out2_bak2\xyw3\positive',
            # r'out2_bak2\xyw3\negative',
            #
            # r'out-2019-11-17\xyw1\positive',
            # r'out-2019-11-17\xyw1\negative',
            # r'out-2019-11-17\xyw2\positive',
            # r'out-2019-11-17\xyw2\negative',
            # r'out-2019-11-17\xyw3\positive',
            # r'out-2019-11-17\xyw3\negative',
            #
            # r'out-2019-11-19\xyw1\positive',
            # r'out-2019-11-19\xyw1\negative',
            # r'out-2019-11-19\xyw2\positive',
            # r'out-2019-11-19\xyw2\negative',
            # r'out-2019-11-19\xyw3\positive',
            # r'out-2019-11-19\xyw3\negative',
        ],
    }
    slds_root = r'H:\TCTDATA'
    preds_root = r'H:\fql\rnnResult\rnn1000'

    gamma_root = r'J:\liusibo\DataBase\RnnData\top100_nothres_384\gamma'

    wsi_data = WSIDataSet(slds_root, preds_root, read_in_dicts)
    """gpu config"""
    gpu_id = '2'
    nb_gpu = len(gpu_id.split(','))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    """encoder weights"""
    resnet_weight = r'H:\weights\000000_ValidModels\190920_1010_model1_szsq700_model2_szsq658\szsq_model2_658_pred.h5'

    """rnn weights"""
    rnn_weight_list = [
        # r'F:\LiuSibo\Exps\191223_rnn_explore\rnnbase\new_rnn_all_10_balance\weights\2_3_01-0.88-0.91.h5',
        # r'F:\LiuSibo\Exps\191223_rnn_explore\rnnbase\new_rnn_all_10_balance\weights\8_10_01-0.88-0.94.h5',

        # r'F:\LiuSibo\Exps\191223_rnn_explore\rnnbase\new_rnn_all_20_balance\weights\0_0_03-0.85-0.85.h5',
        # r'F:\LiuSibo\Exps\191223_rnn_explore\rnnbase\new_rnn_all_20_balance\weights\2_9_01-0.81-0.88.h5',

        r'F:\LiuSibo\Exps\191223_rnn_explore\rnnbase\new_rnn_all_30_balance\weights\0_0_03-0.86-0.83.h5',
        r'F:\LiuSibo\Exps\191223_rnn_explore\rnnbase\new_rnn_all_30_balance\weights\3_12_01-0.80-0.88.h5',
    ]

    """set encoder"""
    print('setting resnet encoder')
    encoder = resnet_encoder((256, 256, 3))
    encoder.trainable = False
    encoder.load_weights(resnet_weight)
    print('load encoder %s' % resnet_weight)

    """encode imgs"""
    start = 0
    nb_rnn = 30
    shift_times = 10
    save_name = 'test_analysis_03'
    rng = [start, start+nb_rnn+shift_times]

    names = []
    labels = []
    gamma_encoded = []

    since = time.time()
    for i, wsi_item in enumerate(wsi_data):
        print('[%d/%d]' % (i+1, len(wsi_data)))

        name = wsi_item[1].slide_dir.lstrip(slds_root)
        labels.append(wsi_item[0])
        names.append(name)

        gamma_save = os.path.join(gamma_root, name)
        gamma_dir_list = [glob(os.path.join(gamma_save, '%d_*.tif' % n))[0] for n in range(rng[0], rng[1])]
        pool = Pool(16)
        imgs = pool.map(trans_img, gamma_dir_list)
        pool.close()
        pool.join()

        gamma_encoded.append(encoder.predict(np.stack(imgs))[1])
        print('avg time: %.3f/slide' % ((time.time()-since)/(i+1)))
    K.clear_session()

    """set rnn and predict"""
    rnn = simple_rnn((nb_rnn, 2048), 512)
    rnn_result = []
    for rnn_weight in rnn_weight_list:
        rnn.load_weights(rnn_weight)
        print('load rnn: %s' % rnn_weight)
        # w_result = []
        for shift in range(shift_times):
            rnn_result.append(list(rnn.predict(np.stack(gamma_encoded)[:, shift: shift+nb_rnn]).ravel()))
            # w_result.append(list(rnn.predict(np.stack(gamma_encoded)[:, shift: shift+nb_rnn]).ravel()))
        # 按照规则处理
        # （待补充）
        # rnn_result.append(w_result)
    K.clear_session()

    """analysis of results"""
    results_save = r'F:\LiuSibo\Exps\191223_rnn_explore\rnnbase\%s_shift.xlsx' %save_name
    writer = pd.ExcelWriter(results_save)
    result_df = rnn_result
    result_df.insert(0, labels)
    result_df.insert(0, names)
    result_df = pd.DataFrame(np.transpose(result_df))
    result_df.to_excel(writer, header=False, index=False)
    writer.close()

