# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: inference_rnn.py
@Date: 2019/12/24 
@Time: 8:55
@Desc: 多个rnn权重的简单前向推理
'''
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
        '3DHistech': [
            # r'Shengfuyou_1th\Positive',
            # r'Shengfuyou_1th\Negative',
            # r'Shengfuyou_2th\Positive',
            # r'Shengfuyou_2th\Negative',
            # # r'3D_Shengfuyou_3th\Positive',
            # # r'3D_Shengfuyou_3th\Negative',
        ],
        'WNLO': [
            # r'Positive\Shengfuyou_3th',
            # r'Negative\ShengFY-N-L240(origin date)',
            # r'Positive\Shengfuyou_4th',
            # # r'Positive\Shengfuyou_5th\svs-20',
            # r'Positive\Tongji_4th',
            # # r'Negative\Tongji_4th_neg',
            # r'Positive\Tongji_5th',
        ],
        'SZSQ_originaldata': [
            # r'Shengfuyou_1th',
            # r'Shengfuyou_3th\positive\Shengfuyou_3th_positive_40X',
            # r'Shengfuyou_3th\negative\Shengfuyou_3th_negative_40X',
            # r'Shengfuyou_5th\positive\Shengfuyou_5th_positive_40X',
            # r'Shengfuyou_6th\Shengfuyou_6th_negtive_40X',
            # r'Shengfuyou_7th\positive\Shengfuyou_7th_positive_40x',
            # r'Shengfuyou_7th\negative\Shengfuyou_7th_negative_40x',
            # r'Shengfuyou_8th\positive\pos_ascus',
            # r'Shengfuyou_8th\positive\pos_hsil',
            # r'Shengfuyou_8th\positive\pos_lsil',
            # r'Shengfuyou_8th\negative',
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
        'SpringRock': [
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
        # 'BD': [
        #
        #     r"Tongji_10th\positive\BD",
        #     # r"Tongji_10th\negative\BD",
        #     # r"Tongji_11th\positive\BD",
        #     # r"Tongji_11th\negative\BD",
        #     # r"Xiehe_1th\positive",
        #     # r"Xiehe_1th\negative",
        #     # r"Shengzhongliu_1th\positive",
        #     # r"Shengzhongliu_1th\negative",
        #
        # ],
        'Appd': [

            r"3D_TJ\P",
            r"3D_TJ\N",
            r"BD_SZL\P",
            r"BD_SZL\N",
            r"BD_TJ\P",
            r"BD_TJ\N",
            r"BD_XH\P",
            r"BD_XH\N",
            r"JY\P",
            r"JY\N",
            r"SZSQ_SFY\P",
            r"SZSQ_SFY\N",
        ],
    }
    slds_root = r'K:\liusibo\20201225_Manu_AppdSlide\Slides/'
    preds_root = r"K:\liusibo\20201225_Manu_AppdSlide\Res/"
    # slds_root = r'H:\TCTDATA'
    # preds_root = r"I:\20200810_Lean2BD\M12_Results"

    origin_root = r"H:\liusibo\manuscripts_top_tiles_for_rnn\top30/"
    gamma_root = r'H:\liusibo\manuscripts_top_tiles_for_rnn\top30/'

    wsi_data = WSIDataSet(slds_root, preds_root, read_in_dicts)
    """gpu config"""
    gpu_id = '1'
    nb_gpu = len(gpu_id.split(','))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    """encoder weights"""
    # resnet_weight = r'H:\weights\200810_Lean2BD\selected_w\models\L2BD_model2_272_encoder.h5'
    resnet_weight = r'F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\manu_yjy\select_w\m2_Block_102.h5'

    """rnn weights"""
    rnn_weight_list = [
        # r'F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\manu_yjy\select_w\rnn\10_0_6_02-0.89-0.92.h5',
        # r'F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\manu_yjy\select_w\rnn\10_5_0_06-0.91-0.95.h5',
        #
        # r'F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\manu_yjy\select_w\rnn\20_1_2_07-0.90-0.91.h5',
        # r'F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\manu_yjy\select_w\rnn\20_9_0_05-0.89-0.91.h5',
        #
        r'F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\manu_yjy\select_w\rnn\30_1_1_09-0.87-0.91.h5',
        r'F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\manu_yjy\select_w\rnn\30_4_0_01-0.87-0.87.h5',
    ]
    # rnn_weight_list = [
    #     # r'I:\20200810_Lean2BD\L2BD_10BD_5HLA\selected_weights\10_9_7_01-0.89-0.96.h5',
    #     # r'I:\20200810_Lean2BD\L2BD_10BD_5HLA\selected_weights\10_14_4_01-0.88-0.95.h5',
    #
    #     # r'I:\20200810_Lean2BD\L2BD_10BD_5HLA\selected_weights\20_1_2_02-0.90-0.89.h5',
    #     # r'I:\20200810_Lean2BD\L2BD_10BD_5HLA\selected_weights\20_1_4_05-0.91-0.89.h5',
    #
    #     r'I:\20200810_Lean2BD\L2BD_10BD_5HLA\selected_weights\30_0_0_04-0.90-0.86.h5',
    #     r'I:\20200810_Lean2BD\L2BD_10BD_5HLA\selected_weights\30_2_9_01-0.89-0.91.h5',
    # ]
    # rnn_weight_list = [
    #     # r'I:\20200810_Lean2BD\l2bd_subcls_rnn\selected_weights\10_0_9_01-0.92-0.91.h5',
    #     # r'I:\20200810_Lean2BD\l2bd_subcls_rnn\selected_weights\10_9_9_01-0.91-0.91.h5',
    #
    #     # r'I:\20200810_Lean2BD\l2bd_subcls_rnn\selected_weights\20_0_9_01-1.00-0.99.h5',
    #     # r'I:\20200810_Lean2BD\l2bd_subcls_rnn\selected_weights\20_6_9_01-0.98-0.97.h5',
    #
    #     # r'I:\20200810_Lean2BD\l2bd_subcls_rnn\selected_weights\30_0_9_01-0.99-0.98.h5',
    #     # r'I:\20200810_Lean2BD\l2bd_subcls_rnn\selected_weights\30_1_0_07-0.98-0.98.h5',
    # ]
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

    with open(r"F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\BDTestSlide.json", "r") as f:
        all_test = json.load(f)
    for k in all_test:
        test_sld_name += (all_test[k])

    """encode imgs"""
    start = 0
    nb_rnn = 30
    save_name = 'top30_appd_%d' %nb_rnn
    rng = [start, start+nb_rnn]

    names = []
    labels = []
    gamma_encoded = []

    since = time.time()
    for i, wsi_item in enumerate(wsi_data):
        print('[%d/%d]' % (i+1, len(wsi_data)))

        name = wsi_item[1].slide_dir.rsplit(slds_root, 1)[-1]
        gamma_save = os.path.join(gamma_root, os.path.splitext(name)[0])
        # gamma_save = os.path.join(origin_root, os.path.splitext(name)[0])

        # if os.path.split(name)[-1].split('.')[0] not in test_sld_name:
        #     continue

        try:
            gamma_dir_list = [glob(os.path.join(gamma_save, '%.2d_*' % n))[0] for n in range(rng[0], rng[1])]
            # gamma_dir_list = [glob(os.path.join(gamma_save, '%.3d_*' % n))[0] for n in range(rng[0], rng[1])]
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
        # rnn.save(os.path.join(r'I:\20200810_Lean2BD\L2BD_10BD_5HLA\selected_weights\models', os.path.split(rnn_weight)[-1]))
        print('load rnn: %s' % rnn_weight)
        rnn_result.append(list(rnn.predict(np.stack(gamma_encoded)).ravel()))
    K.clear_session()

    """analysis of results"""
    results_save = r'K:\liusibo\20201225_Manu_AppdSlide\%s.xlsx' %save_name
    writer = pd.ExcelWriter(results_save)
    result_df = rnn_result
    result_df.insert(0, labels)
    result_df.insert(0, names)
    result_df = pd.DataFrame(np.transpose(result_df))
    result_df.to_excel(writer, header=False, index=False)
    writer.close()

