# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: train_rnn.py
@Date: 2019/12/24 
@Time: 10:39
@Desc: 此脚本用于训练rnn模型
'''
import os
import random
import numpy as np
import json
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import keras.backend as K

from utils.networks.wsi_related import simple_rnn
from utils.grading.dataset import WSIDataSet
from utils.grading.train_func import imgs_encoder


if __name__ == '__main__':
    """dataset setting"""
    read_in_dicts = {
        '': [
            r'Shengfuyou_1th\Positive',
            r'Shengfuyou_1th\Negative',
            r'Shengfuyou_2th\Positive',
            r'Shengfuyou_2th\Negative',
            # # r'3D_Shengfuyou_3th\Positive',
            # # r'3D_Shengfuyou_3th\Negative',
        ],
        'our': [
            r'Positive\Shengfuyou_3th',
            r'Negative\ShengFY-N-L240(origin date)',
            r'Positive\Shengfuyou_4th',
            r'Positive\Shengfuyou_5th',
            # r'Positive\Shengfuyou_5th\svs-20',
            r'Positive\Tongji_4th',
            # r'Negative\Tongji_4th_neg',
            r'Positive\Tongji_5th',
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
            r'Tongji_3th\positive\tongji_3th_positive_40x',
            r'Tongji_3th\negative\tongji_3th_negtive_40x',
            r'Tongji_4th\positive',
            r'Tongji_4th\negative',
            r'Tongji_5th\tongji_5th_positive\tongji_5th_positive_7us',
            r'Tongji_5th\tongji_5th_negative\tongji_5th_negative_7us',
            r'Tongji_6th\positive',
            r'Tongji_6th\negative',
            r'Tongji_7th\positive',
            r'Tongji_7th\negative',
            r'Tongji_8th\positive',
            r'Tongji_8th\negative',
            r'Tongji_9th\positive',
            r'Tongji_9th\negative',
            r'XiaoYuwei\positive',
            r'XiaoYuwei\negative',
            r'XiaoYuwei2\positive',
            r'XiaoYuwei2\negative',
        ],
        'SrpData': [
        #     r'out\xyw1\positive',
        #     r'out\xyw1\negative',
        #     r'out\xyw2\positive',
        #     r'out\xyw2\negative',
        #     r'out\xyw3\positive',
        #     r'out\xyw3\negative',
        #
        #     r'out_csh\xyw1\positive',
        #     r'out_csh\xyw1\negative',
        #     r'out_csh\xyw2\positive',
        #     r'out_csh\xyw2\negative',
        #     r'out_csh\xyw3\positive',
        #     r'out_csh\xyw3\negative',
        #
        #     r'out2\xyw1\positive',
        #     r'out2\xyw1\negative',
        #     r'out2\xyw2\positive',
        #     r'out2\xyw2\negative',
        #     r'out2\xyw3\positive',
        #     r'out2\xyw3\negative',
        #
        #     r'out2_bak\xyw1\positive',
        #     r'out2_bak\xyw1\negative',
        #     r'out2_bak\xyw2\positive',
        #     r'out2_bak\xyw2\negative',
        #     r'out2_bak\xyw3\positive',
        #     r'out2_bak\xyw3\negative',
        #
        #     r'out2_bak2\xyw1\positive',
        #     r'out2_bak2\xyw1\negative',
        #     r'out2_bak2\xyw2\positive',
        #     r'out2_bak2\xyw2\negative',
        #     r'out2_bak2\xyw3\positive',
        #     r'out2_bak2\xyw3\negative',
        #
        #     r'out-2019-11-17\xyw1\positive',
        #     r'out-2019-11-17\xyw1\negative',
        #     r'out-2019-11-17\xyw2\positive',
        #     r'out-2019-11-17\xyw2\negative',
        #     r'out-2019-11-17\xyw3\positive',
        #     r'out-2019-11-17\xyw3\negative',
        #
        #     r'out-2019-11-19\xyw1\positive',
        #     r'out-2019-11-19\xyw1\negative',
        #     r'out-2019-11-19\xyw2\positive',
        #     r'out-2019-11-19\xyw2\negative',
        #     r'out-2019-11-19\xyw3\positive',
        #     r'out-2019-11-19\xyw3\negative',
        #
            r"tj10\tj10-pos\BD",
            r"tj10\tj10-neg\BD",
            r"tj11\positive\BD",
            r"tj11\negative\BD",
            r"xiehe1\positive",
            r"xiehe1\negative",
            r"zhongliuyiyuan1\negative",
        ],
    }
    slds_root = r'H:\TCTDATA'
    preds_root = r"I:\20200810_Lean2BD\M12_Results"

    origin_root = r"I:\20200810_Lean2BD\top100"
    gamma_root = r'I:\20200810_Lean2BD\top100_gamma'

    wsi_data = WSIDataSet(slds_root, preds_root, read_in_dicts)

    """gpu config"""
    gpu_id = '2'
    nb_gpu = len(gpu_id.split(','))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    """encoder weights"""
    resnet_weight = r'H:\weights\200810_Lean2BD\selected_w\models\L2BD_model2_272_encoder.h5'
    img_shape = (256, 256, 3)
    rng = [0, 30]
    nb_rnn = 30
    # rng = [0, 30]
    # nb_rnn = 20
    # rng = [0, 20]
    # nb_rnn = 10
    """rnn config"""
    lr = 1e-3
    rnn_input = (nb_rnn, 2048)
    """ train config"""
    img_trans_times = 15
    order_shuffle_times = 15
    w_save_root = r'I:\20200810_Lean2BD\rnn_all\rnn_all_%d_balance\weights' % nb_rnn
    log_root = r'I:\20200810_Lean2BD\rnn_all\rnn_all_%d_balance\logs' % nb_rnn
    os.makedirs(w_save_root, exist_ok=True)
    os.makedirs(log_root, exist_ok=True)

    """training"""
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

    tensorboard = TensorBoard(log_root)
    tmp_weight = None
    for img_trans in range(img_trans_times):
        names, labels, origin_encoded, gamma_encoded = imgs_encoder(wsi_data, slds_root, origin_root, gamma_root,
                                                                    resnet_weight, img_shape,
                                                                    rng, nb_rnn)
        all_data = origin_encoded + gamma_encoded
        all_label = labels + labels
        all_name = names + names

        # 本段为了使训练集的样本均衡，而获取训练集的阳性阴性idx
        all_pos_idx = [idx for idx, l in enumerate(all_label) if l]
        all_neg_idx = [idx for idx, l in enumerate(all_label) if not l]
        # 划分train和test idx
        all_train_idx = []
        all_val_idx = []
        for idx, n in enumerate(all_name):
            n = os.path.split(n)[-1].split('.')[0]
            if n in test_sld_name:
                all_val_idx.append(idx)
            else:
                all_train_idx.append(idx)

        all_train_pos_idx = list(np.intersect1d(all_pos_idx, all_train_idx))
        all_train_neg_idx = list(np.intersect1d(all_neg_idx, all_train_idx))
        # 获取困难阳性切片，加重比例
        with open(r'F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\rnn_exp\hard_pos.txt', 'r') as f:
            hard_list = [l.strip() for l in f.readlines()]
        hard_idx = [idx for idx, n in enumerate(all_name) if n in hard_list]
        with open(r'F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\rnn_exp\BD.txt', 'r') as f:
            BD_list = [l.strip() for l in f.readlines()]
        BD_idx = [idx for idx, n in enumerate(all_name) if n in BD_list]

        K.clear_session()
        rnn = simple_rnn(rnn_input, 512)
        rnn.compile(optimizer=Adam(lr), loss=['binary_crossentropy'], metrics=['binary_accuracy'])
        es = EarlyStopping(patience=4)
        rp = ReduceLROnPlateau(patience=2, epsilon=1e-8)

        if tmp_weight:
            rnn.load_weights(tmp_weight)

        for order_shuffle in range(order_shuffle_times):
            # 样本均衡
            if len(all_train_pos_idx) <= len(all_train_neg_idx):
                pos_idx = all_train_pos_idx
                neg_idx = random.sample(all_train_neg_idx, len(all_train_pos_idx))
            elif len(all_train_pos_idx) > len(all_train_neg_idx):
                pos_idx = random.sample(all_train_pos_idx, len(all_train_neg_idx))
                neg_idx = all_train_neg_idx

            train_idx = list(np.intersect1d(all_train_idx, pos_idx + neg_idx))

            # 内部排列顺序扰动
            all_data = [np.random.permutation(item) for item in all_data]

            filename = "%s_%s_{epoch:02d}-{val_binary_accuracy:.2f}-{binary_accuracy:.2f}.h5" % (str(img_trans), str(order_shuffle))
            ckpt_saver = ModelCheckpoint(os.path.join(w_save_root, filename),
                                         monitor='val_binary_accuracy',
                                         save_best_only=True,
                                         save_weights_only=True)

            # hist = rnn.fit(np.stack(all_data)[train_idx], np.stack(all_label)[train_idx],
            #                validation_data=[np.stack(all_data)[all_val_idx], np.stack(all_label)[all_val_idx]],
            #                epochs=10, verbose=1, shuffle=True, callbacks=[ckpt_saver, tensorboard, es, rp])
            hist = rnn.fit(np.stack(all_data)[train_idx + hard_idx*4 + BD_idx*10], np.stack(all_label)[train_idx + hard_idx*4 + BD_idx*10],
                           validation_data=[np.stack(all_data)[all_val_idx], np.stack(all_label)[all_val_idx]],
                           epochs=10, verbose=1, shuffle=True, callbacks=[ckpt_saver, tensorboard, es, rp])
        rnn.save_weights(os.path.join(w_save_root, 'tmp.h5'))
        tmp_weight = os.path.join(w_save_root, 'tmp.h5')


