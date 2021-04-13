# -*- coding:utf-8 -*-
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

    origin_root = r"root to cropped images"
    gamma_root = r'root to cropped images with gamma crct'

    wsi_data = WSIDataSet(slds_root, preds_root, read_in_dicts)

    """gpu config"""
    gpu_id = '0'
    nb_gpu = len(gpu_id.split(','))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    """encoder weights"""
    resnet_weight = r'model2.h5'
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
    w_save_root = r'rnn_%d\weights' % nb_rnn
    log_root = r'rnn_%d\logs' % nb_rnn
    os.makedirs(w_save_root, exist_ok=True)
    os.makedirs(log_root, exist_ok=True)

    """training"""
    with open(r"totalTestSlide.json", "r") as f:
        all_test = json.load(f)
    test_sld_name = []
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

        all_pos_idx = [idx for idx, l in enumerate(all_label) if l]
        all_neg_idx = [idx for idx, l in enumerate(all_label) if not l]

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

        K.clear_session()
        rnn = simple_rnn(rnn_input, 512)
        rnn.compile(optimizer=Adam(lr), loss=['binary_crossentropy'], metrics=['binary_accuracy'])
        es = EarlyStopping(patience=4)
        rp = ReduceLROnPlateau(patience=2, epsilon=1e-8)

        if tmp_weight:
            rnn.load_weights(tmp_weight)

        for order_shuffle in range(order_shuffle_times):
            if len(all_train_pos_idx) <= len(all_train_neg_idx):
                pos_idx = all_train_pos_idx
                neg_idx = random.sample(all_train_neg_idx, len(all_train_pos_idx))
            elif len(all_train_pos_idx) > len(all_train_neg_idx):
                pos_idx = random.sample(all_train_pos_idx, len(all_train_neg_idx))
                neg_idx = all_train_neg_idx

            train_idx = list(np.intersect1d(all_train_idx, pos_idx + neg_idx))

            all_data = [np.random.permutation(item) for item in all_data]

            filename = "%s_%s_{epoch:02d}-{val_binary_accuracy:.2f}-{binary_accuracy:.2f}.h5" % (str(img_trans), str(order_shuffle))
            ckpt_saver = ModelCheckpoint(os.path.join(w_save_root, filename),
                                         monitor='val_binary_accuracy',
                                         save_best_only=True,
                                         save_weights_only=True)

            hist = rnn.fit(np.stack(all_data)[train_idx], np.stack(all_label)[train_idx],
                           validation_data=[np.stack(all_data)[all_val_idx], np.stack(all_label)[all_val_idx]],
                           epochs=10, verbose=1, shuffle=True, callbacks=[ckpt_saver, tensorboard, es, rp])
        rnn.save_weights(os.path.join(w_save_root, 'tmp.h5'))
        tmp_weight = os.path.join(w_save_root, 'tmp.h5')

