# -*- coding:utf-8 -*-
import os
import random
import time
from glob2 import glob
from multiprocessing.dummy import Pool
import numpy as np
import keras.backend as K
from utils.grading.preprocess import enhance_img
from utils.networks.wsi_related import resnet_encoder


def imgs_encoder(wsi_data, slds_root, origin_root, gamma_root, resnet_weight, img_shape, rng, nb_rnn):
    """对传入的数据集进行图片读取变换以及编码供rnn使用
    :param wsi_data: 组织好的数据集
    :param slds_root: 切片根目录
    :param origin_root: 原始图片根目录
    :param gamma_root: gamma变换后图片根目录
    :param resnet_weight: 用于编码的resnet权重
    :param img_shape: 输入resnet图片的形状
    :param rng: 获取切片中top图片的范围 如前二十[0, 20]
    :param nb_rnn: 获取top图片的数量 如随机选十张 10
    :return:
    """
    K.clear_session()
    # set encoder
    print('setting resnet encoder')
    encoder = resnet_encoder(img_shape)
    encoder.trainable = False
    encoder.load_weights(resnet_weight)
    print('load encoder %s' % resnet_weight)
    names = []
    labels = []
    gamma_encoded = []
    origin_encoded = []

    since = time.time()
    for i, wsi_item in enumerate(wsi_data):
        name = wsi_item[1].slide_dir.lstrip(slds_root)
        print('[%d/%d] %s ' % (i+1, len(wsi_data), name))

        gamma_save = os.path.join(gamma_root, os.path.splitext(name)[0])
        origin_save = os.path.join(origin_root, os.path.splitext(name)[0])
        try:
            gamma_dir_list = [glob(os.path.join(gamma_save, '%.2d_*' % n))[0] for n in range(rng[0], rng[1])]
            origin_dir_list = [glob(os.path.join(origin_save, '%.2d_*' % n))[0] for n in range(rng[0], rng[1])]
        except:
            print("no images %s" % wsi_item[1].slide_dir)
            continue

        if len(gamma_dir_list) < nb_rnn or len(origin_dir_list) < nb_rnn:
            print("not enough images %s" % wsi_item[1].slide_dir)
            continue

        gamma_dir_list = random.sample(gamma_dir_list, nb_rnn)
        origin_dir_list = random.sample(origin_dir_list, nb_rnn)

        pool = Pool(16)
        origin_imgs = pool.map(enhance_img, origin_dir_list)
        pool.close()
        pool.join()

        pool = Pool(16)
        gamma_imgs = pool.map(enhance_img, gamma_dir_list)
        pool.close()
        pool.join()

        labels.append(wsi_item[0])
        names.append(name)
        origin_encoded.append(encoder.predict(np.stack(origin_imgs))[1])
        gamma_encoded.append(encoder.predict(np.stack(gamma_imgs))[1])
        print('avg time: %.3f/slide' % ((time.time()-since)/(i+1)))
    return names, labels, origin_encoded, gamma_encoded
