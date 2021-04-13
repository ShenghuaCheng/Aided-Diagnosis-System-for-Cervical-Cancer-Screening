# -*- coding:utf-8 -*-
import os
from functools import partial
from multiprocessing.dummy import Pool
import cv2
import openslide
import numpy as np
import pandas as pd
from others.manu_yjy_code.networks.resnet50_2classes import ResNet, ResNet_f

IN_SHAPE = {
    'm1': (512, 512, 3),
    'm2': (256, 256, 3),
}


MODEL_NAMES = {
    'm1': ["LR_model"],
    'm2': ["HR_model", "Baseline", "Mined", "Enhanced", "Origin"]
}


GROUPS = ["A", "B", "C", "D", "E", "F",
          "G", "H", "I", "J", "K", "L"]


SLIDE_TOP = {
    'E': {
        'P': [
        ],
        'N': [
        ]
    },
    'F': {
        'P': [
        ],
        'N': [
        ]
    }
}


def rd_img(img_dir, crop_size=(256,256), resize=None):
    img = cv2.imread(img_dir)
    if crop_size == (256, 256) and resize is None:
        # resize
        img = cv2.resize(img, (1852, 1852))
    elif crop_size == (512, 512) and resize is None:
        # resize
        img = cv2.resize(img, (926, 926))
    elif resize is not None:
        # resize
        img = cv2.resize(img, resize)
    else:
        raise RuntimeError
    # center crop
    img = center_crop(img, crop_size)

    # BGR2RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # normalize to [-1, 1] and return
    return (img/255 - 0.5)*2


def center_crop(img, size=(256, 256)):
    w, h, _ = img.shape
    x_ini = h//2 - size[1]//2
    y_ini = w//2 - size[0]//2
    return img[x_ini: x_ini+size[1], y_ini: y_ini+size[0], :]


def rd_imgs(img_dirs, in_shape, re_size=None):
    rd_func = partial(rd_img, crop_size=in_shape[:2], resize=re_size)
    pool = Pool(32)
    data = pool.map(rd_func, img_dirs)
    pool.close()
    pool.join()
    return np.stack(data)


def create_encoder(m_w, in_shape=(256, 256, 3)):
    """ create model2 encoder with weight m2_w
    """
    print("set {} model with weight: {}".format(in_shape, m_w))
    Net = ResNet_f(input_shape=in_shape)
    Net.load_weights(m_w)
    return Net


def create_encoders(m_name):
    if m_name == 'HR_model':
        in_shape = (256,256,3)
        m_w = r""  # model2
    elif m_name == 'LR_model':
        in_shape = (512,512,3)
        m_w = r""  # model1
    elif m_name == 'Baseline':
        in_shape = (256,256,3)
        m_w = r""  # model2
    elif m_name == 'Mined':
        in_shape = (256,256,3)
        m_w = r""  # model2
    elif m_name == 'Enhanced':
        in_shape = (256,256,3)
        m_w = r""  # model2
    elif m_name == 'Origin':
        in_shape = (256,256,3)
        m_w = r""  # model2
    else: raise ValueError('no keyword {}'.format(m_name))
    print('Create {}:\nInput: {}\nWeights:{}'.format(m_name, in_shape, m_w))
    return create_encoder(m_w, in_shape), in_shape


def parse_group(group, TEST_DATASET_XLSX=r'.xlsx'):
    # 获取group对应的样本列表文件列表
    sheet = pd.read_excel(TEST_DATASET_XLSX, sheetname=group)
    file_list = sheet['txt_name'].tolist()
    sample_num = sheet['sample_num'].tolist()
    label = sheet['label'].tolist()
    total_num = sheet['total_num'].tolist()
    return file_list, label, sample_num, total_num


def get_test_data(file_img_dir, label, in_shape):
    if os.path.exists(file_img_dir.replace('.txt', '.npz')):
        data =np.load(file_img_dir.replace('.txt', '.npz'))
        print('Return cached images and label: %s' % file_img_dir.replace('.txt', '.npz'))
        return data['data'], data['label']
    else:
        print('reading ...')
        s_dirs = [l .strip() for l in open(file_img_dir, 'r').readlines()]
        imgs = rd_imgs(s_dirs, in_shape)
        labs = np.array([label]*len(s_dirs))
        print('%s done, num: %d' % (file_img_dir, len(labs)))
        return imgs, labs


def get_openslide_attrs(handle):
    if ".mrxs" in handle._filename:
        attrs = {
            "mpp": float(handle.properties[openslide.PROPERTY_NAME_MPP_X]),
            "level": handle.level_count,
            "width": int(handle.properties[openslide.PROPERTY_NAME_BOUNDS_WIDTH]),
            "height": int(handle.properties[openslide.PROPERTY_NAME_BOUNDS_HEIGHT]),
            "bound_init": (int(handle.properties[openslide.PROPERTY_NAME_BOUNDS_X]),
                           int(handle.properties[openslide.PROPERTY_NAME_BOUNDS_Y])),
            "level_ratio": int(handle.level_downsamples[1])
        }
    elif ".svs" in handle._filename:
        try:
            attrs = {
                "mpp": float(handle.properties[openslide.PROPERTY_NAME_MPP_X]),
                "level": handle.level_count,
                "width": int(handle.dimensions[0]),
                "height": int(handle.dimensions[1]),
                "bound_init": (0, 0),
                "level_ratio": int(handle.level_downsamples[1])
            }
        except KeyError:
            attrs = {
                "mpp": float(handle.properties['aperio.MPP'].strip(';')),
                "level": handle.level_count,
                "width": int(handle.dimensions[0]),
                "height": int(handle.dimensions[1]),
                "bound_init": (0, 0),
                "level_ratio": int(handle.level_downsamples[1])
            }
    return attrs

