# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: Screening
@File: test_read_in.py
@Date: 2019/3/30 
@Time: 21:28
@Desc: 用于读入训练或测试的图片数据
'''

import os
import random
import multiprocessing.dummy as multiprocessing

import numpy as np
import cv2

from others.manu_yjy_code.read_image.data_enhancement import Sharp, Gauss, RGB_trans, HSV_trans, rotate, rotate_batch


def img_size_adapt(img, out_size=(921, 921)):
    """依据数据集将视野相同的数据调整到相同的尺寸
    :param img: 输入图片
    :param out_size: 输出尺寸 (w, h)
    :return: 返回调整尺寸后的图片
    """
    if img is None:
        raise ValueError('The image should not be "None".'
                         ' Please check the input array '
                         'to make sure image was read correctly')
    img = cv2.resize(img, out_size)
    return img


def center_crop(img, crop_size=(256, 256)):
    """中心裁剪
    :param img: 输入图片
    :param crop_size: 中心裁剪尺寸 (w, h)
    :return: 返回裁剪后的图片
    """
    if img is None:
        raise ValueError('The image should not be "None".'
                         ' Please check the input array '
                         'to make sure image was read correctly')
    img_shape = img.shape

    if img_shape[0] < crop_size[1] or img_shape[1] < crop_size[0]:
        raise ValueError('The image shape is small than the crop size')

    if img_shape[0] == crop_size[1] and img_shape[1] == crop_size[0]:
        return img
    else:
        h_start = int((img_shape[0] - crop_size[1])/2)
        w_start = int((img_shape[1] - crop_size[0])/2)
        img = img[h_start: h_start+crop_size[1], w_start: w_start+crop_size[0]]
    return img


def random_crop(img, crop_size=(512, 512), crop_num=1):
    """随机裁剪crop_num个图片
    :param img: 输入图片
    :param crop_size: 裁剪尺寸 (w, h)
    :param crop_num: 裁剪数量
    :return: 返回裁剪后四维列表 (n, h, w, c) 裁剪点起始坐标数组(start_h, start_w)
    """
    if img is None:
        raise ValueError('The image should not be "None".'
                         ' Please check the input array '
                         'to make sure image was read correctly')

    img_shape = img.shape
    if img_shape[0] < crop_size[1] or img_shape[1] < crop_size[0]:
        raise ValueError('The image shape is small than the crop size')


    crop_index = []  # (h_start, w_start)
    img_batch = []
    if not crop_num:
        return img_batch, crop_index

    if img_shape[0] == crop_size[1] and img_shape[1] == crop_size[0]:
        img_batch.append(img)
        crop_index.append((0, 0))
        return img_batch, crop_index
    else:
        for i in range(crop_num):
            h_start = random.randint(0, img_shape[0] - crop_size[1])
            w_start = random.randint(0, img_shape[1] - crop_size[0])
            img_batch.append(img[h_start: h_start+crop_size[1], w_start: w_start+crop_size[0]])
            crop_index.append((h_start, w_start))
    return img_batch, crop_index


def index_crop(img, crop_index, crop_size=(512, 512)):
    """依据给定的坐标来裁剪图片
    :param img:
    :param crop_index: 给定坐标数组 (start_h, start_w)
    :param crop_size: (w, h)
    :return:
    """
    img_batch=[]
    for crop_ind in crop_index:
        img_batch.append(img[crop_ind[0]: crop_ind[0] + crop_size[1], crop_ind[1]: crop_ind[1] + crop_size[0]])
    return img_batch


def img_enhance(img, img_mask=None):
    """对图片进行变换
    :param img: 输入需要变换的图片
    :return: 返回变换后的图片
    """
    m = random.randint(0, 2)
    if m:
        n = np.binary_repr(np.random.randint(0, 16), width=4)
        if int(n[0]):
            img = Sharp(img)
        if int(n[1]):
            img = Gauss(img)
        if int(n[2]):
            img = HSV_trans(img)
        if int(n[3]):
            img = RGB_trans(img)
        if img_mask is None:
            img = rotate(img)
            return img
        else:
            img, img_mask = rotate_batch([img, img_mask])
            return [img, img_mask]
    if img_mask is None:
        return img
    else:
        return [img, img_mask]


def img_scale(img, out_size=(768, 768), scale_ratio=1):
    """对图片视野范围进行扰动，适用于ScaleData
    :param img:
    :param out_size: (w, h)
    :param scale_ratio: 扰动的倍率，范围: [0.80, 1.20]
    :return: 返回视野扰动后的图片
    """
    if img is None:
        raise ValueError('The image should not be "None".'
                         ' Please check the input array '
                         'to make sure image was read correctly')
    img_shape = img.shape
    if img_shape[0] < out_size[1] or img_shape[1] < out_size[0]:
        print('img_shape:' + str(img_shape[0]))
        print('out_size:' + str(out_size[0]))
        raise ValueError('The image shape is small than the out size')

    scaled_w = int(out_size[0]*scale_ratio)
    scaled_h = int(out_size[1]*scale_ratio)
    img = center_crop(img, (scaled_w, scaled_h))
    img = cv2.resize(img, out_size)
    return img


def img_norm(img):
    """图片归一化
    :param img: 输入图片
    :return: 返回归一化后的矩阵
    """
    img = (img/255.-0.5)*2
    return img


def read_in(dirImg,
            init_size=(921, 921),
            enlarge_size=(768, 768),
            train_size=(512, 512),
            crop_num=1,
            norm_flag=False,
            enhance_flag=False,
            scale_flag=False,
            include_center=False,
            mask_flag=False,
            dirMask=None):
    """读图函数，读入指定变换后的图片，可选读入对应mask
    :param dirImg: 图片路径
    :param init_size: 统一指定数据集初始图片尺寸，一般为 输出大小*1.5*1.2
    :param enlarge_size: 冗余输出图片大小，一般为 输出大小*1.5
    :param out_size: 输出图片尺寸
    :param crop_num: 冗余图片上随机裁剪输出大小图片数量
    :param norm_flag: 是否归一化
    :param enhance_flag: 数据增强
    :param scale_flag: 是否Pixel Size扰动
    :param include_center: 随机裁剪是否含有中央裁剪，若有则最后一张为中央裁剪
    :param mask_flag: 读入mask的信号
    :param dirMask: mask的路径
    :return:
    """
    dirImg_0 = dirImg
    if mask_flag and dirMask is None:
        raise ValueError('Missing "dirMask"')
    if isinstance (dirImg,str):
        dirImg = dirImg
    elif not isinstance (dirImg,list):
        dirImg = list(dirImg)[0]
#    dirImg    
#    dirImg_o = dirImg
    if dirImg[0] is 'J':
        dirImg = 'H'+dirImg[1:]
#    print(dirImg)
    img = cv2.imread(dirImg)
    if img is None:
#        print(dirImg_o)
        print(dirImg_0,'ValueError')
        raise ValueError('The image should not be "None".'
                         ' Please check the input array '
                         'to make sure image was read correctly')
    if mask_flag:
        img_mask = cv2.imread(dirMask)
        if img_mask is None:
            raise ValueError('Get image path "%s", image mask path "%s"'
                             'Please check image mask' % (dirImg, dirMask))

    # 将数据集内图片尺寸归一
    img = img_size_adapt(img, out_size=init_size)
    if mask_flag:
        img_mask = img_size_adapt(img_mask, out_size=init_size)

    # 是否有Pixel Size扰动
    if scale_flag:
        scale_ratio = round(random.uniform(0.9, 1.1), 2)
    else:
        scale_ratio = 1

    img = img_scale(img, enlarge_size, scale_ratio)
    if mask_flag:
        img_mask = img_scale(img_mask, enlarge_size, scale_ratio)

    # 裁剪大图得到喂入网络图片
    if include_center:
        imgbatch, crop_index = random_crop(img, train_size, crop_num-1)
        imgbatch.append(center_crop(img, train_size))
        crop_index.append((int((img.shape[0] - train_size[1])/2),
                           int((img.shape[1] - train_size[0])/2)))
    else:
        # print(dirImg)
        imgbatch, crop_index = random_crop(img, train_size, crop_num)

    if mask_flag:
        imgmaskbatch = index_crop(img_mask, crop_index, train_size)

    # 增强
    if enhance_flag:
        if mask_flag:
            imgbatch = [img_enhance(img, img_mask) for img, img_mask in zip(imgbatch, imgmaskbatch)]
        else:
            imgbatch = [img_enhance(img) for img in imgbatch]
    else:
        if mask_flag:
            imgbatch = [[img, img_mask] for img, img_mask in zip(imgbatch, imgmaskbatch)]

    # 归一化
    if norm_flag:
        if mask_flag:
            imgbatch = [[img_norm(img_pair[0]), img_pair[1]] for img_pair in imgbatch]
        else:
            imgbatch = [img_norm(img) for img in imgbatch]

    return imgbatch


def multiprocess_read_in(pool_num=16, **kwargs):
    """多进程读图
    :param pool_num: 进程数 default=16
    :param kwargs: read_in 所需所有参数，其中有关路径传入路径列表，若有mask则需要对应路径
    :return: 返回读入的所有图片
    """

    pool = multiprocessing.Pool(pool_num)

    img_dir_list = kwargs['dirImg']
    img_batchs = []
    for i, dirImg in enumerate(img_dir_list):
        if not kwargs['mask_flag']:
            config = (dirImg, kwargs['init_size'], kwargs['enlarge_size'], kwargs['train_size'],
                      kwargs['crop_num'],  kwargs['norm_flag'], kwargs['enhance_flag'], kwargs['scale_flag'], kwargs['include_center'],
                      kwargs['mask_flag'], kwargs['dirMask'])
        else:
            config = (dirImg, kwargs['init_size'], kwargs['enlarge_size'], kwargs['train_size'],
                      kwargs['crop_num'], kwargs['norm_flag'], kwargs['enhance_flag'], kwargs['scale_flag'], kwargs['include_center'],
                      kwargs['mask_flag'], kwargs['dirMask'][i])
        img_batchs.append(pool.apply_async(read_in, config))
    pool.close()
    pool.join()

    img_batchs = [img_batch.get() for img_batch in img_batchs]

    return img_batchs




if __name__ == "__main__":
    train_size = (1936, 1216)
    enlarge_size = (int(train_size[0]*1.5), int(train_size[1]*1.5))
    init_size = (int(train_size[0]*1.5*1.2), int(train_size[1]*1.5*1.2))
    dirImg = r'I:\BigData\train\1_ASCUS\1M05_99_77555_125987_162_162_ASCUS.tif'
    dirMask = r'I:\BigData\localmask\1_ASCUS\1M05_99_77555_125987_162_162_ASCUS.tif'
    imgbatchs = multiprocess_read_in(16, dirImg=[dirImg],
            init_size=init_size,
            enlarge_size=enlarge_size,
            train_size=train_size,
            crop_num=7,
            norm_flag=True,
            enhance_flag=False,
            scale_flag=False,
            include_center=False,
            mask_flag=False,
            dirMask=None)

    # import matplotlib.pyplot as plt
    # for imgbatch in imgbatchs:
    #     for img in imgbatch:
    #         plt.figure()
    #         # plt.subplot(121)
    #         plt.imshow(img/2+0.5)
    #         # plt.subplot(122)
    #         # plt.imshow(img[1])

    # imgbatch = read_in(dirImg, init_size, enlarge_size, train_size, crop_num=10,
    #                    scale_flag=False, enhance_flag=True, include_center=False)
    # for i, img in enumerate(imgbatch):
    #     cv2.imwrite('I:/liusibo/%d_img.tif' % i, img)
    #     # cv2.imwrite('I:/liusibo/%d_mask.tif' % i, img[1])
