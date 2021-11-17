# -*- coding: utf-8 -*-
"""
This file is a part of project "Aided-Diagnosis-System-for-Cervical-Cancer-Screening".
See https://github.com/ShenghuaCheng/Aided-Diagnosis-System-for-Cervical-Cancer-Screening for more information.

File name: augmentations.py
Description: augmentation functions.
"""

import functools
import random
import cv2
import numpy as np
from skimage.exposure import adjust_gamma
from loguru import logger

__all__ = [
    "Augmentations",
    "StylisticTrans",
    "SpatialTrans",
]


class Augmentations:
    """
    All parameters in each augmentations have been fixed to a suitable range.
    img = [size, size, ch]
    ch = 3: only img
    ch = 4: img with mask at 4th dim
    """
    @staticmethod
    def Compose(*funcs):
        funcs = list(funcs)
        func_names = [f.__name__ for f in funcs]
        # ensure the norm opt is the last opt
        if 'norm' in func_names:
            idx = func_names.index('norm')
            funcs = funcs[:idx] + funcs[idx:] + [funcs[idx]]

        def compose(img: np.ndarray):
            return functools.reduce(lambda f, g: lambda x: g(f(x)), funcs)(img)
        return compose
    """
    # ===========================================================================================================
    # random stylistic augmentations
    """

    @staticmethod
    def RandomGamma(p: float=0.5):
        def random_gamma(img: np.ndarray):
            if random.random() < p:
                gamma = 0.6 + random.random()*0.6
                img[..., :3] = StylisticTrans.gamma_adjust(img[..., :3], gamma)
            return img
        return random_gamma

    @staticmethod
    def RandomSharp(p: float=0.5):
        def random_sharp(img: np.ndarray):
            if random.random() < p:
                sigma = 8.3 + random.random()*0.4
                img[..., :3] = StylisticTrans.sharp(img[..., :3], sigma)
            return img
        return random_sharp

    @staticmethod
    def RandomGaussainBlur(p: float=0.5):
        def random_gaussian_blur(img: np.ndarray):
            if random.random() < p:
                sigma = 0.1 + random.random()*1
                img[..., :3] = StylisticTrans.gaussian_blur(img[..., :3], sigma)
            return img
        return random_gaussian_blur

    @staticmethod
    def RandomHSVDisturb(p: float=0.5):
        def random_hsv_disturb(img: np.ndarray):
            if random.random() < p:
                k = np.random.random(3)*[0.1, 0.8, 0.45] + [0.95, 0.7, 0.75]
                b = np.random.random(3)*[6, 20, 18] + [-3, -10, -10]
                img[..., :3] = StylisticTrans.hsv_disturb(img[..., :3], k.tolist(), b.tolist())
            return img
        return random_hsv_disturb

    @staticmethod
    def RandomRGBSwitch(p: float=0.5):
        def random_rgb_switch(img: np.ndarray):
            if random.random() < p:
                bgr_seq = list(range(3))
                random.shuffle(bgr_seq)
                img[..., :3] = StylisticTrans.bgr_switch(img[..., :3], bgr_seq)
            return img
        return random_rgb_switch
    """
    # ===========================================================================================================
    # random spatial augmentations, funcs can be implement to tiles and their masks.  
    """
    @staticmethod
    def RandomRotate90(p: float=0.5):
        def random_rotate90(img: np.ndarray):
            if random.random() < p:
                angle = 90*random.randint(1,3)
                img = SpatialTrans.rotate(img, angle)
            return img
        return random_rotate90

    @staticmethod
    def RandomHorizontalFlip(p: float=0.5):
        def random_horizontal_flip(img: np.ndarray):
            if random.random() < p:
                img = SpatialTrans.flip(img, 0)
            return img
        return random_horizontal_flip

    @staticmethod
    def RandomVerticalFlip(p: float=0.5):
        def random_vertical_flip(img: np.ndarray):
            if random.random() < p:
                img = SpatialTrans.flip(img, 1)
            return img
        return random_vertical_flip

    @staticmethod
    def RandomScale(p: float=0.5):
        def random_scale(img: np.ndarray):
            if random.random() < p:
                ratio = 0.8 + random.random()*0.4
                img = SpatialTrans.scale(img, ratio, True)
            return img
        return random_scale

    @staticmethod
    def RandomCrop(p: float=1., size: tuple=(512, 512)):
        def random_crop(img: np.ndarray):
            if random.random() < p:
                # for a large FOV, control the translate range
                new_shape = list(img.shape[:2][::-1])
                if img.shape[0] > size[1] * 1.5:
                    new_shape[1] = int(size[1] * 1.5)
                if img.shape[1] > size[0] * 1.5:
                    new_shape[0] = int(size[0] * 1.5)
                img = SpatialTrans.center_crop(img.copy(), tuple(new_shape))
                # do translate
                xy = np.random.random(2)*(np.array(img.shape[:2]) - list(size))
                bbox = tuple(xy.astype(np.int).tolist() + list(size))
                img = SpatialTrans.crop(img, bbox)
            else:
                img = SpatialTrans.center_crop(img, size)
            return img
        return random_crop
    
    @staticmethod
    def Normalization(rng: list=[-1, 1]):
        def norm(img: np.ndarray):
            img = StylisticTrans.normalization(img, rng)
            return img
        return norm
    
    @staticmethod
    def CenterCrop(size: tuple=(512, 512)):
        def center_crop(img: np.ndarray):
            img = SpatialTrans.center_crop(img, size)
            return img
        return center_crop


class StylisticTrans:
    # TODO Some implementations of augmentation need a efficient way
    """
    set of augmentations applied to the content of image
    """
    @staticmethod
    def gamma_adjust(img: np.ndarray, gamma: float):
        """ adjust gamma
        :param img: a ndarray, better a BGR
        :param gamma: gamma, recommended value 0.6, range [0.6, 1.2]
        :return: a ndarray
        """
        return adjust_gamma(img.copy(), gamma)

    @staticmethod
    def sharp(img: np.ndarray, sigma: float):
        """sharp image
        :param img: a ndarray, better a BGR
        :param sigma: sharp degree, recommended range [8.3, 8.7]
        :return: a ndarray
        """
        kernel = np.array([[-1, -1, -1], [-1, sigma, -1], [-1, -1, -1]], np.float32) / (sigma - 8)  # 锐化
        return cv2.filter2D(img.copy(), -1, kernel=kernel)

    @staticmethod
    def gaussian_blur(img: np.ndarray, sigma: float):
        """blurring image
        :param img: a ndarray, better a BGR
        :param sigma: blurring degree, recommended range [0.1, 1.1]
        :return: a ndarray
        """
        return cv2.GaussianBlur(img.copy(), (int(6 * np.ceil(sigma) + 1), int(6 * np.ceil(sigma) + 1)), sigma)

    @staticmethod
    def hsv_disturb(img: np.ndarray, k: list, b: list):
        """ disturb the hsv value
        :param img: a BGR ndarray
        :param k: low_b = [0.95, 0.7, 0.75] ,upper_b = [1.05, 1.5, 1.2]
        :param b: low_b = [-3, -10, -10] ,upper_b = [3, 10, 8]
        :return: a BGR ndarray
        """
        img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)
        img = img.astype(np.float)
        for ch in range(3):
            img[..., ch] = k[ch] * img[..., ch] + b[ch]
        img = np.uint8(np.clip(img, np.array([0, 1, 1]), np.array([180, 255, 255])))
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    @staticmethod
    def bgr_switch(img: np.ndarray, bgr_seq: list):
        """ switch bgr
        :param img: a ndarray, better a BGR
        :param bgr_seq: new ch seq
        :return: a ndarray
        """
        return img.copy()[..., bgr_seq]

    @staticmethod
    def normalization(img: np.ndarray, rng: list):
        """normalize image according to min and max
        :param img: a ndarray
        :param rng: normalize image value to range[min, max]
        :return: a ndarray
        """
        lb, ub = rng
        delta = ub - lb
        return (img.copy().astype(np.float)/255.) * delta + lb


class SpatialTrans:
    """
    set of augmentations applied to the spatial space of image
    """
    @staticmethod
    def rotate(img: np.ndarray, angle: int):
        """ rotate image
        # todo Square image and central rotate only, a universal version is needed
        :param img: a ndarray
        :param angle: rotate angle
        :return: a ndarray has same size as input, padding zero or cut out region out of picture
        """
        assert img.shape[0] == img.shape[1], "Square image needed."
        mat = cv2.getRotationMatrix2D(tuple(np.array(img.shape[:2])//2), angle, scale=1)
        return cv2.warpAffine(img.copy(), mat, img.shape[:2])

    @staticmethod
    def flip(img: np.ndarray, flip_axis: int):
        """flip image horizontal or vertical
        :param img: a ndarray
        :param flip_axis: 0 for horizontal, 1 for vertical
        :return:  a flipped image
        """
        return cv2.flip(img.copy(), flip_axis)

    @staticmethod
    def scale(img: np.ndarray, ratio: float, fix_size: bool=False):
        """scale image
        :param img: a ndarray
        :param ratio: scale ratio
        :param fix_size: return the center area of scaled image, size of area is same as the image before scaling
        :return:  a scaled image
        """
        shape = img.shape[:2][::-1]
        img = cv2.resize(img.copy(), None, fx=ratio, fy=ratio)
        if fix_size:
            img = SpatialTrans.center_crop(img, shape)
        return img

    @staticmethod
    def crop(img: np.ndarray, bbox: tuple):
        """crop image according to given bbox
        :param img: a ndarray
        :param bbox: bbox of cropping area (x, y, w, h)
        :return: cropped image,padding with zeros
        """
        ch = [] if len(img.shape) == 2 else [img.shape[-1]]
        template = np.zeros(list(bbox[-2:])[::-1]+ch)

        if (bbox[1] >= img.shape[0] or bbox[1] >= img.shape[1]) or (bbox[0]+bbox[2] <= 0 or bbox[1]+bbox[3] <= 0):
            logger.warning("Crop area contains nothing, return a zeros array {}".format(template.shape))
            return template
        
        foreground = img[
            np.maximum(bbox[1], 0): np.minimum(bbox[1]+bbox[3], img.shape[0]),
            np.maximum(bbox[0], 0): np.minimum(bbox[0]+bbox[2], img.shape[1]), :]

        template[
            np.maximum(-bbox[1], 0): np.minimum(-bbox[1]+img.shape[0], bbox[3]),
            np.maximum(-bbox[0], 0): np.minimum(-bbox[0]+img.shape[1], bbox[2]), :] = foreground
        return template.astype(np.uint8)

    @staticmethod
    def center_crop(img: np.ndarray, shape: tuple):
        """return the center area in shape
        :param img: a ndarray
        :param shape: center crop shape (w, h)
        :return:
        """
        center = np.array(img.shape[:2])//2
        init = center[::-1] - np.array(shape)//2
        bbox = tuple(init.tolist() + list(shape))
        return SpatialTrans.crop(img, bbox)
