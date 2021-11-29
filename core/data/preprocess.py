# -*- coding: utf-8 -*-
"""
This file is a part of project "Aided-Diagnosis-System-for-Cervical-Cancer-Screening".
See https://github.com/ShenghuaCheng/Aided-Diagnosis-System-for-Cervical-Cancer-Screening for more information.

File name: preprocess
Description: preprocess data
"""

from .augmentations import Augmentations

__all__ = [
    "Preprocess"
]


class Preprocess:
    def __init__(
            self,
            dst_size,
            crop,
            gamma=0.,
            scale=0.,
            sharp=0.5,
            blur=0.5,
            hsv_disturb=0.5,
            rgb_switch=0.5,
            rotate=0.5,
            h_flip=0.5,
            v_flip=0.5,
            norm=[-1, 1],
    ):
        self.dst_size = dst_size
        self.pre_funcs = [
            Augmentations.RandomCrop(crop, dst_size),
            Augmentations.RandomGamma(gamma),
            Augmentations.RandomScale(scale),
            Augmentations.RandomSharp(sharp),
            Augmentations.RandomGaussainBlur(blur),
            Augmentations.RandomHSVDisturb(hsv_disturb),
            Augmentations.RandomRGBSwitch(rgb_switch),
            Augmentations.RandomRotate90(rotate),
            Augmentations.RandomHorizontalFlip(h_flip),
            Augmentations.RandomVerticalFlip(v_flip),
        ]
        self.preprocess = Augmentations.Compose(*self.pre_funcs)
        self.normalize = Augmentations.Normalization(norm)

    def __call__(self, image):
        h, w, ch = image.shape
        image = self.preprocess(image)
        if ch == 4:
            mask = image[..., -1].copy()
            image = image[..., :-1].copy()
        elif ch == 3:
            mask = None
        else:
            raise ValueError("image channel should be 3 or 4(with mask)")
        image = self.normalize(image)
        return image, mask
