# -*- coding: utf-8 -*-
"""
This file is a part of project "Aided-Diagnosis-System-for-Cervical-Cancer-Screening".
See https://github.com/ShenghuaCheng/Aided-Diagnosis-System-for-Cervical-Cancer-Screening for more information.

File name: lr_scheduler
Description: callback function of lr scheduler
"""

import math
import keras.backend as K

__all__ = [
    "Schedulers",
]

_EPSILON = K.epsilon()


class Schedulers:
    @staticmethod
    def Cosine(epochs, init_lr):
        def cosine(epoch):
            lr = init_lr*(math.cos((epoch*math.pi)/epochs) + 1)/2
            return lr if lr>_EPSILON else _EPSILON
        return cosine

    @staticmethod
    def Linear(epochs, init_lr):
        def linear(epoch):
            lr = init_lr*(1-epoch/epochs)
            return lr if lr>_EPSILON else _EPSILON
        return linear