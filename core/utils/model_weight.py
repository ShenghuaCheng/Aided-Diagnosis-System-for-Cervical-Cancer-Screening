# -*- coding: utf-8 -*-
"""
This file is a part of project "Aided-Diagnosis-System-for-Cervical-Cancer-Screening".
See https://github.com/ShenghuaCheng/Aided-Diagnosis-System-for-Cervical-Cancer-Screening for more information.

File name: model_weight
Description: 
"""

from collections import Iterable
from loguru import logger

from keras import callbacks

__all__ = [
    "load_weights",
]


def load_weights(model, weights=None):
    """ load weight for model, inplace opt
    """
    if weights is None:
        return model
    elif isinstance(weights, str):
        model.load_weights(weights, by_name=True)
        logger.info(f"load {weights}")
    elif isinstance(weights, Iterable):
        for w in weights:
            model = load_weights(model, w)
    return model

