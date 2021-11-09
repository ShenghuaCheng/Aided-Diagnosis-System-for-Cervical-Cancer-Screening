# -*- coding: utf-8 -*-
"""
This file is a part of project "Aided-Diagnosis-System-for-Cervical-Cancer-Screening".
See https://github.com/ShenghuaCheng/Aided-Diagnosis-System-for-Cervical-Cancer-Screening for more information.

File name: models
Description: create related models.
"""

from __future__ import print_function
from .network_components import *

__all__ = [
    "model1_clf",
    "model1_loc",
    "model2_clf",
    "model1",
    "model2"
]

MODEL1_INPUT = (512, 512, 3)
MODEL2_INPUT = (256, 256, 3)


def model1_clf(frozen_layers=37):
    return resnet50_clf(MODEL1_INPUT, frozen_layers)


def model1_loc():
    return resnet50_loc(MODEL1_INPUT)


def model2_clf(frozen_layers=37):
    return resnet50_clf(MODEL2_INPUT, frozen_layers)


def model1():
    return resnet50_det(MODEL1_INPUT)


def model2():
    return resnet50_enc(MODEL2_INPUT)


def wsi_clf(top_n):
    return simple_rnn((top_n, 2048), 512)
