# -*- coding: utf-8 -*-
"""
This file is a part of project "Aided-Diagnosis-System-for-Cervical-Cancer-Screening".
See https://github.com/ShenghuaCheng/Aided-Diagnosis-System-for-Cervical-Cancer-Screening for more information.

File name: model1_cls.py
Description: example for model1_cls
"""

from core.config.resnet_base import ResnetConfig


class Model1Cls(ResnetConfig):
    def __init__(self):
        super(Model1Cls, self).__init__()
        # --------------  model config --------------------- #
        # --------------  data loader config --------------------- #
        self.use_multiprocessing = True
        self.nb_workers = 4
        # --------------  dataset config --------------------- #
        # --------------  transform config --------------------- #
        # --------------  training config --------------------- #
