# -*- coding: utf-8 -*-
"""
This file is a part of project "Aided-Diagnosis-System-for-Cervical-Cancer-Screening".
See https://github.com/ShenghuaCheng/Aided-Diagnosis-System-for-Cervical-Cancer-Screening for more information.

File name: model2_cls.py
Description: example for model2_cls
"""

import os

from core.models import model2_clf
from core.config.resnet_base import Config as MyConfig


class Config(MyConfig):
    def __init__(self):
        super(Config, self).__init__()
        # --------------  model config --------------------- #
        self._model_func = model2_clf
        self.input_size = (256, 256)
        self.input_mpp = 0.243
        # --------------  data loader config --------------------- #
        # --------------  dataset config --------------------- #
        self.dataset_config = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                           "datasets/model2/model2_cls_sample.xlsx")
        # create dataset
        from core.data.datasets import ResnetDataset
        self._dataset = ResnetDataset(self.dataset_config, self.with_mask)
        # --------------  transform config --------------------- #
        # --------------  training config --------------------- #
        self.init_lr = 0.001
        self.config_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        # save real config file
        os.makedirs(os.path.join(self.output_dir, self.config_name), exist_ok=True)
        self._dataset.write_config(os.path.join(self.output_dir, self.config_name, "dataset.xlsx"))
