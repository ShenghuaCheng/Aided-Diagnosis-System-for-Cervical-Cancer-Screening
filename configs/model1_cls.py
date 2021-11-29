# -*- coding: utf-8 -*-
"""
This file is a part of project "Aided-Diagnosis-System-for-Cervical-Cancer-Screening".
See https://github.com/ShenghuaCheng/Aided-Diagnosis-System-for-Cervical-Cancer-Screening for more information.

File name: model1_cls.py
Description: example for model1_cls
"""

import os
from core.config.resnet_base import Config as MyConfig


class Config(MyConfig):
    def __init__(self):
        super(Config, self).__init__()
        # --------------  model config --------------------- #
        # --------------  data loader config --------------------- #
        self.use_multiprocessing = False
        self.nb_workers = 1
        # --------------  dataset config --------------------- #
        # create dataset
        from core.data.datasets import ResnetDataset
        self._dataset = ResnetDataset(self.dataset_config, self.with_mask)
        # --------------  transform config --------------------- #
        # --------------  training config --------------------- #
        self.config_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        # save real config file
        os.makedirs(os.path.join(self.output_dir, self.config_name), exist_ok=True)
        self._dataset.write_config(os.path.join(self.output_dir, self.config_name, "dataset.xlsx"))
