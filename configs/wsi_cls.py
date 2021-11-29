# -*- coding: utf-8 -*-
"""
This file is a part of project "Aided-Diagnosis-System-for-Cervical-Cancer-Screening".
See https://github.com/ShenghuaCheng/Aided-Diagnosis-System-for-Cervical-Cancer-Screening for more information.

File name: wsi_cls
Description:  example for rnn cls top10
"""

import os

from core.models import wsi_clf
from core.config.rnn_base import Config as MyConfig


class Config(MyConfig):
    def __init__(self):
        super(Config, self).__init__()
        # --------------  model config --------------------- #
        # self.top_n = 20  # if top 20
        # self._model_func = wsi_clf(self.top_n)
        # --------------  data loader config --------------------- #
        # --------------  dataset config --------------------- #
        from core.data.datasets import RnnDataset
        self._dataset = RnnDataset(self.dataset_config)
        # --------------  transform config --------------------- #
        # --------------  training config --------------------- #
        self.interval_image_re_encode = 1  # this is time consuming opt, interval should be long.

        self.config_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        # save real config file
        os.makedirs(os.path.join(self.output_dir, self.config_name), exist_ok=True)
        self._dataset.write_config(os.path.join(self.output_dir, self.config_name, "dataset.xlsx"))
