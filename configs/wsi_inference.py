# -*- coding: utf-8 -*-
"""
This file is a part of project "Aided-Diagnosis-System-for-Cervical-Cancer-Screening".
See https://github.com/ShenghuaCheng/Aided-Diagnosis-System-for-Cervical-Cancer-Screening for more information.

File name: wsi_inference
Description: config for WSI inference.
"""

import os

from loguru import logger

from core.common import check_dir
from core.config.inference_base import InferenceConfig as MyConfig


class Config(MyConfig):

    def __init__(self):
        super(Config, self).__init__()

        import os
        weights_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets/pretrained_weights")
        # --------------  model 1 config --------------------- #
        self.m1_weight = {
            "classify": os.path.join(weights_dir, "m1_pred_Block_333.h5"),
            "locate": os.path.join(weights_dir, "m1_local_Block_374.h5")
        }
        self.m1_batch_size = 4
        # --------------  model 2 config --------------------- #
        self.m2_weight = os.path.join(weights_dir, "m2_Block_102.h5")
        self.m2_batch_size = 16
        # --------------  wsi clf config --------------------- #
        self.wc_weights = {
            self.wsi_clf_n[0]: [
                os.path.join(weights_dir, "wsi_clf", "10_0_6_02-0.89-0.92.h5"),
                os.path.join(weights_dir, "wsi_clf", "10_5_0_06-0.91-0.95.h5")
            ],
            self.wsi_clf_n[1]: [
                os.path.join(weights_dir, "wsi_clf", "20_1_2_07-0.90-0.91.h5"),
                os.path.join(weights_dir, "wsi_clf", "20_9_0_05-0.89-0.91.h5")
            ],
            self.wsi_clf_n[2]: [
                os.path.join(weights_dir, "wsi_clf", "30_1_1_09-0.87-0.91.h5"),
                os.path.join(weights_dir, "wsi_clf", "30_4_0_01-0.87-0.87.h5")
            ],
        }
        # --------------  output config --------------------- #
        self.config_file = os.path.basename(__file__).split('.')[0]
        self.output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "OUTPUTS", "results", self.config_file)
        check_dir(self.output_dir, True, logger)
