# -*- coding: utf-8 -*-
"""
This file is a part of project "Aided-Diagnosis-System-for-Cervical-Cancer-Screening".
See https://github.com/ShenghuaCheng/Aided-Diagnosis-System-for-Cervical-Cancer-Screening for more information.

File name: inference_base
Description: base config file for calculate whole WSI.
"""

import pprint

from loguru import logger
from tabulate import tabulate
import keras.backend as K

from core.models import model1, model2, wsi_clf
from core.utils import load_weights


class InferenceConfig:

    def __init__(self):
        # --------------  model 1 config --------------------- #
        self.m1_weight = {
            "classify": "",
            "locate": ""
        }
        self.m1_input_size = (512, 512)
        self.m1_input_mpp = 0.486
        self.m1_score_threshold = 0.5
        self.m1_results_num_range = [600, 1200]

        self.m1_batch_size = 4
        # --------------  model 2 config --------------------- #
        self.m2_weight = ""
        self.m2_input_size = (256, 256)
        self.m2_input_mpp = 0.243

        self.m2_batch_size = 8
        # --------------  wsi clf config --------------------- #
        self.wsi_clf_n = [10, 20, 30]
        self.wc_weights = {
            self.wsi_clf_n[0]: [
                "",
                ""
            ],
            self.wsi_clf_n[1]: [
                "",
                ""
            ],
            self.wsi_clf_n[2]: [
                "",
                ""
            ],
        }
        # --------------  WSI config --------------------- #
        self.overlap = 1/4  # or (1/4, 1/4) for (left overlapping ratio, right overlapping ratio)
        self.gamma = False
        # --------------  output config --------------------- #
        self.most_suspicious_n = 10

        self.config_file = None
        self.output_dir = None

    def get_model1(self):
        return load_weights(model1(), [self.m1_weight[k] for k in ["locate", "classify"]])

    def get_model2(self):
        return load_weights(model2(), self.m2_weight)

    def get_wsi_clf(self, top_n, id):
        return load_weights(wsi_clf(top_n)(), self.wc_weights[top_n][id])

    def wipe_models(self):
        logger.info("clear model to avoid naming conflict when loading multiple models.")
        K.clear_session()

    def __repr__(self):
        table_header = ["keys", "values"]
        exp_table = [
            (str(k), pprint.pformat(v))
            for k, v in vars(self).items()
            if not k.startswith("_")
        ]
        return tabulate(exp_table, headers=table_header, tablefmt="fancy_grid")
