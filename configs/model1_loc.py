# -*- coding: utf-8 -*-
"""
This file is a part of project "Aided-Diagnosis-System-for-Cervical-Cancer-Screening".
See https://github.com/ShenghuaCheng/Aided-Diagnosis-System-for-Cervical-Cancer-Screening for more information.

File name: model1_loc.py
Description: example for model1_loc
"""

import os
from core.config.resnet_base import Config as MyConfig
from core.models import model1_loc


class Config(MyConfig):
    def __init__(self):
        super(Config, self).__init__()
        # --------------  model config --------------------- #
        self._model_func = model1_loc
        # --------------  data loader config --------------------- #
        # --------------  dataset config --------------------- #
        self.dataset_config = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                           "datasets/model1/model1_loc_sample.xlsx")
        self.with_mask = True
        # create dataset
        from core.data.datasets import ResnetDataset
        self._dataset = ResnetDataset(self.dataset_config, self.with_mask)
        # --------------  transform config --------------------- #
        self.translate = 1.
        # --------------  training config --------------------- #
        self.init_lr = 1e-4
        self.config_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        # save real config file
        os.makedirs(os.path.join(self.output_dir, self.config_name), exist_ok=True)
        self._dataset.write_config(os.path.join(self.output_dir, self.config_name, "dataset.xlsx"))

    def create_model(self, weight=None):
        import tensorflow as tf
        from keras.utils import multi_gpu_model
        from core.utils import load_weights
        from core.config.resnet_base import GPU_DEVICES

        nb_gpu = len(GPU_DEVICES)
        if nb_gpu > 1:
            with tf.device('/cpu:0'):
                model = self._model_func()
                model = load_weights(model, weight)
            self.model = multi_gpu_model(model, gpus=nb_gpu)
        else:
            model = self._model_func()
            self.model = load_weights(model, weight)
        return self.model

    def get_loss(self):
        return ["categorical_crossentropy"]

    def get_metrics(self):
        return ["categorical_accuracy"]
