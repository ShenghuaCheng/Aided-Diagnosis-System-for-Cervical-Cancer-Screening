# -*- coding: utf-8 -*-
"""
This file is a part of project "Aided-Diagnosis-System-for-Cervical-Cancer-Screening".
See https://github.com/ShenghuaCheng/Aided-Diagnosis-System-for-Cervical-Cancer-Screening for more information.

File name: rnn_base.py
Description: base class for rnn training.
"""

import os
from loguru import logger

from keras import callbacks, optimizers
from keras.utils import multi_gpu_model
import tensorflow as tf
from tensorflow.python.client import device_lib

import core
from core.models import wsi_clf
from core.config import BaseConfig
from core.utils import Schedulers, load_weights

GPU_DEVICES = [item.name for item in device_lib.list_local_devices() if item.device_type == 'GPU']


class Config(BaseConfig):
    def __init__(self):
        super().__init__()
        # --------------  model config --------------------- #
        self.top_n = 10
        self._model_func = wsi_clf(10)
        self.input_size = (256, 256)
        self.input_mpp = 0.243
        # --------------  data loader config --------------------- #
        self.max_queue_size = 10
        # FIXME 20211117: can not apply multiprocessing
        self.use_multiprocessing = False
        self.nb_workers = 1
        # --------------  dataset config --------------------- #
        self.dataset_config = os.path.join(os.path.dirname(os.path.dirname(core.__file__)),
                                           "datasets/wsi_clf/top_10/wsi_cls_top_10_sample.xlsx")
        self.top_rng = 10  # top image selected range is [0, top_n + top_rng]
        # create dataset
        self._dataset = None
        # --------------  transform config --------------------- #
        # for encode image
        self.norm_range = [-1., 1.]
        self.translate = 0.5
        self.gamma = 0.
        self.scale = 0.
        self.sharp = 0.5
        self.blur = 0.5
        self.hsv_disturb = 0.5
        self.rgb_switch = 0.5
        self.rotate = 0.5
        self.h_flip = 0.5
        self.v_flip = 0.5
        # for train features
        self.permutation = 0.5
        # --------------  training config --------------------- #
        self.max_epoch = 300
        self.interval_image_re_encode = 50  # this is time consuming opt, interval should be long.
        self.init_lr = 1e-3
        self.scheduler = None
        self.optimizer = 'Adam'
        self.config_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def create_model(self, weight=None):
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

    def get_optimizer(self):
        if self.optimizer is None:
            logger.warning("use default adam optimizer.")
            optimizer = optimizers.Adam
        elif self.optimizer.lower() == 'adam':
            optimizer = optimizers.Adam
        elif self.optimizer.lower() == 'sgd':
            optimizer = optimizers.SGD
        else:
            raise ValueError(f"no {self.optimizer} optimizer")
        return optimizer(lr=self.init_lr)

    def get_loss(self):
        return ["binary_crossentropy"]

    def get_metrics(self):
        return ["binary_accuracy"]

    def get_lr_scheduler(self):
        from keras.callbacks import ReduceLROnPlateau
        return ReduceLROnPlateau(patience=10, epsilon=1e-5)

    def get_callbacks(self):
        return []

    def get_train_loader(self):
        df_cfg = self._dataset.df_config["train"]
        if df_cfg is None:
            raise ValueError(f"train dataset is None, Check {self.dataset_config}.")
        from core.data.datasets import RnnDataloader
        from core.data import Preprocess
        # init preprocessor
        preprocess = Preprocess(
            self.input_size,
            self.translate,
            self.gamma,
            self.scale,
            self.sharp,
            self.blur,
            self.hsv_disturb,
            self.rgb_switch,
            self.rotate,
            self.h_flip,
            self.v_flip,
            self.norm_range
        )
        train_loader = RnnDataloader(
            self._dataset,
            "train",
            self.encoder,
            self.top_n,
            self.top_rng,
            self.input_mpp,
            preprocess,
            self.permutation,
            self.interval_image_re_encode
        )
        return train_loader

    def get_validate_loader(self):
        df_cfg = self._dataset.df_config["val"]
        if df_cfg is None:
            raise ValueError(f"val dataset is None, Check {self.dataset_config}.")
        from core.data.datasets import RnnDataloader
        from core.data import Preprocess
        # set all transform to zero
        preprocess = Preprocess(
            self.input_size,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            self.norm_range
        )
        val_loader = RnnDataloader(
            self._dataset,
            "val",
            self.encoder,
            self.top_n,
            0,  # rng is zero
            self.input_mpp,
            preprocess,
            0,  # no permutation
            self.interval_image_re_encode
        )
        return val_loader

    def get_test_loader(self):
        df_cfg = self._dataset.df_config["test"]
        if df_cfg is None:
            raise ValueError(f"test dataset is None, Check {self.dataset_config}.")
        from core.data.datasets import RnnDataloader
        from core.data import Preprocess
        # set all transform to zero
        preprocess = Preprocess(
            self.input_size,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            self.norm_range
        )
        test_loader = RnnDataloader(
            self._dataset,
            "test",
            self.encoder,
            self.top_n,
            0,
            self.input_mpp,
            preprocess,
            0,
            self.interval_image_re_encode
        )
        return test_loader

    def get_encoder(self, weights):
        from core.models import model2
        encoder = model2()
        self.encoder = load_weights(encoder, weights)
        return self.encoder

