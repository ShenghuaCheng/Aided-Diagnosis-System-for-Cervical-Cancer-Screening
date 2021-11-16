# -*- coding: utf-8 -*-
"""
This file is a part of project "Aided-Diagnosis-System-for-Cervical-Cancer-Screening".
See https://github.com/ShenghuaCheng/Aided-Diagnosis-System-for-Cervical-Cancer-Screening for more information.

File name: resnet_base.py
Description: base class for model1 model2 training.
"""

import os
from loguru import logger

from keras import callbacks, optimizers
from keras.utils import multi_gpu_model
import tensorflow as tf
from tensorflow.python.client import device_lib

from core.models import model1_clf
from core.config import BaseConfig
from core.utils import Schedulers, load_weights

GPU_DEVICES = [item.name for item in device_lib.list_local_devices() if item.deice_type == 'GPU']


class ResnetConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        # --------------  model config --------------------- #
        self.model_func = model1_clf
        self.frozen_layers = 37
        self.input_size = (512, 512)
        self.input_mpp = 0.486
        # --------------  data loader config --------------------- #
        self.max_queue_size = 10
        self.use_multiprocessing = False
        self.nb_workers = 1
        # --------------  dataset config --------------------- #
        from core.data.datasets import ResnetDataset
        self.dataset_config = "./datasets/model1/model1_cls_sample.xlsx"
        self.with_mask = False
        # create dataset
        self.dataset = ResnetDataset(self.dataset_config, self.with_mask)
        # --------------  transform config --------------------- #
        self.norm_range = [-1., 1.]
        self.crop = 0.5
        self.gamma = 0.
        self.scale = 0.
        self.sharp = 0.5
        self.blur = 0.5
        self.hsv_disturb = 0.5
        self.rgb_switch = 0.5
        self.rotate = 0.5
        self.h_flip = 0.5
        self.v_flip = 0.5
        # --------------  training config --------------------- #
        self.max_epoch = 300
        self.init_lr = 0.01
        self.scheduler = None
        self.optimizer = 'Adam'
        self.config_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def create_model(self, weight=None):
        nb_gpu = len(GPU_DEVICES)
        if nb_gpu > 1:
            with tf.device('/cpu:0'):
                model = self.model_func(self.frozen_layers)
                model = load_weights(model, weight)
            self.model = multi_gpu_model(model, gpus=nb_gpu)
        else:
            model = self.model_func(self.frozen_layers)
            self.model = load_weights(model, weight)
        return self.model

    def optimizer(self):
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

    def loss(self):
        return ["binary_crossentropy"]

    def metrics(self):
        return ["binary_accuracy"]

    def get_lr_scheduler(self):
        if self.scheduler is None:
            logger.warning(f"no lr scheduler.")
            return None
        elif self.scheduler == "cos":
            scheduler = Schedulers.Cosine(self.max_epoch, self.init_lr)
        elif self.scheduler == "linear":
            scheduler = Schedulers.Linear(self.max_epoch, self.init_lr)
        else:
            raise ValueError(f"no {self.scheduler} scheduler")
        return callbacks.LearningRateScheduler(scheduler)

    def get_callbacks(self):
        return []

    def get_train_loader(self):
        df_cfg = self.dataset.df_config["train"]
        if df_cfg is None:
            raise ValueError(f"train dataset is None, Check {self.dataset_config}.")
        from core.data.datasets import ResnetDataloader
        from core.data import Preprocess
        # init preprocessor
        preprocess = Preprocess(
            self.input_size,
            self.crop,
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
        train_loader = ResnetDataloader(
            self.dataset,
            "train",
            self.input_size,
            self.input_mpp,
            preprocess
        )

    def get_validate_loader(self):
        df_cfg = self.dataset.df_config["val"]
        if df_cfg is None:
            raise ValueError(f"val dataset is None, Check {self.dataset_config}.")
        from core.data.datasets import ResnetDataloader
        from core.data import Preprocess
        # set all transform to zero
        preprocess = Preprocess(
            self.input_size,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            self.norm_range
        )
        val_loader = ResnetDataloader(
            self.dataset,
            "train",
            self.input_size,
            self.input_mpp,
            preprocess
        )

    def get_test_loader(self):
        df_cfg = self.dataset.df_config["test"]
        if df_cfg is None:
            raise ValueError(f"test dataset is None, Check {self.dataset_config}.")
        from core.data.datasets import ResnetDataloader
        from core.data import Preprocess
        # set all transform to zero
        preprocess = Preprocess(
            self.input_size,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            self.norm_range
        )
        test_loader = ResnetDataloader(
            self.dataset,
            "train",
            self.input_size,
            self.input_mpp,
            preprocess
        )
