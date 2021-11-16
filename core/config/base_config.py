# -*- coding: utf-8 -*-
"""
This file is a part of project "Aided-Diagnosis-System-for-Cervical-Cancer-Screening".
See https://github.com/ShenghuaCheng/Aided-Diagnosis-System-for-Cervical-Cancer-Screening for more information.

File name: base_config
Description: abstract base class for all train config.
"""

import pprint
from abc import ABCMeta, abstractmethod
from tabulate import tabulate

from keras.models import Model


class BaseConfig(metaclass=ABCMeta):

    def __init__(self):
        self.output_dir = "./OUTPUTS"

    @abstractmethod
    def create_model(self, weight=None) -> Model:
        raise NotImplementedError

    @abstractmethod
    def get_optimizer(self):
        raise NotImplementedError

    @abstractmethod
    def get_loss(self):
        raise NotImplementedError

    @abstractmethod
    def get_metrics(self):
        raise NotImplementedError

    @abstractmethod
    def get_lr_scheduler(self):
        raise NotImplementedError

    @abstractmethod
    def get_callbacks(self):
        raise NotImplementedError

    @abstractmethod
    def get_train_loader(self):
        pass

    @abstractmethod
    def get_validate_loader(self):
        pass

    @abstractmethod
    def get_test_loader(self):
        pass

    def __repr__(self):
        table_header = ["keys", "values"]
        exp_table = [
            (str(k), pprint.pformat(v))
            for k, v in vars(self).items()
            if not k.startswith("_")
        ]
        return tabulate(exp_table, headers=table_header, tablefmt="fancy_grid")
