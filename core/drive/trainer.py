# -*- coding: utf-8 -*-
"""
This file is a part of project "Aided-Diagnosis-System-for-Cervical-Cancer-Screening".
See https://github.com/ShenghuaCheng/Aided-Diagnosis-System-for-Cervical-Cancer-Screening for more information.

File name: trainer
Description: trainer for training
"""

import os

from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard

from core.config import BaseConfig
from core.common import check_dir

from loguru import logger

__all__ = ["Trainer"]


class Trainer:
    def __init__(self, args, config: BaseConfig):
        self.config = config
        self.args = args
        # train
        self.max_epoch = config.max_epoch
        self.ini_epoch = 0 if args.start_epoch is None else args.start_epoch
        # data
        self.batch_size = self.args.batch_size
        self.max_queue = config.max_queue_size
        self.workers = config.nb_workers
        self.mp = config.use_multiprocessing
        # record
        self.callbacks = config.get_callbacks()
        self.record_dir = os.path.join(config.output_dir, config.config_name)
        check_dir(self.record_dir, True)

    def train(self):
        self.before_train()
        try:
            self.on_train()
        except Exception:
            raise
        finally:
            self.after_train()

    def before_train(self):
        logger.info(f"args: {self.args}")
        logger.info(f"config:\n{self.config}")

        # compile model
        model = self.config.create_model(weight=self.args.weights)
        logger.info(f"Model summary:\n{model.summary()}")

        optimizer = self.config.optimizer
        loss = self.config.loss
        metrics = self.config.metrics
        model.compile(optimizer, loss, metrics)

        lr_scheduler = self.config.get_lr_scheduler()
        if lr_scheduler is not None:
            self.callbacks.append(lr_scheduler)

        # save weights
        weight_path = os.path.join(self.record_dir, "Epoch_{epoch:04d}_{val_loss:.3f}_{val_acc:.3f}.h5")
        best_weight_path = os.path.join(self.record_dir, "best_ckpt.h5")
        cb_save_each_weight = ModelCheckpoint(weight_path, "val_acc", save_weights_only=True)
        cb_save_best_weight = ModelCheckpoint(best_weight_path, "val_acc", save_best_only=True, mode="max", save_weights_only=True)
        self.callbacks += [
            cb_save_each_weight,
            cb_save_best_weight
        ]

        # monitor training
        log_path = os.path.join(self.record_dir, "train_log.csv")
        cb_csv_log = CSVLogger(log_path, append=True)
        cb_tensorboard = TensorBoard(self.record_dir, write_graph=False)
        self.callbacks += [
            cb_csv_log,
            cb_tensorboard
        ]

        if self.ini_epoch != 0:
            logger.info(f"resume train from {self.ini_epoch - 1}")

        # dataset init
        self.train_loader = self.config.get_train_loader()
        self.train_loader.batch_size = self.batch_size
        self.val_loader = self.config.get_validate_loader()
        self.val_loader.batch_size = self.batch_size

        self.model = model

        logger.info("Start training...")

    def on_train(self):
        self.model.fit_generator(
            generator=self.train_loader,
            steps_per_epoch=None,
            epochs=self.max_epoch,
            verbose=1,
            callbackes=self.callbacks,
            validation_data=self.val_loader,
            validation_steps=None,
            class_weight=None,
            max_queue_size=self.max_queue,
            workers=self.workers,
            use_multiprocessing=self.mp,
            shuffle=True,
            initial_epoch=self.ini_epoch
        )

    def after_train(self):
        pass
