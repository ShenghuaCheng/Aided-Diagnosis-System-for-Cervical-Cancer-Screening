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
        model_sum = []
        model.summary(print_fn=lambda l: model_sum.append(l))
        logger.info("Model summary:\n {}".format("\n".join(model_sum)))

        optimizer = self.config.get_optimizer()
        loss = self.config.get_loss()
        metrics = self.config.get_metrics()
        model.compile(optimizer, loss, metrics)

        lr_scheduler = self.config.get_lr_scheduler()
        if lr_scheduler is not None:
            self.callbacks.append(lr_scheduler)

        # save weights
        acc_name = [acc for acc in model.metrics_names if "acc" in acc]
        if len(acc_name):
            acc_name = acc_name[0]
            val_acc_name = "val_" + acc_name
            weight_path = os.path.join(self.record_dir, "Epoch_{epoch:04d}_{%s:.3f}_{%s:.3f}.h5" % (acc_name, val_acc_name))
            monitor = val_acc_name
            monitor_mode = "max"
        else:
            weight_path = os.path.join(self.record_dir, "Epoch_{epoch:04d}_{loss:.3f}_{val_loss:.3f}.h5")
            monitor = "val_loss"
            monitor_mode = "min"
        logger.info(f"Weight monitor: {monitor} Monitor mode: {monitor_mode}")
        best_weight_path = os.path.join(self.record_dir, "best_ckpt.h5")
        cb_save_each_weight = ModelCheckpoint(weight_path, monitor, save_weights_only=True)
        cb_save_best_weight = ModelCheckpoint(best_weight_path, monitor, save_best_only=True, mode=monitor_mode, save_weights_only=True)
        self.callbacks += [
            cb_save_each_weight,
            cb_save_best_weight
        ]

        # monitor training
        log_path = os.path.join(self.record_dir, "train_log.csv")
        cb_csv_log = CSVLogger(log_path, append=bool(self.ini_epoch))
        cb_tensorboard = TensorBoard(self.record_dir, write_graph=False)
        self.callbacks += [
            cb_csv_log,
            cb_tensorboard
        ]

        if self.ini_epoch != 0:
            logger.info(f"resume train from {self.ini_epoch}")

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
            callbacks=self.callbacks,
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
