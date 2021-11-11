# -*- coding: utf-8 -*-
"""
This file is a part of project "Aided-Diagnosis-System-for-Cervical-Cancer-Screening".
See https://github.com/ShenghuaCheng/Aided-Diagnosis-System-for-Cervical-Cancer-Screening for more information.

File name: train.py
Description: train model
"""

import argparse

from core.config import load_config
from core.drive import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Train parser")
    parser.add_argument("-n", "--config_name", type=str, default=None, help="train config name")
    parser.add_argument("-f", "--config_file", type=str, default=None, help="train config file")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("-d", "--devices", type=int, default=None, help="devices used for training")
    parser.add_argument("-w", "--weights", default=None, type=str, nargs=argparse.ONE_OR_MORE, help="checkpoint file")
    parser.add_argument("-e", "--start_epoch", type=int, default=None, help="resume training start epoch")
    args = parser.parse_args()
    config = load_config(args.config_file, args.config_name)

    trainer = Trainer(args, config)
    trainer.train()
