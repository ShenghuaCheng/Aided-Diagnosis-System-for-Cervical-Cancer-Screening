# -*- coding: utf-8 -*-
"""
This file is a part of project "Aided-Diagnosis-System-for-Cervical-Cancer-Screening".
See https://github.com/ShenghuaCheng/Aided-Diagnosis-System-for-Cervical-Cancer-Screening for more information.

File name: resnet_dataloader
Description: keras.utils.Sequence data loader for resnet related task
"""

import math
import random
import cv2
import numpy as np
from loguru import logger
from keras.utils import Sequence, to_categorical

from .resnet_dataset import ResnetDataset
from ..preprocess import Preprocess

__all__ = [
    "ResnetDataloader",
]


class ResnetDataloader(Sequence):
    def __init__(
            self,
            dataset: ResnetDataset,
            name: str,
            input_size,
            input_mpp,
            preprocessor: Preprocess
    ):
        # dataset setting
        self.dataset = dataset
        self.name = name
        self.with_mask = dataset.with_mask
        self._sampling()
        self.batch_size = 8
        # data setting
        self.input_size = input_size
        self.input_mpp = input_mpp
        # create preprocessor
        self.preprocessor = preprocessor

    def __len__(self):
        return math.ceil(len(self.images) / self.batch_size)

    def __getitem__(self, batch_idx):
        ls_sample_idx = self.shuffled_index[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
        batch_x = []
        batch_y = []
        for sample_idx in ls_sample_idx:
            x, y = self._preprocess(sample_idx)
            batch_x.append(x)
            batch_y.append(y)
        return np.array(batch_x), np.array(batch_y)

    def _preprocess(self, sample_idx):
        label = self.labels[sample_idx]
        image = cv2.imread(self.images[sample_idx])
        mpp = self.mpp_ls[sample_idx]

        scale_ratio = mpp / self.input_mpp
        image = cv2.resize(image, None, fx=scale_ratio, fy=scale_ratio, interpolation=cv2.INTER_LINEAR)

        if self.with_mask and label:
            # if mask and label=1, attach mask to image.
            mask_path = self.masks[sample_idx]
            mask = cv2.imread(mask_path)
            mask = cv2.resize(mask, None, fx=scale_ratio, fy=scale_ratio, interpolation=cv2.INTER_LINEAR)
            image = np.append(image, mask[..., 0:1], axis=-1)

        image, mask = self.preprocessor(image)

        if self.with_mask:
            if mask is None:
                # for label 0
                mask = np.zeros((16, 16), dtype=np.uint8)
            else:
                # for label 1
                mask = cv2.resize((mask / 255).astype(np.uint8), (16, 16))
            label = to_categorical(mask, 2)
        return image, label

    def _sampling(self):
        self.labels, self.images, self.mpp_ls, self.masks = self.dataset.sampling(self.name)
        self.shuffled_index = list(range(len(self.images)))
        random.shuffle(self.shuffled_index)

    def on_epoch_end(self):
        print()
        self._sampling()
        logger.info(f"Resampling {self.name} from image pool ...")
