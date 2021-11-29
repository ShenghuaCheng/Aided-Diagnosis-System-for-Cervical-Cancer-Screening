# -*- coding: utf-8 -*-
"""
This file is a part of project "Aided-Diagnosis-System-for-Cervical-Cancer-Screening".
See https://github.com/ShenghuaCheng/Aided-Diagnosis-System-for-Cervical-Cancer-Screening for more information.

File name: rnn_dataloader
Description: keras.utils.Sequence data loader for rnn related task
"""

import math
import os
import random
import cv2
import numpy as np
from glob2 import glob
from loguru import logger
from keras.utils import Sequence

from .rnn_dataset import RnnDataset
from ..preprocess import Preprocess

__all__ = [
    "RnnDataloader",
]

FEATURE_LENGTH = 2048


class TopImagesLoader(Sequence):
    def __init__(
            self,
            wsi_pool,
            wsi_mpp,
            output_mpp,
            top_n,
            preprocessor
    ):
        self.wsi_pool = wsi_pool
        self.wsi_mpp = wsi_mpp
        self.top_n = top_n

        self.batch_size = 8
        self.output_mpp = output_mpp
        self.preprocessor = preprocessor

        self._merge_top_images()

    def __len__(self):
        return math.ceil(len(self.merged_images)/self.batch_size)

    def __getitem__(self, batch_idx):
        image_paths = self.merged_images[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
        images = []
        for img_p in image_paths:
            images.append(self._preprocess(img_p))
        return np.array(images)

    def _preprocess(self, image_path):
        image = cv2.imread(image_path)
        scale_ratio = self.wsi_mpp / self.output_mpp
        image = cv2.resize(image, None, fx=scale_ratio, fy=scale_ratio, interpolation=cv2.INTER_LINEAR)
        image, _ = self.preprocessor(image)
        return image

    def _merge_top_images(self):
        self.merged_images = []
        for top_dir in self.wsi_pool:
            # ordered reading
            for idx in range(self.top_n):
                self.merged_images.append(glob(os.path.join(top_dir, "{:0>3d}_[01].*.[jpt][pni][gf]".format(idx)))[0])

    def __repr__(self):
        return f"WSI num: {len(self.wsi_pool)}, Encode Top: {self.top_n}, Total Images: {len(self.merged_images)}"


class RnnDataloader(Sequence):
    def __init__(
            self,
            dataset: RnnDataset,
            name: str,
            encoder,
            top_n,
            top_rng,
            input_mpp,
            preprocessor: Preprocess,
            permutation,
            interval_image_re_encode=50,
    ):
        self.encoder = encoder
        # dataset setting
        self.dataset = dataset
        self.name = name
        self.dataset_cfg, self.wsi_pools = dataset.get_set(name)
        self.batch_size = 8
        # data setting
        self.top_n = top_n
        self.top_rng = top_rng
        self.interval_image_re_encode = interval_image_re_encode
        self.input_mpp = input_mpp
        # create preprocessor
        self.preprocessor = preprocessor
        self.permutation = permutation
        # init encode
        self.feature_pools = self.encode_whole_set()
        self._sampling()
        # epoch counter
        self._epoch_counter = 0

    def __len__(self):
        return math.ceil(len(self.features) / self.batch_size)

    def __getitem__(self, batch_idx):
        ls_sample_idx = self.shuffled_index[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
        batch_x = []
        batch_y = []
        for sample_idx in ls_sample_idx:
            x, y = self._preprocess(sample_idx)
            batch_x.append(x)
            batch_y.append(y)
        if self.name is "test":
            # for "test" in evaluating which is driven by predict_generator, only data
            return np.array(batch_x)
        else:
            # for "train" and "val", the labels is used for calculating loss
            return np.array(batch_x), np.array(batch_y)

    def _preprocess(self, sample_idx):
        feature = self.features[sample_idx]
        label = self.labels[sample_idx]
        if random.random() < self.permutation:
            feature = np.random.permutation(feature)
        return feature[:self.top_n], label

    def _sampling(self):
        self.labels = []
        self.features = []
        self.grp_names = []
        sampled_idx_pools = self.dataset.sampling(self.name)
        for i in range(len(sampled_idx_pools)):
            label = int(self.dataset_cfg["subset_label"][i])
            features = self.feature_pools[i][sampled_idx_pools[i]]
            grp_name = self.dataset_cfg["group_name"][i] + '_' + self.dataset_cfg["subset_name"][i]

            self.labels += [label] * len(features)
            self.features.append(features)
            self.grp_names += [grp_name] * len(features)
        self.features = np.concatenate(self.features, axis=0)
        self.shuffled_index = list(range(len(self.features)))
        if self.name is not "test":
            # "test: is for evaluator, shuffle is not necessary.
            random.shuffle(self.shuffled_index)

    def on_epoch_end(self):
        self._epoch_counter += 1
        print()
        self._sampling()
        logger.info(f"Resampling {self.name} from image pool ...")
        if self._epoch_counter % self.interval_image_re_encode == 0:
            logger.info(f"do period encoding ...")
            self.feature_pools = self.encode_whole_set()

    def encode_whole_set(self):
        feature_pools = []
        total_top = self.top_n + self.top_rng
        for set_idx in range(len(self.wsi_pools)):
            wsi_pool = self.wsi_pools[set_idx]
            wsi_mpp = float(self.dataset_cfg["subset_mpp"][set_idx])
            grp_name = self.dataset_cfg["group_name"][set_idx] + '_' + self.dataset_cfg["subset_name"][set_idx]
            top_loader = TopImagesLoader(
                wsi_pool, wsi_mpp, self.input_mpp, total_top, self.preprocessor
            )
            logger.info(f"encoding Group: {grp_name}, {top_loader}")
            res = self.encoder.predict_generator(top_loader, verbose=1)
            feature_pool = res[1].reshape((-1, total_top, FEATURE_LENGTH))
            feature_pools.append(feature_pool)
        logger.info("finish encoding")
        return feature_pools

    def get_cur_labels(self):
        if self.name is "test":
            return self.labels, np.array(self.wsi_pools).flatten().tolist(), [None]*len(self.labels), self.grp_names
        else:
            raise ValueError(f"This API is used for 'test' set only. Order will be confusing in {self.name}.")

