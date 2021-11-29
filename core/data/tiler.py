# -*- coding: utf-8 -*-
"""
This file is a part of project "Aided-Diagnosis-System-for-Cervical-Cancer-Screening".
See https://github.com/ShenghuaCheng/Aided-Diagnosis-System-for-Cervical-Cancer-Screening for more information.

File name: tiler
Description: tiling wsi.
"""

import math

import cv2
import numpy as np
from keras.utils import Sequence

from .slide_reader import SlideReader
from ..data.augmentations import StylisticTrans

__all__ = [
    "Tiler",
    "TileReader"
]


class Tiler:

    def __init__(
            self,
            wsi_path,
            gamma=False
    ):
        # create slide handle
        self._handle = SlideReader()
        self._handle.open(wsi_path)
        # get attrs
        self._slide_attrs = self._handle.get_attrs()  # this will cause a SIGABRT in ubuntu
        self.mpp = self._slide_attrs["mpp"]
        self.width = self._slide_attrs["width"]
        self.height = self._slide_attrs["height"]
        self.bound_init = self._slide_attrs["bound_init"]
        self.level = self._slide_attrs["level"]
        self.level_ratio = self._slide_attrs["level_ratio"]  # FIXME: slide in some srp having wrong level ratio
        # tile setting
        self.tiles = []
        self.read_level = 0
        self.gamma = gamma
        # multiprocess
        self._lock = None

    def tiling_set_to(self, tiles, dst_size, dst_mpp, read_level=0):
        # set tiles directly
        self.tiles = tiles
        self.read_level = read_level

        self.dst_size = dst_size
        self.dst_mpp = dst_mpp

    def set_multiprocess_lock(self, lock):
        self._lock = lock

    def tiling_according_to(self, dst_size, dst_mpp, overlap=0., exclude_border=True):
        """Tiling WSI according to params.
        """
        if not isinstance(dst_size, np.ndarray):
            dst_size = np.array(dst_size)
        dst_overlap_size = dst_size * overlap
        self.dst_size = dst_size
        self.dst_mpp = dst_mpp
        # area range
        content_begin = np.array([0, 0])
        if exclude_border:
            content_begin += self.bound_init
        content_end = np.array([self.width, self.height])
        # operate on src level0
        src_size = (dst_size * dst_mpp / self.mpp).astype(int)
        src_overlap_size = (dst_overlap_size * dst_mpp / self.mpp).astype(int)
        # tiling on level 0
        nb_col, nb_row = np.ceil(
            (content_end - content_begin - src_size) / (src_size - src_overlap_size) + 1
        ).astype(int).tolist()
        # get all initial points
        init_points = np.stack(np.meshgrid(range(nb_col), range(nb_row)), axis=-1)
        init_points = init_points * (src_size - src_overlap_size)
        init_points[:, -1, 0] = content_end[0] - src_size[0]  # y of final col
        init_points[-1, :, 1] = content_end[1] - src_size[1]  # x of final row
        # get all tile bbox
        box_sizes = np.ones_like(init_points) * src_size
        tiles = np.concatenate((init_points, box_sizes), axis=-1)
        # get the most close mpp and the read level
        read_level = int(math.log(dst_mpp/self.mpp, self.level_ratio))
        # TODO: filter tiles having no content according to binary mask to accelerate computing
        # done
        self.tiles = tiles.reshape((-1, 4))
        self.read_level = read_level

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            raise ValueError("slicing is not support.")
        x, y, w, h = self.tiles[idx].tolist()
        if self._lock is None:
            tile = self._handle.get_tile((x, y), (w, h), self.read_level)
        else:
            self._lock.acquire()
            tile = self._handle.get_tile((x, y), (w, h), self.read_level)
            self._lock.release()
        return tile  # RGB

    def __len__(self):
        return len(self.tiles)

    def __del__(self):
        self._handle.close()

    def __repr__(self):
        repr_str = f"Require mpp: {self.dst_mpp} size: {self.dst_size.tolist()}\n" + \
                   f"Read at level {self.read_level}\n" \
                   f"Tile mpp: {self.mpp * self.level_ratio ** self.read_level} " + \
                   f"size: {((self.dst_size * self.dst_mpp / self.mpp).astype(int) / self.level_ratio ** self.read_level).astype(int).tolist()}\n" + \
                   f"Total tiles: {len(self)}"
        return repr_str


class TileReader(Sequence):

    def __init__(
            self,
            tiler,
    ):
        self.tiler = tiler
        self.batch_size = 8
        # dst info
        self.dst_size = tiler.dst_size
        self._gamma = tiler.gamma

    def __getitem__(self, batch_idx):
        """ output rgb, norm to [-1, 1]
        """
        batch_images = []
        for idx in range(batch_idx * self.batch_size, (batch_idx + 1) * self.batch_size):
            if idx >= len(self.tiler):
                break
            tile = self.tiler[idx]
            batch_images.append(self._preprocess(tile))
        return np.stack(batch_images)

    def _preprocess(self, image):
        image = image[..., ::-1]  # rgb 2 bgr
        if self._gamma:
            image = StylisticTrans.gamma_adjust(image, 0.6)
        # resize and normalize
        image = cv2.resize(image, tuple(self.dst_size.tolist()))
        image = StylisticTrans.normalization(image, [-1, 1])
        return image[..., ::-1]

    def __len__(self):
        return math.ceil(len(self.tiler)/self.batch_size)
