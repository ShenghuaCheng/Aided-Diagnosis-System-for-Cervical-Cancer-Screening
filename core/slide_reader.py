# -*- coding: utf-8 -*-
"""
This file is a part of project "Aided-Diagnosis-System-for-Cervical-Cancer-Screening".
See https://github.com/ShenghuaCheng/Aided-Diagnosis-System-for-Cervical-Cancer-Screening for more information.

File name: slide_reader.py
Description: Integrated API for reading WSI.
"""

import os
import openslide
import numpy as np
from loguru import logger
from collections import Iterable
from openslide import OpenSlide
from thirdparty.slide_api import Sdpc, Srp

__all__ = [
    "SlideReader",
    "mpp_transformer",
    "get_attrs"
]


class SlideReader:
    handle = None
    suffix = None
    attrs = None
    _path = None

    def __init__(self):
        pass

    @property
    def path(self):
        return self._path

    def open(self, path):
        if path == self._path or path is None:
            return
        else:
            self.close()
        try:
            self.suffix = os.path.splitext(path)[-1]
            if self.suffix == ".sdpc":
                self.handle = Sdpc()
                self.handle.open(path)
            elif self.suffix == ".srp":
                self.handle = Srp()
                self.handle.open(path)
            elif self.suffix == ".svs" or self.suffix == ".mrxs":
                self.handle = OpenSlide(path)
            else:
                raise ValueError("File type: {} is not supported.".format(self.suffix))
            self._path = path
        except Exception as e:
            logger.error(f'{e}\nNote: some system requires absolute path for wsi image.')
            self.close()

    def close(self):
        self._path = None
        if self.handle is None:
            return
        self.handle.close()
        self.handle = None

    def get_attrs(self):
        return get_attrs(self)

    def get_tile(self, location: tuple, size: tuple, level: int):
        """ get tile from slide
        :param location: (x, y) at level 0
        :param size:  (w, h) at level 0
        :param level: read in level
        :return:  RGB img array
        """
        if level == 0:
            return self.get_tile_for_level0(location, size)
        # ensure the attrs had been created
        if self.attrs:
            pass
        else:
            self.get_attrs()
        # main operations for reading tile
        tile_size = mpp_transformer(size, self.attrs["mpp"], self.attrs["mpp"] * (self.attrs["level_ratio"] ** level))
        if self.suffix in ['.sdpc', '.srp']:
            # parts exceed right and bottom is filled with (255, 255, 255)
            location = mpp_transformer(location,
                                       self.attrs["mpp"], self.attrs["mpp"] * (self.attrs["level_ratio"] ** level))
            tile = self.handle.getTile(level, location[0], location[1], tile_size[0], tile_size[1])
        elif self.suffix in ['.svs', '.mrxs']:
            # parts exceed right and bottom is filled with (0, 0, 0)
            tile = np.array(self.handle.read_region(location, level, tile_size).convert('RGB'))
        return tile

    def get_tile_for_level0(self, location: tuple, size: tuple):
        """ get tile from slide in level 0
        :param location: (x, y) at level 0
        :param size:  (w, h) at level 0
        :return:  RGB img array
        """
        # main operations for reading tile
        if self.suffix in ['.sdpc', '.srp']:
            # parts exceed right and bottom is filled with (255, 255, 255)
            tile = self.handle.getTile(0, location[0], location[1], size[0], size[1])
        elif self.suffix in ['.svs', '.mrxs']:
            # parts exceed right and bottom is filled with (0, 0, 0)
            tile = np.array(self.handle.read_region(location, 0, size).convert('RGB'))
        return tile

    def __del__(self):
        self.close()


# == Aux Functions ==
def get_attrs(wsi_handle):
    attrs = {}
    try:
        if wsi_handle.suffix in [".sdpc", ".srp"]:
            attrs = wsi_handle.handle.getAttrs()
            attrs["bound_init"] = (0, 0)
            attrs["level_ratio"] = 2
        elif wsi_handle.suffix in ['.svs', '.mrxs']:
            attrs = get_openslide_attrs(wsi_handle.handle)
    except Exception as e:
        logger.error(e)

    return attrs


def mpp_transformer(origin, origin_mpp, aim_mpp):
    """transform numbers according to mpp
    :param origin: original numbers
    :param origin_mpp:  original mpp
    :param aim_mpp: aim mpp
    :return: transed numbers
    """
    if isinstance(origin, Iterable):
        transed = []
        for num in origin:
            transed.append(int(np.around(num * origin_mpp / aim_mpp)))
    else:
        transed = int(np.around(origin * origin_mpp / aim_mpp))
    return transed


def get_openslide_attrs(handle):
    if ".mrxs" in handle._filename:
        attrs = {
            "mpp": float(handle.properties[openslide.PROPERTY_NAME_MPP_X]),
            "level": handle.level_count,
            "width": int(handle.properties[openslide.PROPERTY_NAME_BOUNDS_WIDTH]),
            "height": int(handle.properties[openslide.PROPERTY_NAME_BOUNDS_HEIGHT]),
            "bound_init": (int(handle.properties[openslide.PROPERTY_NAME_BOUNDS_X]),
                           int(handle.properties[openslide.PROPERTY_NAME_BOUNDS_Y])),
            "level_ratio": int(handle.level_downsamples[1])
        }
    elif ".svs" in handle._filename:
        try:
            attrs = {
                "mpp": float(handle.properties[openslide.PROPERTY_NAME_MPP_X]),
                "level": handle.level_count,
                "width": int(handle.dimensions[0]),
                "height": int(handle.dimensions[1]),
                "bound_init": (0, 0),
                "level_ratio": int(handle.level_downsamples[1])
            }
        except KeyError:
            attrs = {
                "mpp": float(handle.properties['aperio.MPP'].strip(';')),
                "level": handle.level_count,
                "width": int(handle.dimensions[0]),
                "height": int(handle.dimensions[1]),
                "bound_init": (0, 0),
                "level_ratio": int(handle.level_downsamples[1])
            }
    return attrs


if __name__ == '__main__':
    dir = 'path to slide (.mrxs .svs .sdpc .srp)'
    handle = SlideReader()
    handle.open(dir)

    img = handle.get_tile((8000, 8000), (500, 500), 1)
    print(handle.attrs)
    handle.close()
