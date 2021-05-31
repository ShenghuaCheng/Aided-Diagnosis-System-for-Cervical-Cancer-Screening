import os
import openslide
import numpy as np
from collections import Iterable
from openslide import OpenSlide
from .slidereaders import Sdpc, Srp


class SlideReader:
    handle = None
    suffix = None
    attrs = None

    def __init__(self):
        pass

    def open(self, path):
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

    def close(self):
        self.handle.close()

    def get_attrs(self):
        if self.suffix in [".sdpc", ".srp"]:
            self.attrs = self.handle.getAttrs()
            self.attrs["bound_init"] = (0, 0)
            self.attrs["level_ratio"] = 2
        elif self.suffix in ['.svs', '.mrxs']:
            self.attrs = get_openslide_attrs(self.handle)

        return self.attrs

    def get_tile(self, location: tuple, size: tuple, level: int):
        """ get tile from slide
        :param location: (x, y) at level 0
        :param size:  (w, h) at level 0
        :param level: read in level
        :return:  RGB img array
        """
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