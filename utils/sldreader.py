# -*- coding:utf-8 -*-
'''
Slide reader class for mrxs svs sdpc srp format.
Default pixel size lists below:
------------------
mrxs | 0.243 | 20x
svs  | 0.293 | 20x
sdpc | 0.180 | 40x
srp  | 0.235 | 20x
------------------
* For some svs transformed from szsq sdpc data, the pixel size is 0.180
'''
import os
import platform
import numpy as np
import matplotlib.pyplot as plt
import openslide
from openslide import OpenSlide
from lib.sdpcreader.sdpc_reader import Sdpc

SYSTEM_TYPE = platform.system()
if SYSTEM_TYPE == "Windows":
    from lib.srpreader.srp_python_win.pysrp import Srp
elif SYSTEM_TYPE == "Linux":
    from lib.srpreader.srp_python_linux.pysrp import Srp


class SlideReader:
    _LEVEL = 0
    _PixelSizeDict = {
        'mrxs': 0.243,
        'svs': 0.293,
        'sdpc': 0.180,
        'srp': 0.235747,
    }

    def __init__(self, slide_dir, transSvs=False):
        """ 切片的读图接口整合
        :param slide_dir: 切片的绝对路径
        :param transSvs: 是否为sdpc转svs的数据，如果是的svs对应pixelsize将更新
        """
        if transSvs:
            self._PixelSizeDict['svs'] = 0.180

        self.slide_dir = slide_dir

        self._slide_name = os.path.split(self.slide_dir)[-1]
        self._file_type = self._slide_name.split('.')[-1]

        self.pixel_size = self._PixelSizeDict[self._file_type]

        self._create_reader_obj()

    def _create_reader_obj(self):
        if self._file_type == 'sdpc':
            self.reader = Sdpc()
            self.reader.open(self.slide_dir)
        elif self._file_type == 'srp':
            self.reader = Srp()
            self.reader.open(self.slide_dir)
        elif self._file_type in ['svs', 'mrxs']:
            self.reader = OpenSlide(self.slide_dir)
        else:
            raise ValueError('Invalid file type: {}' .format(self._file_type))

    def get_attr(self):
        """获取并返回全切片宽高信息(w, h)
        :return:
        """
        if self._file_type in ['sdpc', 'srp']:
            attr = self.reader.getAttrs()
            return (attr['width'], attr['height'])
        elif self._file_type in ['svs', 'mrxs']:
            return self.reader.level_dimensions[self._LEVEL]

    def get_foreground_coor(self):
        if self._file_type == 'mrxs':
            foreground_x = int(self.reader.properties[openslide.PROPERTY_NAME_BOUNDS_X])
            foreground_y = int(self.reader.properties[openslide.PROPERTY_NAME_BOUNDS_Y])
            return [foreground_x, foreground_y]
        else:
            raise TypeError("file type: %s has no such func" %self._file_type)

    def get_tile(self, location, patch_size):
        """
        返回rgb图像, np.array(), np.uint8
        :param location: (x, y)
        :param patch_size: (w, h)
        :return: rgb numpy array
        """
        if self._file_type in ['sdpc', 'srp']:
            return self.reader.getTile(self._LEVEL, location[0], location[1], patch_size[0], patch_size[1])
        elif self._file_type in ['svs', 'mrxs']:
            return np.array(self.reader.read_region(location, self._LEVEL, patch_size).convert('RGB'), dtype=np.uint8)

    def get_tile_level(self, location, patch_size, level):
        """
        返回rgb图像, np.array(), np.uint8
        :param location: (x, y)
        :param patch_size: (w, h)
        :return: rgb numpy array
        """
        if self._file_type in ['sdpc', 'srp']:
            return self.reader.getTile(level, location[0], location[1], patch_size[0], patch_size[1])
        elif self._file_type in ['svs', 'mrxs']:
            return np.array(self.reader.read_region(location, level, patch_size).convert('RGB'), dtype=np.uint8)

    def __del__(self):
        self.reader.close()


if __name__ == '__main__':
    # test mrxs
    slide_dir = r'H:\TCTDATA\Shengfuyou_1th\Positive\718likehuiH13.mrxs'
    sr = SlideReader(slide_dir)

    # # test svs
    # slide_dir = r'H:\TCTDATA\SZSQ_originaldata\Tongji_3th\positive\tongji_3th_positive_40x\svs\042910101.svs'
    # sr = SlideReader(slide_dir, transSvs=True)

    # # test sdpc
    # slide_dir = r'H:\TCTDATA\SZSQ_originaldata\XiaoYuWei2\positive\1909060062.sdpc'
    # sr = SlideReader(slide_dir)

    # # test srp
    # slide_dir = r'H:\transSrp\SZSQ_originaldata\Shengfuyou_5th\positive\Shengfuyou_5th_positive_40X/1104099_0893172.srp'
    # sr = SlideReader(slide_dir)
    # print(sr.get_attr())

    img = sr.get_tile((54596, 13030), (600, 600))
    plt.imshow(img)
