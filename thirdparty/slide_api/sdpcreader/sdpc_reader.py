import ctypes
import os
import platform

import numpy as np

SYSTEM_TYPE = platform.system()


class SdpcInfo(ctypes.Structure):
    _fields_ = [("mpp", ctypes.c_double),
                ("level", ctypes.c_int32),

                ("width", ctypes.c_int32),
                ("height", ctypes.c_int32),

                ("tileWidth", ctypes.c_int32),
                ("tileHeight", ctypes.c_int32),

                ("L0_nx", ctypes.c_int32),
                ("L0_ny", ctypes.c_int32)
                ]


class Sdpc(object):
    def __init__(self):
        self.__hand = 0

        if SYSTEM_TYPE == "Windows":
            dll_path = "libSdpcSDK.dll"
        elif SYSTEM_TYPE == "Linux":
            dll_path = "libSdpcSdk.so"

        current_file_path = os.path.abspath(__file__)
        dll_path = os.path.join(os.path.split(current_file_path)[0], dll_path)

        self.__dll = ctypes.cdll.LoadLibrary(dll_path)

        self.__dll.openSdpc.argtypes = [ctypes.c_char_p]
        self.__dll.openSdpc.restype = ctypes.c_ulonglong
        self.__dll.closeSdpc.argtypes = [ctypes.c_ulonglong]
        self.__dll.getSdpcInfo.argtypes = [ctypes.c_ulonglong, ctypes.c_void_p]
        self.__dll.getTile.argtypes = [ctypes.c_ulonglong,  #
                                       ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,  #
                                       ctypes.c_int32, ctypes.c_int32,  #
                                       ctypes.c_char_p]

    def say_hello(self):
        print("hello")

    def open(self, path):
        pStr = ctypes.c_char_p()
        pStr.value = bytes(path, 'utf-8')
        self.__hand = self.__dll.openSdpc(pStr)
        pass

    def close(self):
        if self.__hand != 0:
            self.__dll.closeSdpc(self.__hand)
            self.__hand = 0

    def getAttrs(self):
        info = SdpcInfo()
        ret = self.__dll.getSdpcInfo(self.__hand, ctypes.byref(info))
        if ret == 1:
            attrs = {"mpp": info.mpp,  #
                     "level": info.level,  #
                     "width": info.width,  #
                     "height": info.height  #
                    }
            return attrs
        else:
            return {}

    def getTile(self, z, x, y, w, h):
        """返回 RGB 图像
        :param z: 图像缩放倍率，0 为不缩放
        :param x: 直角坐标系纵轴坐标，原点在左上角，向下为正方向
        :param y: 直角坐标系横轴坐标，原点在左上角，向右为正方向
        :param w: 图片宽
        :param h: 图片高
        :return: 返回 形状为(h, w, c)的numpy数组，数值类型 为np.uint8
        """
        img = ctypes.create_string_buffer(w * h * 3)
        ret = self.__dll.getTile(self.__hand, z, y, x, w, h, img)

        if ret == 1:
            img = np.ctypeslib.as_array(img)
            img.dtype = np.uint8
            img = img.reshape((h, w, 3))
            return img[..., ::-1]
        else:
            return None



