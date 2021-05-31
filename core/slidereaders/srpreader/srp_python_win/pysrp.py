import os
import ctypes
import numpy as np

current_file_path = os.path.abspath(__file__)
os.chdir(os.path.split(current_file_path)[0])


class Srp(object):
    def __init__(self):
        self.__hand = 0
        dll_name = "srp.dll"
        current_file_path = os.path.abspath(__file__)
        dll_path = os.path.join(os.path.split(current_file_path)[0], dll_name)
        # print(dll_path)
        self.__dll = ctypes.cdll.LoadLibrary(dll_path)
        self.__dll.OpenRW.argtypes = [ctypes.c_char_p]
        self.__dll.OpenRW.restype = ctypes.c_ulonglong
        self.__dll.Close.argtypes = [ctypes.c_ulonglong]
        self.__dll.ReadRegionRGB.argtypes = [ctypes.c_ulonglong,  #
                                       ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,  #
                                       ctypes.c_int32, ctypes.c_int32,  #
                                       ctypes.c_char_p, ctypes.POINTER(ctypes.c_int32)]
        self.__dll.ReadParamInt32.argtypes = [ctypes.c_ulonglong, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int32)]
        self.__dll.ReadParamInt64.argtypes = [ctypes.c_ulonglong, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int64)]
        self.__dll.ReadParamFloat.argtypes = [ctypes.c_ulonglong, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float)]
        self.__dll.ReadParamDouble.argtypes = [ctypes.c_ulonglong, ctypes.c_char_p, ctypes.POINTER(ctypes.c_double)]
        self.__dll.WriteParamDouble.argtypes = [ctypes.c_ulonglong, ctypes.c_char_p, ctypes.c_double]
        self.__dll.WriteOneAnno.argtypes = [ctypes.c_ulonglong, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double]
        self.__dll.CleanAnno.argtypes = [ctypes.c_ulonglong]

    def say_hello(self):
        self.__dll.hello()

    def open(self, path):
        pStr = ctypes.c_char_p()
        pStr.value = bytes(path, 'utf-8')
        self.__hand = self.__dll.OpenRW(pStr)
        pass

    def close(self):
        if self.__hand != 0:
            self.__dll.Close(self.__hand)
            self.__hand = 0

    def getAttrs(self):
        pwKey = ctypes.c_char_p()
        pwKey.value = bytes("width", 'utf-8')
        pw = ctypes.c_int32(0)
        b0 = self.__dll.ReadParamInt32(self.__hand, pwKey, ctypes.byref(pw))
        phKey = ctypes.c_char_p()
        phKey.value = bytes("height", 'utf-8')
        ph = ctypes.c_int32(0)
        b1 = self.__dll.ReadParamInt32(self.__hand, phKey, ctypes.byref(ph))
        pzKey = ctypes.c_char_p()
        pzKey.value = bytes("level", 'utf-8')
        plevel = ctypes.c_int32(0)
        b2 = self.__dll.ReadParamInt32(self.__hand, pzKey, ctypes.byref(plevel))
        ppKey = ctypes.c_char_p()
        ppKey.value = bytes("mpp", 'utf-8')
        pmpp = ctypes.c_double(0)
        b3 = self.__dll.ReadParamDouble(self.__hand, ppKey, ctypes.byref(pmpp))
        if b0 and b1 and b2 and b3:
            attrs = {"mpp": pmpp.value,  #
                     "level": plevel.value,  #
                     "width": pw.value,  #
                     "height": ph.value  #
                    }
            return attrs
        else:
            return {}

    def ReadRegionRGB(self, level, x, y, width, height):
        buf_len = width*height*3
        plen = ctypes.c_int32(buf_len)
        img = ctypes.create_string_buffer(buf_len)
        ret = self.__dll.ReadRegionRGB(self.__hand, level, x, y, width, height, img, ctypes.byref(plen))
        if ret != 0:
            return img
        else:
            return None

    def WriteScore(self, score):
        keyStr = ctypes.c_char_p()
        keyStr.value = bytes("score", 'utf-8')
        return self.__dll.WriteParamDouble(self.__hand, keyStr, score)

    def WriteAnno(self, x, y, score):
        return self.__dll.WriteOneAnno(self.__hand, x, y, 0, score);

    def CleanAnno(self):
        return self.__dll.CleanAnno(self.__hand);

    def getTile(self, z, x, y, w, h):
        """返回 RGB 图像
        :param z: 图像缩放倍率，0 为不缩放
        :param x: 直角坐标系纵轴坐标，原点在左上角，向下为正方向
        :param y: 直角坐标系横轴坐标，原点在左上角，向右为正方向
        :param w: 图片宽
        :param h: 图片高
        :return: 返回 形状为(h, w, c)的numpy数组，数值类型 为np.uint8
        """
        img = self.ReadRegionRGB(z, x, y, w, h)
        if img is not None:
            img = np.ctypeslib.as_array(img)
            img.dtype = np.uint8
            img = img.reshape((h, w, 3))
            return img[...,::-1]
        else:
            return None

