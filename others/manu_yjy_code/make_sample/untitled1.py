import sys
sys.path.append('../')
import xml.etree.cElementTree as et
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import scipy.ndimage as ndi
from openslide import OpenSlide
from multiprocessing import Manager, Lock
from utils.sdpc_python import sdpc
from tqdm import tqdm
from utils.function_set import Get_predictimgMultiprocess
import multiprocessing.dummy as multiprocessing


pathfile = r'Z:\Zhang Ming\tongji5_sq\neg\tj19052281.sdpc'
ors = sdpc.Sdpc()
ors.open(pathfile)
# =============================================================================
# 
# manager = Manager()
# lock = manager.Lock()
# =============================================================================
startpointlist = [(20000, 20000),
                  (25000, 25000),
                  (30000, 30000),
                  (35000, 35000),
                  (40000, 40000),
                  (45000, 45000),
                  (60000, 60000)]
size = 833
for i in range(len(startpointlist)):
    print(i)
# =============================================================================
#     break
#     print(i)
#     lock.acquire()
# =============================================================================
    img = ors.getTile(0, startpointlist[i][0], startpointlist[i][1], size, size)
# =============================================================================
#     lock.release()
# =============================================================================
    img = np.ctypeslib.as_array(img)
    img.dtype = np.uint8
    img = img.reshape((size, size, 3))
    img = img[...,::-1]
    pathsave1 = r'F:\yujingya\test\new'
    cv2.imwrite(pathsave1 + '/{}_{}_{}.tif'.format('neg_tj19052281', i, startpointlist[i]), img[...,::-1])
     
