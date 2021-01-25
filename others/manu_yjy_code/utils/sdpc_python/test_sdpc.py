import numpy as np
from PIL import Image
from utils.sdpc_python import sdpc

sdpc = sdpc.Sdpc()

sdpc.open(r'H:\TCTDATA\SZSQ_originaldata\Shengfuyou_3th\positive\Shengfuyou_3th_positive_40X\1161816 0893184.sdpc')

attrs = sdpc.getAttrs()
print(attrs)

img = sdpc.getTile(0, 100, 200, 1024, 1024)
if img is not None:
    nimg = np.ctypeslib.as_array(img)
    nimg.dtype = np.uint8
    nimg = nimg.reshape((1024, 1024, 3))
    im = Image.fromarray(np.uint8(nimg))
    im.show()
    pass

sdpc.close()
