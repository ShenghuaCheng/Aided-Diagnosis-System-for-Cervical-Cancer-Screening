# -*- coding: utf-8 -*-
import numpy as np
import multiprocessing.dummy as multiprocessing
import cv2
import time
from multiprocessing import Manager
from skimage import exposure

from lib.sdpcreader.sdpc_reader import sdpc
from lib.srpreader.srp_python_win import pysrp
from openslide import OpenSlide

def szsq_color_Normal(img):
    img_szsq_trans = exposure.adjust_gamma(img,0.6)#这里img一定是bgr
    return img_szsq_trans 

'''sizepatch为(w,h),input_str中的read_size_model1_resolution也是以(w,h)表示'''
def GetImg_whole(startpointlist, k, ors, level, sizepatch, pathsave = None):
    img = ors.read_region(startpointlist[k], level, sizepatch)
    img = np.array(img)#RGBA
#    print("startpointlist:"+str(startpointlist[k])+"_"+str(sizepatch)+"imgshape: "+str(img.shape))
    if str(img.shape)=="()":
        print("out of Memory,img is empty")
    else:
        img = img[:, :, 0 : 3]
        print(k)
        if pathsave!=None:
            cv2.imwrite(pathsave + str(k)+".tif",img[...,::-1])
    return img


def GetImg_whole_sdpc(startpointlist, k, ors, level, sizepatch, lock,pathsave=None):
    """ 返回RGB通道 sizepatch大小图片
    """
    lock.acquire()
    if k==None:
        img = ors.getTile(level, startpointlist[1], startpointlist[0], int(sizepatch[0]), int(sizepatch[1]))  
    else:
        img = ors.getTile(level, startpointlist[k][1], startpointlist[k][0], int(sizepatch[0]), int(sizepatch[1]))
    lock.release()

    img = np.ctypeslib.as_array(img)
    img.dtype = np.uint8
    img = img.reshape((int(sizepatch[1]), int(sizepatch[0]), 3))
    img = img[...,::-1]#rgb通道
#  下面lsb那边进行颜色矫正的代码，由于制作样本时需要用到该函数，所以这里把颜色矫正的代码放到了外面预测上进行处理    
#    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#    img = szsq_color_Normal(img)
#    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if pathsave!=None:        
        cv2.imwrite(pathsave + str(k)+".tif", img[...,::-1])
    
    return img

def GetImg_whole_srp(startpointlist, k, ors, level, sizepatch, lock,pathsave=None):
    """ 返回bgr通道 sizepatch大小图片
    """
    lock.acquire()
    if k==None:
        img = ors.ReadRegionRGB(level, startpointlist[0], startpointlist[1], int(sizepatch[0]), int(sizepatch[1]))  
    else:
        img = ors.ReadRegionRGB(level, startpointlist[k][0], startpointlist[k][1], int(sizepatch[0]), int(sizepatch[1]))
    lock.release()

    img = np.ctypeslib.as_array(img)
    img.dtype = np.uint8
    img = img.reshape((int(sizepatch[1]), int(sizepatch[0]), 3))
#    img = img[...,::-1]#rgb通道


#  下面lsb那边进行颜色矫正的代码，由于制作样本时需要用到该函数，所以这里把颜色矫正的代码放到了外面预测上进行处理    
#    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#    img = szsq_color_Normal(img)
#    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if pathsave!=None:        
        cv2.imwrite(pathsave + str(k)+"_"+str(startpointlist[0])+".tif", img)
#    temp=(startpointlist[k][1],startpointlist[k][0])
    return img

class GetImg(object):
    
    def __init__(self, pathfile, startpointlist, level, sizepatch_read, pathsave=None):
        self.pathfile=pathfile
        self.startpointlist=startpointlist
        self.level=level
        self.sizepatch_read=sizepatch_read
        self.pathsave=pathsave
        
    def Get_predictimgMultiprocess(self):
        """ 添加参数lock，保证多线程读图
        """
        pathfile=self.pathfile
        startpointlist=self.startpointlist
        level=self.level
        sizepatch_read=self.sizepatch_read
        pathsave=self.pathsave
        
        filetype = '.'+pathfile.split('.')[-1]
        if filetype == '.sdpc':
            ors = sdpc.Sdpc()
            ors.open(pathfile)
        elif filetype == '.srp':
            ors = pysrp.Srp()
            ors.open(pathfile)
        else:
            ors = OpenSlide(pathfile)
        start = time.time()
        pool = multiprocessing.Pool(16)
        imgTotals = []
        print(pathfile)
    #在不同程序间如果有同时对同一个队列操作的时候，
    #为了避免错误，可以在某个函数操作队列的时候给它加把锁，这样在同一个时间内则只能有一个子进程对队列进行操作，锁也要在manager对象中的锁
        manager = Manager()
        lock = manager.Lock()

        for k in range(len(startpointlist)):
            print(k)
            if filetype == '.sdpc':
                imgTotal = pool.apply_async(GetImg_whole_sdpc, args=(startpointlist, k, ors, level, sizepatch_read, lock, pathsave))
            elif filetype == '.srp':
                imgTotal= pool.apply_async(GetImg_whole_srp, args=(startpointlist, k, ors, level, sizepatch_read, lock, pathsave))
            else:
                imgTotal = pool.apply_async(GetImg_whole, args=(startpointlist, k, ors, level, sizepatch_read, pathsave))
            imgTotals.append(imgTotal)
            
        pool.close()
        pool.join()
        imgTotals = np.array([x.get() for x in imgTotals])
        end = time.time()
        print('多线程读图耗时{}\n{}张\nlevel = {}读入大小{}'.format(end-start,len(startpointlist),level,sizepatch_read))
        return imgTotals

