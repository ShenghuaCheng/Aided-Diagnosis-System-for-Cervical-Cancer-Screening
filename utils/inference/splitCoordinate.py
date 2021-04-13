# -*- coding: utf-8 -*-
"""
功能：
1、从整张切片开始对其进行拆分，计算拆分后每个块的左上角坐标
上面的拆分后坐标只有startpointlist为相对于整张切片的坐标，其他的都是相对坐标
2、将上述获得的坐标转变为相对于整张切片的坐标
"""
import numpy as np
import cv2
import time
from .data_process import EstimateRegion
from .parameterSet import Size_set
from lib.sdpcreader.sdpc_reader import sdpc
from lib.srpreader.srp_python_win import pysrp
from openslide import OpenSlide

# =============================================================================
# 将整张切片按照冗余为widthOverlap，拆分为sizepatch_read的图，保存的为左上角坐标，为绝对坐标
# =============================================================================
'''
例子：如果输入为our扫描仪:
level=0
levelratio=4
sizepatch_read=(w,h)=(1216*0.586/0.293,1936*0.586/0.293)=(2432,3872)
widthOverlap=(w,h)=(120*0.586/0.293,120*0.586/0.293)=(240,240)
如果为szsq扫描仪：
level=0
levelratio=4
sizepatch_read=(int(1216*0.586/0.18),int(1936*0.586/0.18))
widthOverlap=(int(120*0.586/0.18),int(120*0.586/0.18))
'''


def Get_startpointlist(pathfile, level, levelratio, sizepatch_read, widthOverlap, threColor = 8):
    sizepatch_y,sizepatch_x = sizepatch_read
    widthOverlap_y,widthOverlap_x = widthOverlap
    ratio = levelratio ** level
    filetype = '.'+pathfile.split('.')[-1]
    if filetype == '.mrxs':#因为3D扫的是整个玻片，除了中间的圆形区域，还有上下的背景区域
        # ors = OpenSlide(pathfile)
        imgmap = EstimateRegion(pathfile,threColor = threColor)#得到的是level5下的imgmap
        start_xmin = imgmap.nonzero()[0][0] * 32
        end_xmax = imgmap.nonzero()[0][-1] * 32
        start_ymin = np.min(imgmap.nonzero()[1]) * 32
        end_ymax = np.max(imgmap.nonzero()[1]) * 32
    else:
        if filetype == '.sdpc':  
            ors = sdpc.Sdpc()
            ors.open(pathfile)
            attr = ors.getAttrs()
            size = (attr['width'], attr['height'])
            print(size[0],size[1])
        elif filetype == '.svs':
            ors = OpenSlide(pathfile)
            size = ors.level_dimensions[0]
        elif filetype == '.srp':
            ors = pysrp.Srp()
            ors.open(pathfile)
            attr = ors.getAttrs()
            size = (attr['width'], attr['height'])
        start_xmin = 0
        end_xmax = size[1]
        start_ymin = 0
        end_ymax = size[0]
    #由于sizepatch为对应level的，而end_xmax，end_ymax为level0下的，所以要除以ratio换算
    numblock_x = np.ceil(((end_xmax-start_xmin)/ratio - sizepatch_x) / (sizepatch_x - widthOverlap_x)+1)
    numblock_y = np.ceil(((end_ymax-start_ymin)/ratio - sizepatch_y) / (sizepatch_y - widthOverlap_y)+1)
    # 得到所有开始点的坐标(排除白色起始点）
    startpointlist = []
    for j in range(int(numblock_x)):
        for i in range(int(numblock_y)):
            startpointtemp = (start_ymin + i*(sizepatch_y-widthOverlap_y)*ratio,start_xmin+j*(sizepatch_x-widthOverlap_x)*ratio)
            startpointlist.append(startpointtemp)
    return startpointlist
#上面的是如果宽度能够得到左上角坐标，即使加上块宽，超出边界也裁剪出来，但是之前sdpc的超出边界为黑色，但是srp超出边界为白色
#所以srp的超出边界的部分用宽或者高边界减去块宽，再取。
def Get_startpointlistSrp(pathfile, level, levelratio, sizepatch_read, widthOverlap, threColor = 8):
    sizepatch_y,sizepatch_x = sizepatch_read
    widthOverlap_y,widthOverlap_x = widthOverlap
    ratio = levelratio ** level
    filetype = '.'+pathfile.split('.')[-1]
    if filetype == '.mrxs':#因为3D扫的是整个玻片，除了中间的圆形区域，还有上下的背景区域
        # ors = OpenSlide(pathfile)
        imgmap = EstimateRegion(pathfile,threColor = threColor)#得到的是level5下的imgmap
        start_xmin = imgmap.nonzero()[0][0] * 32
        end_xmax = imgmap.nonzero()[0][-1] * 32
        start_ymin = np.min(imgmap.nonzero()[1]) * 32
        end_ymax = np.max(imgmap.nonzero()[1]) * 32
    else:
        if filetype == '.sdpc':  
            ors = sdpc.Sdpc()
            ors.open(pathfile)
            attr = ors.getAttrs()
            size = (attr['width'], attr['height'])
            print(size[0],size[1])
        elif filetype == '.svs':
            ors = OpenSlide(pathfile)
            size = ors.level_dimensions[0]
        elif filetype == '.srp':
            ors = pysrp.Srp()
            ors.open(pathfile)
            attr = ors.getAttrs()
            size = (attr['width'], attr['height'])
        start_xmin = 0
        end_xmax = size[1]
        start_ymin = 0
        end_ymax = size[0]
    #由于sizepatch为对应level的，而end_xmax，end_ymax为level0下的，所以要除以ratio换算
    numblock_x = np.ceil(((end_xmax-start_xmin)/ratio - sizepatch_x) / (sizepatch_x - widthOverlap_x)+1)
    numblock_y = np.ceil(((end_ymax-start_ymin)/ratio - sizepatch_y) / (sizepatch_y - widthOverlap_y)+1)
    # 得到所有开始点的坐标(排除白色起始点）
    startpointlist = []
    for j in range(int(numblock_x)):
        for i in range(int(numblock_y)):
            startpointtemp = (start_ymin + i*(sizepatch_y-widthOverlap_y)*ratio,start_xmin+j*(sizepatch_x-widthOverlap_x)*ratio)
            if startpointtemp[0]+sizepatch_y*ratio>size[0] or startpointtemp[1]+sizepatch_x*ratio>size[1]:
                startpointtemp=(min(startpointtemp[0],size[0]-sizepatch_y*ratio),min(startpointtemp[1],size[1]-sizepatch_x*ratio))
            startpointlist.append(startpointtemp)
    return startpointlist
# =============================================================================
# 将上面得到的一张大图在0.586分辨率下拆分为cutnum个model1个输入图像的左上角坐标，该坐标为相对该大图而言
# 假如整张切片拆分为了600个0.293分辨率下尺寸为(2432,3872)的大图，那么这600个大图在0.586下的Startpointlist_split坐标为相同的
# =============================================================================
'''
输入：
sizepatch_predict:为在0.586分辨率下上述大图的尺寸，如果为our扫描仪时，sizepatch_predict=(1216,1936)
sizepatch_predict_small:为在0.586分辨率下大图被拆分为cutnum个model1个输入图像的尺寸,sizepatch_predict_small=(512,512)
widthOverlap_predict：为0.586分辨率下大图拆分为cutnum个model1个输入图像时的冗余，widthOverlap_predict=(160,156)
输出：相对于0.586下大图，拆分为(512,512)的左上角坐标，以及cutnum
'''
def Startpointlist_split(sizepatch_predict,sizepatch_predict_small, widthOverlap_predict):
    sizepatch_predict_y,sizepatch_predict_x = sizepatch_predict
    sizepatch_predict_small_y,sizepatch_predict_small_x = sizepatch_predict_small
    widthOverlap_predict_y,widthOverlap_predict_x = widthOverlap_predict
       
    numblock_y = np.around(( sizepatch_predict_y - sizepatch_predict_small_y) / (sizepatch_predict_small_y - widthOverlap_predict_y)+1)
    numblock_x = np.around(( sizepatch_predict_x - sizepatch_predict_small_x) / (sizepatch_predict_small_x - widthOverlap_predict_x)+1)
    startpointlist = []
    for j in range(int(numblock_x)):
        for i in range(int(numblock_y)):
            startpointtemp = (i*(sizepatch_predict_small_y- widthOverlap_predict_y),j*(sizepatch_predict_small_x-widthOverlap_predict_x))
            startpointlist.append(startpointtemp)
    return int(numblock_y*numblock_x),startpointlist

# =============================================================================
# 根据0.586分辨率下大图的矩阵，获得所有大图对应的cutnum个小图矩阵,得到的imgTotals为0.586下的512*512
# =============================================================================
'''
输入：
imgTotal:0.586下该切片的所有大图，尺寸为(n,1216,1936,3)
startpointlist_split:上面代码产生的cutnum个小图的相对于imgTotal的左上角坐标
sizepatch_predict_small:512*512
输出：
imgTotals:每个大图对应的cutnum个小图的矩阵，尺寸为(n*cutnum,512,512,3)
'''
def Split_into_small(imgTotal,startpointlist_split,sizepatch_predict_small):
    start = time.time()
    imgTotals = []
    for k in range(len(imgTotal)):
        for i in range(len(startpointlist_split)):
            startpointtemp = startpointlist_split[i]
            startpointtempy,startpointtempx = startpointtemp
            img = imgTotal[k][startpointtempx:startpointtempx+sizepatch_predict_small[1],startpointtempy:startpointtempy+sizepatch_predict_small[0]]
            imgTotals.append(img)
    imgTotals = np.array(imgTotals)
    end = time.time()
    print('Split_into_small耗时' + str(end-start)+'/'+ str(len(imgTotals))+'张')
    return imgTotals

## =============================================================================
## 下面为计算Startpointlist_split的绝对坐标,即输入model1模型中图片左上角坐标在全片中的坐标
## =============================================================================
'''
输入：
pathsave1:保存s,startpointlist_split,p.npy的路径
filename:读取的切片名字
input_str:即为Size_set中使用到的input_str
输出：输入到model1中图片在全片中左上角的坐标，预测值，与预测相同大小的，512*0.586/input_resolution
'''
def absolute_get_startpointlist1(pathsave1, filename, input_str):#get_startpointlist1
    ratio = input_str['model1_resolution']/input_str['input_resolution']
    sizepatch_small1 = Size_set(input_str)['sizepatch_small1']
    startpointlist0 = np.load(pathsave1 + filename + '_s.npy')
    startpointlist_split = np.load(pathsave1 + 'startpointlist_split.npy')
    predict1 = np.load(pathsave1 + filename + '_p.npy')
    startpointlist_split = np.int16(np.array(startpointlist_split)*ratio)
    startpointlist1 = []
    for i in range(len(startpointlist0)):
        for j in range(len(startpointlist_split)):
            points = (startpointlist0[i][0]+startpointlist_split[j][0],startpointlist0[i][1]+startpointlist_split[j][1])
            startpointlist1.append(points)
    Sizepatch_small1 = [sizepatch_small1]*len(startpointlist1)
    return startpointlist1, predict1, Sizepatch_small1   
## =============================================================================
## 计算每个0.586分辨率下512*512中定位系统定位出的m个区域的绝对坐标 ,即model2模型输入图像左上角坐标在全片中的坐标
## =============================================================================
def absolute_get_startpointlist2(pathsave1,pathsave2,filename,input_str):
        ratio = input_str['model1_resolution']/input_str['input_resolution']
        cutnum= Size_set(input_str)['cutnum']
        sizepatch_small2 = Size_set(input_str)['sizepatch_small2']
        
        startpointlist0 = np.load(pathsave1 + filename + '_s.npy')
        dictionary = np.load(pathsave2 + filename + '_dictionary.npy')
        startpointlist2_relat = np.load(pathsave2 + filename + '_s2.npy')
        predict2 = np.load(pathsave2 + filename + '_p2.npy')
        startpointlist2_relat = np.int16(np.array(startpointlist2_relat)*ratio)    
        startpointlist2 = []
        for index, item in enumerate(dictionary):
            count = sum(dictionary[:index])
            for i in range(item):
                points = (startpointlist0[index//cutnum][0] + startpointlist2_relat[count][0],startpointlist0[index//cutnum][1] + startpointlist2_relat[count][1])
                startpointlist2.append(points)
                count = count+1
        Sizepatch_small2 = [sizepatch_small2]*len(startpointlist2)
        return startpointlist2, predict2, Sizepatch_small2
# =============================================================================
# 对上述model1.model2的绝对坐标进行合并
# =============================================================================
def absolute_get_startpointlist12(pathsave1,pathsave2,filename,input_str):
    """model1 和 model2 绝对坐标简单合并"""
    startpointlist1, predict1, Sizepatch_small1 = absolute_get_startpointlist1(pathsave1, filename, input_str)
    startpointlist2, predict2, Sizepatch_small2 = absolute_get_startpointlist2(pathsave1,pathsave2,filename,input_str)
    dictionary = np.load(pathsave2 + filename + '_dictionary.npy')
    startpointlist12 = []
    predict12 = []
    Sizepatch_small12 = []
    for index, item in enumerate(dictionary):
        if item!=0:
            a = predict2[int(np.sum(dictionary[:index])):int(np.sum(dictionary[:index])+item)]
            index_max = int(np.sum(dictionary[:index])) + a.argmax()
            startpointlist12.append(startpointlist2[index_max])
            predict12.append(a.max())
            Sizepatch_small12.append(Sizepatch_small2[index_max])
        else:
            startpointlist12.append(startpointlist1[index])
            predict12.append(predict1[index])
            Sizepatch_small12.append(Sizepatch_small1[index])
            
    predict12 = np.vstack(predict12)
    return startpointlist12, predict12, Sizepatch_small12
## =============================================================================
## 计算输入为0.293分辨率下256*256的核模型推荐出来的核的绝对坐标、以及核的面积、typeImg特征，这里直接为[]
## =============================================================================
'''
输入：
imgTotal2_core:model2的输入图像
predict2_core:
'''
def get_contours_and_feature_of_core(imgTotal2_core,predict2_core,input_str,pathsave1,pathsave2,filename,removeSmallRegion = 100,removeConvex = 1.1):        
    # 计算核特征 起始点           
    finalContours = []
    finalArea = []
    startpointlist2, _, _ = absolute_get_startpointlist2(pathsave1,pathsave2,filename,input_str)
    for i in range(len(imgTotal2_core)):
        saveResult = predict2_core[i, :,:,0]
        saveResult[saveResult>0.5]=int(255)
        saveResult[saveResult<0.5]=int(0)

        finalContourstemp = []
        finalAreatemp = []

        _, contours, _ = cv2.findContours(saveResult.astype(np.uint8), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for j in range(len(contours)):   
            #面积
            area = cv2.contourArea(contours[j])
            #凸包面积
            convexHull = cv2.convexHull(contours[j])
            convexArea = cv2.contourArea(convexHull)
            #去除小的连通域
            if area>removeSmallRegion and convexArea/area<removeConvex:
                length,_,_=contours[j].shape
                for k in range(length):
                    contours[j][k,::][0][0]+=startpointlist2[i][0]
                    contours[j][k,::][0][1]+=startpointlist2[i][1]
                finalContourstemp.append(contours[j][:,0,:])#目前这里得到是相对于model2模型输入图像的坐标，要转换到全片上
                finalAreatemp.append(area)
        finalContours.append(finalContourstemp)
        if len(finalAreatemp)==0:
            finalAreatemp=[0]
        finalArea.append(finalAreatemp) 
    # np.save("Z:\\FQL\\restruct\\code\\yjycode_copy\\test\\ShengFY-P-L240 (origin date)\\finalContours.npy", finalContours)
    return finalArea, [], finalContours