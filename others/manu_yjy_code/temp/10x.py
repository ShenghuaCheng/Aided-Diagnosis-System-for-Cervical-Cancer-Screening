import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from matplotlib import pyplot as plt
from model_re_dep import Resnet_atrous as RA
from ResNetClassification import ResNet as Model2
from ResNetClassification_79_multiclasses import ResNet as Model2_multi
from keras.utils import multi_gpu_model 
import tensorflow as tf
import time
import numpy as np
import cv2
from openslide import OpenSlide
from predict_310_function import *
from matplotlib import pyplot as plt
# 1:positive 0:negative
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
    return startpointlist


level = 0
levelratio = 4
flag ='1'
strategy = 'max'
model2_type = '2'
#10x
sizepatch = (1216,1936)#2432 3872
sizepatch_small = (512,512)
widthOverlap = (120,120)
#predict
sizepatch_predict = (1216,1936)#1459 2323
sizepatch_predict_small = (512,512)
widthOverlap_predict = (160,156)#3*5
num = 15
#widthOverlap_predict = (197,150) #4*6
#20x model2
sizepatch_small2 = (128,128)
sizepatch_predict_small2 = (256,256)
pathfolder_svs = 'H:/TCTDATA/our/10x/Positive/ShengFY-P-L240 (origin date)/'#ShengFY-P-L240 (origin date)
pathfolder_xml = 'H:/recom_30/model12/10x/ShengFY-P-L240 (origin date)/'
# 建立并行预测模型
gpu_num = 1
# =============================================================================
# model1 = RA(input_shape = (512,512,3))
# model1.load_weights('H:/weights/model1_340_3d1_3d2_our_10x_adapt.h5',by_name=True)
# model1.load_weights('H:/weights/model1_340417_3d1_3d2_our_10x_adapt.h5',by_name=True)
# =============================================================================
model2 = Model2(input_shape = (256,256,3))
model2.load_weights('H:/weights/model2_670_3d1_3d2_our_10x_adapt.h5')#
# 获取操作对象列表
CZ = os.listdir(pathfolder_svs)
CZ = [cz for cz in CZ if '.svs' in cz]

for cz in CZ:
    filename = cz[:-4]
    pathsvs = pathfolder_svs + cz
    print(pathsvs)
    ors = OpenSlide(pathsvs)
# =============================================================================
#     start = time.time()
#     startpointlist = Get_startpointlist(ors, level, levelratio, sizepatch, widthOverlap,flag)      
#     imgTotal = Get_predictimgMultiprocess(startpointlist, ors, level, sizepatch, sizepatch_predict)
#     end= time.time()
#     print('还原拍摄过程的小图片并读入_耗时>>>>'+str(end -start)[0:5] + 's' )
#     print('还原拍摄过程的小图片并读入_耗时>>>>'+str(end -start)[0:5] + 's' )
#     #将拍摄小图拆分成适合网络大小的小图
#     start = time.time()
#     imgTotal,num,startpointlist_split = Split_into_small(imgTotal,sizepatch_predict,sizepatch_predict_small, widthOverlap_predict)
#     imgTotal = trans_Normalization(imgTotal)
#     #获取512*512小图起始坐标点
#     startpointlist_split = np.uint16(np.array(startpointlist_split))
#     end = time.time()
#     print('将拍摄小图拆分成适合网络大小的小图并归一化_耗时>>>>'+str(end -start)[0:5] + 's' )
#     # model1开始预测
#     start = time.time()
#     imgMTotal1 = model1.predict(imgTotal, batch_size = 32*gpu_num, verbose=1)
#     end = time.time()
#     predict1 = imgMTotal1[0].copy()
#     feature = imgMTotal1[1].copy()
#     print('model1开始预测_耗时>>>>'+str(end -start)[0:5] + 's' )
#     # 保存model1预测文件
#     np.save(pathfolder_xml + filename + '_s.npy',startpointlist)
#     np.save(pathfolder_xml + filename + '_p.npy',predict1)
#     np.save(pathfolder_xml + filename + '_f.npy',feature) 
# =============================================================================
    # 还原拍摄过程的小图片并读入
    startpointlist = np.load(pathfolder_xml + filename + '_s.npy')
    predict1 = np.load(pathfolder_xml + filename + '_p.npy')
    feature = np.load(pathfolder_xml + filename + '_f.npy') 
    # 计算model1预测小块起始坐标
    startpointlist_split = Startpointlist_split(sizepatch_predict,sizepatch_predict_small, widthOverlap_predict)
    startpointlist_split = np.uint16(np.array(startpointlist_split))
    startpointlist_1 = []
    for i in range(len(startpointlist)):
        for j in range(num):
            startpointlist_1.append((startpointlist[i][0]+startpointlist_split[j][0],startpointlist[i][1]+startpointlist_split[j][1]))
    num_recom = min(np.sum(predict1>0.5),200)
    Index = np.argsort(np.array(predict1),axis=0)[::-1][:num_recom]
    
    # 返回model1定位结果
    startpointlist_2 = []
    count = []
    for index in Index:
        index = index[0]
        item = feature[index]
        _, _, Local_3 = region_proposal(item[...,1])
        count.append(len(Local_3))
        for x in Local_3:
            x = np.uint16(np.array(x))
            startpointlist_2.append((startpointlist_1[index][0]+x[1]-int(sizepatch_small2[0]/2),startpointlist_1[index][1]+x[0]-int(sizepatch_small2[1]/2))) 
    # 为model2读入定位结果图并归一化
    imgTotal2 = Get_predictimgMultiprocess(startpointlist_2, ors,0,sizepatch_small2,sizepatch_predict_small2)
    imgTotal2 = trans_Normalization(imgTotal2)
    # model2开始预测
    predict2 = model2.predict(imgTotal2,batch_size = 8*gpu_num,verbose=1)

    predict_color12 = []
    label_annotation_small12 = []
    start_sizepatch_small12 = []
    sizepatch_small12 = []
    for index, item in enumerate(predict2):
        if item>0.8:
            predict_color12.append('#ff0000')
            label_annotation_small12.append('p')
        else:
            predict_color12.append('#00ff00')
            label_annotation_small12.append('n')
        start_sizepatch_small12.append(startpointlist_2[index])
        sizepatch_small12.append((128,128))
    contourslist_small12 = Get_rectcontour(start_sizepatch_small12,sizepatch_small12)           
    saveContours_xml([contourslist_small12],[label_annotation_small12],[predict_color12],pathfolder_xml +'xml/'+ filename + '.xml')
       