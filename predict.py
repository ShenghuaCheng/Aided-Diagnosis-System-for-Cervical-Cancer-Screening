# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
from skimage import measure
from multiprocessing import Manager
from skimage import exposure
import multiprocessing.dummy as multiprocessing
import time

from utils.networks.model1 import Resnet_location_and_predict as Model1_LP
from utils.networks.resnet50_2classes import ResNet_f as Model2_2

from utils.inference.getImgUbantuSrp import GetImg, GetImg_whole_srp
from utils.inference.data_process import *
from utils.inference.parameterSet import Size_set_srp,TCT_set
from utils.inference.splitCoordinate import *

# =============================================================================
# 根据定位模型对model1输入图像预测得到的16*16的heatmap,获得该图中每个连通域中概率最大的像素点坐标
# 该坐标为相对于该512*512的图而言
# =============================================================================
def region_proposal(heatmap, output_shape, img=None, threshold=0.7):
    prob_max = np.max(heatmap)
    heatmap[heatmap < prob_max *threshold] = 0#根据设置的阈值，将feature中小于最大值的70%的变为黑色
    mask = (heatmap!=0).astype(np.uint8)
    mask_label = measure.label(mask.copy())
    local = []
    for c in range(mask_label.max()):#mask_label.max()获得了连通域的个数，即小图中响应强烈的区域个数O
        heatmap_temp = heatmap*(mask_label == (c+1))
        a = np.where(heatmap_temp == heatmap_temp.max())
        #因为热图为16*16,要转变为512*512则在热图中一个像素为512中的32个像素，想要取中心点，所以要加上16，即0.5*32
        local_y = np.around((a[1][0]+0.5)*output_shape[1]/heatmap.shape[1]).astype(int)
        local_x = np.around((a[0][0]+0.5)*output_shape[0]/heatmap.shape[0]).astype(int)
        local.append((local_y, local_x))
        
    if np.any(img != None):#img中只要有一个不为None,即不是img=None这种情况,将mask画上到原图上来显示
        img_prop = []
        heatmap = cv2.resize(heatmap, output_shape[::-1])#resize到512*512
        prob_max = np.max(heatmap)
        heatmap[heatmap < prob_max *threshold] = 0
        mask = (heatmap!=0).astype(np.uint8)
        _, mask_c, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(output_shape)
        img_prop = np.uint8((img / 2 + 0.5) * 255)
        for c in mask_c:
            hull = cv2.convexHull(c)  # 凸包
            img_prop = cv2.polylines(img_prop, [hull], True, (0, 255, 0), 2)  # 绘制凸包
            mask = cv2.fillPoly(mask, [hull], 255)
        return img_prop, mask, local

    else:
        return local
# =============================================================================
# 多线程gamma变换      
# =============================================================================
def szsq_color_Normal(img,lock):
    lock.acquire()
    img_szsq_trans = exposure.adjust_gamma(img,0.6)#这里img一定是bgr，但是进网络是rgb
    lock.release()
    return img_szsq_trans 
    
def MultiGammaTrans(imgTotal):
    pool = multiprocessing.Pool(16)
    manager = Manager()
    lock = manager.Lock()
    results=[]
    for img in imgTotal:
        img=pool.apply_async(szsq_color_Normal,args=(img,lock))
        results.append(img)
    results=[i.get() for i in results]
    return results

#use distance=200um as the restrict to remove the overlap regions
#p is the predict
#s is the startpoint of the predict region
'返回的predict2,s2为去重后降序排列的概率及其对应的左上角点'
def RemoveOverlap(predict2,startpointlist_2,input_resolution,thre_dis=200):
    thre_dis=thre_dis/input_resolution 
    c=list(zip(predict2,startpointlist_2))
    c.sort(key=lambda y : y[0],reverse=True)
    predict2=[]
    startpointlist_2=[]
    predict2[:],startpointlist_2[:]=zip(*c) 
    
    predict2=[i for i in predict2 if i]
    s2=[]
    for i in startpointlist_2:
        s2.append(i)
#    predict2=[i[0] for i in predict2]
    startpointlist_2=s2

    for i in range(len(startpointlist_2)):
        point=startpointlist_2[i]
        if point!=[]:
            for j in range(i+1,len(startpointlist_2)):
                temp=startpointlist_2[j]
#                print(point)
#                print(temp)
                if temp!=[]:
                    dis=np.sqrt(sum(np.power((np.array(point) - np.array(temp)), 2)))
                    if dis<thre_dis:
                        startpointlist_2[j]=[]
                        predict2[j]=[]
#                        print(str(i)+" and"+str(j)+"is overlap so delete")   
    predict2=[i for i in predict2 if i]
    s2=[]
    for i in startpointlist_2:
        if i!=[]:
            s2.append(i)
    return predict2,s2
    
    
    
    
class Predict(object):
    def __init__(self,pathfolder,filename,input_str,pathsave1,pathsave2,pathsave_img,model1WeightPath,locationWeightPath,model2WeightPath,device_id, gpu_num=1):
        self.pathfile = pathfolder + filename + input_str['filetype']
        size_set,bin_img = Size_set_srp(self.pathfile, input_str)
        self.input_resolution = size_set['input_resolution']
        self.bin_img = bin_img
        self.model1ReadLevel = size_set['model1ReadLevel']
        self.binImgLevel = size_set['binImgLevel']
        self.thre_num = size_set['thre_num']
        self.level = size_set['level']
        self.levelratio = size_set['levelratio']
        self.widthOverlap = size_set['widthOverlap']#在输入分辨率下512之间的冗余
        self.sizepatch_read = size_set['sizepatch_read']#输入分辨率下与model1分辨率下512同视野的大图尺寸
        self.sizepatch_predict_model1 = size_set['sizepatch_predict_model1']#model1分辨率下的大图尺寸
        self.sizepatch_predict_small1 = size_set['sizepatch_predict_small1']#model1模型的输入图像大小
        self.sizepatch_predict_small2 = size_set['sizepatch_predict_small2']#model2模型的输入图像大小
        self.input_str=input_str
        self.pathsave1=pathsave1
        self.pathsave2=pathsave2
        self.filename=filename
        self.pathsave_img=pathsave_img
        self.model1WeightPath=model1WeightPath
        self.model2WeightPath=model2WeightPath
        self.locationWeightPath=locationWeightPath
        self.gpu_num=gpu_num
        self.device_id=device_id
        self.sizepatch_small2=size_set['sizepatch_small2']        
        
 # =============================================================================
    # 获得直接输入到model1中的图像及输入分辨率下的(1216,1936)*0.586/input_resolution的图像
    # =============================================================================
    def Model1Predict(self):
        pathfile = self.pathfile
        level = self.level
        levelratio = self.levelratio
        widthOverlap = self.widthOverlap#在输入分辨率下大图之间的冗余
        sizepatch_read = self.sizepatch_read#输入分辨率下与model1分辨率下大图同视野的大图尺寸
        sizepatch_predict_model1 = self.sizepatch_predict_model1#model1分辨率下的大图尺寸
        pathsave1=self.pathsave1
        filename=self.filename
        pathsave_img=self.pathsave_img
        model1WeightPath=self.model1WeightPath
        locationWeightPath=self.locationWeightPath
        input_str=self.input_str
        gpu_num=self.gpu_num
        model1ReadLevel=self.model1ReadLevel
        binImgLevel=self.binImgLevel
        thre_num=self.thre_num
        bin_img=self.bin_img
        #load模型
        device_id=self.device_id
        os.environ['CUDA_VISIBLE_DEVICES'] = device_id
        model1 = Model1_LP(input_shape = input_str['model1_input'])
        model1.load_weights(locationWeightPath,by_name=True)
        model1.load_weights(model1WeightPath,by_name=True)
        print("model loading end...........")
        # 计算最高分辨率下裁剪为与0.586分辨率下512*512同视野的块左上角坐标
        try:
            startpointlist = Get_startpointlistSrp(pathfile, level, levelratio, sizepatch_read, widthOverlap)
        except:
            startpointlist = Get_startpointlistSrp(pathfile, level, levelratio, sizepatch_read, widthOverlap, threColor = 1)  
        #将全片绝对坐标是startpointlist转变为model1ReadLevel下        
        startpointlist_1=[ np.array(i)/(levelratio**model1ReadLevel) for i in startpointlist]
        startpointlist_1=[(int(i[0]),int(i[1])) for i in startpointlist_1]
        #去掉细胞占比较小的model1块
        optStartpointlist=[]
        newstartpointlist=[]
        tempSizepatch_read=(int(sizepatch_read[0]/(levelratio**binImgLevel)),int(sizepatch_read[1]/(levelratio**binImgLevel)))
        print("在binImgLevel上的读取大小为{}".format(tempSizepatch_read))
        for idex,item in enumerate(startpointlist_1): 
#            temp=(int(item[0]/(levelratio**(binImgLevel-model1ReadLevel))),int(item[1]/(levelratio**(binImgLevel-model1ReadLevel))))
            temp=(int(startpointlist[idex][0]/(levelratio**binImgLevel)),int(startpointlist[idex][1]/(levelratio**binImgLevel)))
            tempBin_img=bin_img[temp[1]:temp[1]+tempSizepatch_read[1],temp[0]:temp[0]+tempSizepatch_read[0]]
            if np.sum(tempBin_img)>thre_num:
                optStartpointlist.append(item)
                newstartpointlist.append(startpointlist[idex])        
        print("optStartpointlist is commputed end....")
        
         #下面是使用多线程读图，使用了lock可以保证图与startpointlist是一致的
        start=time.time()
        tempsizepatch_read=(int(sizepatch_read[0]/levelratio**model1ReadLevel),int(sizepatch_read[0]/levelratio**model1ReadLevel))
        objectGetImg=GetImg(pathfile, optStartpointlist, model1ReadLevel, tempsizepatch_read, pathsave_img)
        '''imgTotal为输入分辨率下'''
        imgTotal = objectGetImg.Get_predictimgMultiprocess() 
        end=time.time()
        print('read img cost: '+str(end-start))
        print("muti is end..........")
        #model1输入图像，做了颜色矫正以及归一化，且输入图像为bgr通道
        start=time.time()
        imgTotal1=MultiGammaTrans(imgTotal)
        end=time.time()
        print('img gammatrans  cost: '+str(end-start))
        imgTotal_model1 = [cv2.resize(img,sizepatch_predict_model1) for img in imgTotal1]
        imgTotal_model1 = np.array(imgTotal_model1)
        print("model1 imgTotal_model1 has made out")
        imgTotal1 = DataProcess(imgTotal_model1).Normalization()
        print("model1 imgTotal1 has made out")
        # 预测定位
        imgMTotal1 = model1.predict(imgTotal1, batch_size = 16*gpu_num, verbose=1)
        predict1 = imgMTotal1[0].copy()
        feature = imgMTotal1[1].copy()
        
        np.save(pathsave1 + filename + '_s.npy',newstartpointlist)
        np.save(pathsave1 + filename + '_p.npy',predict1)
        np.save(pathsave1 + filename + '_f.npy',feature) 
        return imgTotal
    
    def Model2Predict(self):
        pathfile=self.pathfile
        level=self.level 
        sizepatch_predict_small1=self.sizepatch_predict_small1 
        sizepatch_predict_small2=self.sizepatch_predict_small2 
        input_str=self.input_str
        pathsave_img=self.pathsave_img
        pathsave1=self.pathsave1
        pathsave2=self.pathsave2
        filename=self.filename
        model2WeightPath=self.model2WeightPath
        gpu_num=self.gpu_num
        device_id=self.device_id
        model1_resolution=input_str['model1_resolution']
        model2_resolution=input_str['model2_resolution']
        input_resolution=self.input_resolution
        sizepatch_small2=self.sizepatch_small2
        #加载model1相关参数
        flag2=os.path.exists(pathsave1 + filename + '_s.npy')
        flag3=os.path.exists(pathsave1 + filename + '_p.npy')
        flag4=os.path.exists(pathsave1 + filename + '_f.npy')
        if not (flag2 and flag3 and flag4):
            print("model1 predict start............")
            Predict.Model1Predict(self)
            print('model1 predict end..........')
        newstartpointlist=np.load(pathsave1 + filename + '_s.npy')
        predict1=np.load(pathsave1 + filename + '_p.npy')
        feature=np.load(pathsave1 + filename + '_f.npy')   
        
        #load模型,这里必须放在model1后面，因为model1中使用到层的名字
        os.environ['CUDA_VISIBLE_DEVICES'] = device_id
        model2 = Model2_2(input_shape = input_str['model2_input'])
        model2.load_weights(model2WeightPath)
        
        #计算定位块的左上角点的全局坐标并取图像
        startpointlist_2 = []
        imgModel2Num = []#记录的为每个特征图在512*512大小中连通域的个数
        predictThreshold=0.5  
        minNum=600
        maxNum=1200
        predict1=np.array(predict1)
        predict1=[i for j in predict1 for i in j]
        idx=np.argsort(-np.array(predict1))
        idx=idx.tolist()
        maxi=sum([i>predictThreshold for i in predict1])
        r=[]
        for i in range(maxi):
            index=idx[i]
            item=feature[index]
            Local_3 = region_proposal(item[...,1], sizepatch_predict_small1, img=None, threshold=0.7)    
            imgModel2Num.append(len(Local_3))
#            r.append((index,len(Local_3)))
            for local in Local_3:
                ratio1 = model1_resolution/input_resolution
                ratio2 = model2_resolution/input_resolution
                # startpointlist_2_w，startpointlist_2_h为在0.235747分辨率下1272大图中以local为中点，取一个与sizepatch_predict_small2同视野的块，该块的左上角坐标                   
                startpointlist_2_w = int(newstartpointlist[index][0] + local[0]*ratio1- (sizepatch_predict_small2[0]*ratio2)/2)
                startpointlist_2_h = int(newstartpointlist[index][1] + local[1]*ratio1- (sizepatch_predict_small2[1]*ratio2)/2)
                startpointlist_2.append((startpointlist_2_w,startpointlist_2_h))
                r.append((index,len(Local_3)))
        if sum(imgModel2Num)<minNum: 
            i=maxi
            while sum(imgModel2Num)<minNum:
                print(sum(imgModel2Num))
                index=idx[i]
                item=feature[index]
                Local_3 = region_proposal(item[...,1], sizepatch_predict_small1, img=None, threshold=0.7)    
                imgModel2Num.append(len(Local_3))                
                for local in Local_3:
                    ratio1 = model1_resolution/input_resolution
                    ratio2 = model2_resolution/input_resolution
                    # startpointlist_2_w，startpointlist_2_h为在0.235747分辨率下1272大图中以local为中点，取一个与sizepatch_predict_small2同视野的块，该块的左上角坐标                   
                    startpointlist_2_w = int(newstartpointlist[index][0] + local[0]*ratio1- (sizepatch_predict_small2[0]*ratio2)/2)
                    startpointlist_2_h = int(newstartpointlist[index][1] + local[1]*ratio1- (sizepatch_predict_small2[1]*ratio2)/2)
                    startpointlist_2.append((startpointlist_2_w,startpointlist_2_h))
                    r.append((index,len(Local_3)))
                i+=1
    
        if sum(imgModel2Num)>maxNum:
            for j in range(len(imgModel2Num)):
                if sum(imgModel2Num[0:j])<maxNum and sum(imgModel2Num[0:j+1])>=maxNum:
                    finalindex=j
            startpointlist_2=startpointlist_2[:sum(imgModel2Num[0:finalindex+1])]
            imgModel2Num=imgModel2Num[:finalindex]
                   
        #根据绝对坐标取图像           
        objectGetImg=GetImg(pathfile, startpointlist_2, level,sizepatch_small2, pathsave_img)
        imgTotal2_model2 = objectGetImg.Get_predictimgMultiprocess()
        imgTotal2 = DataProcess(imgTotal2_model2).szsq_color_Normal()
        imgTotal2_model2 = [cv2.resize(img,sizepatch_predict_small2) for img in imgTotal2]     
        imgTotal2_model2 = np.array(imgTotal2_model2)
        print("model2 imgTotal_model2 has made out")  
        imgTotal2 = DataProcess(imgTotal2_model2).Normalization()
        predict2 = model2.predict(imgTotal2,batch_size = 16*gpu_num,verbose=1)  
         
        # 保存model2预测结果       
        np.save(pathsave2 + filename + '_s2.npy',startpointlist_2)
        np.save(pathsave2 + filename + '_p2.npy',predict2)
        start=time.time()
        predict12 = predict1
        for i in range(len(imgModel2Num)):
            index=idx[i]
            a = predict2[int(np.sum(imgModel2Num[:i])):int(np.sum(imgModel2Num[:i])+imgModel2Num[i])]
            a = a.max()
            predict12[index]=a
        end=time.time()
        print('read img cost: '+str(start-end))
# =============================================================================
# 按照序列来进行排序
# =============================================================================
        list1=[]
        for i in range(len(imgModel2Num)):      
            for j in range(int(np.sum(imgModel2Num[:i])),int(np.sum(imgModel2Num[:i])+imgModel2Num[i])):
                list1.append(idx[i])
        predict12 = np.vstack(predict12)
        np.save(pathsave2 + filename + '_p12.npy',predict12)
        return imgTotal2,startpointlist_2,predict2
                
        
if __name__ == '__main__':
    input_str = {'filetype': '.mrxs',# .svs .sdpc .mrxs
                     'level':0,
                     'read_size_model1_resolution': (512, 512), #0.586分辨率
                     'scan_over': (128, 128),#0.586分辨率下 (512, 512)之间的冗余
                     'model1_input': (512, 512, 3),
                     'model2_input': (256, 256, 3),
                     'model1_resolution':0.586,
                     'model2_resolution':0.293}
    pathsave_img = None
    device_id = '2'
    model1WeightPath='.h5'
    locationWeightPath='.h5'
    model2WeightPath='.h5'
    model3WeightPath='.h5'
    num_recom=10
    online=False
    pathfolder='/'
    filename=""
    pathsave1="/"
    pathsave2=pathsave1 + 'model2/'
    if not os.path.exists(pathsave1):
            os.mkdir(pathsave1) 
    if not os.path.exists(pathsave2):
        os.makedirs(pathsave2)  
    
    object_getSlideModelInput = Predict(pathfolder,filename,input_str,pathsave1,pathsave2,pathsave_img,model1WeightPath,locationWeightPath,model2WeightPath,device_id,gpu_num=1)
    _ = object_getSlideModelInput.Model1Predict()
    _,predict2,startpointlist_2 = object_getSlideModelInput.Model2Predict()
    #按照predict2的概率，并使用200um去重后的概率及其对应左上角点的坐标
    input_resolution=size_set['input_resolution']
    predict2,startpointlist_2 = RemoveOverlap(predict2,startpointlist_2,input_resolution,thre_dis=200)
    
#   获得去重后所有的model2在原始尺寸的图像
    objectGetImg = GetImg(os.apth.join(pathfolder,filename), startpointlist_2, level=0, (256, 256), pathsave_img)
    imgTotal2_model2 = objectGetImg.Get_predictimgMultiprocess()
    #将前10个保存在pathsave_img中
    save_img=r'2\model2_rec10'
    for i in range(10):
        cv2.imwrite(save_img+"\\"+str(i)+"_"+str(predict2[i])+"_"+str(startpointlist_2[i])+".tif",imgTotal2_model2[i])

