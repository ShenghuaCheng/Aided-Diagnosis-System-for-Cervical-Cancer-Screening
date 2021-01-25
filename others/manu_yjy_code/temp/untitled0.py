# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 14:14:33 2019

@author: A-WIN10
"""

imgMTotal_combine = []
    for i in range(len(startpointlist)):
        predict_12_temp = np.vstack(predict_12[i*num:(i+1)*num])
        score = np.sort(predict_12_temp,axis= 0)
        if score[-1]<0.9:
            imgMTotal_combine.append(score[-1])
        else:
            score[score<0.9]=0
            imgMTotal_combine.append(np.sum(score))
    Index = np.argsort(np.array(imgMTotal_combine),axis=0)[::-1][:num_recom]
    
    
    startpointlist_12 = startpointlist_1.copy()
    predict_12 = predict1.copy().tolist()
    Sizepatch_small12 = count.copy()
    for index, item in enumerate(count):
        Sizepatch_small12[index] = sizepatch_small1
        if item!=0:
            a = predict2[np.sum(count[:index]):np.sum(count[:index])+item]
            b = startpointlist_2[np.sum(count[:index]):np.sum(count[:index])+item]
            a = a.tolist()
            predict_12[index] = a
            startpointlist_12[index] = b
            Sizepatch_small12[index] = [sizepatch_small2]*len(a)
    imgMTotal_combine = []
    for i in range(len(startpointlist)):
        predict_12_temp = np.vstack(predict_12[i*num:(i+1)*num])
        score = np.sort(predict_12_temp,axis= 0)
        if score[-1]<0.9:
            imgMTotal_combine.append(score[-1])
        else:
            score[score<0.9]=0
            imgMTotal_combine.append(np.sum(score))
    Index = np.argsort(np.array(imgMTotal_combine),axis=0)[::-1][:num_recom]
    
    start_sizepatch = []
    label_annotation = []
    start_sizepatch_small12 = []
    sizepatch_small12 = []
    label_annotation_small12 = []
    for item in Index:
        item = int(item)
        start_sizepatch.append(startpointlist[item])
        label_annotation.append(imgMTotal_combine[item])
 
        predict_12_temp = np.vstack(predict_12[item*num:(item+1)*num])
        startpointlist_12_temp = np.vstack(startpointlist_12[item*num:(item+1)*num])
        Sizepatch_small12_temp = np.vstack(Sizepatch_small12[item*num:(item+1)*num])
        
        index = np.argsort(np.array(predict_12_temp),axis=0)[::-1]
       
        if predict_12_temp[index[0]]<0.9:
            start_sizepatch_small12.append(startpointlist_12_temp[index[0]][0])
            label_annotation_small12.append(predict_12_temp[index[0]][0])
            sizepatch_small12.append(Sizepatch_small12_temp[index[0]][0])
        else:
            i=0
            while (predict_12_temp[index[i]]>=0.9):
                start_sizepatch_small12.append(startpointlist_12_temp[index[i]][0])
                label_annotation_small12.append(predict_12_temp[index[i]][0])
                sizepatch_small12.append(Sizepatch_small12_temp[index[i]][0])
                i+=1
    contourslist = Get_rectcontour(start_sizepatch,sizepatch)
    contourslist_small12 = Get_rectcontour(start_sizepatch_small12,sizepatch_small12)
    #xml结果保存
    saveContours_xml([contourslist,contourslist_small12],[label_annotation,label_annotation_small12],['#00ff00','#ff0000'],pathfolder_xml +'model2_370_regionnew/'+ filename + '.xml')


    startpointlist = Get_startpointlist(ors, level, levelratio, sizepatch, widthOverlap,flag)
    imgTotal = Get_predictimgMultiprocess(startpointlist, ors, level, sizepatch, sizepatch_predict)
    # 3.2为model1预测裁小图
    imgTotal = Split_into_small(imgTotal,startpointlist_split,sizepatch_predict_small1)
    imgTotal = trans_Normalization(imgTotal,channal_trans_flag= True)
    # 3.3model1预测定位
    imgMTotal1 = model1.predict(imgTotal, batch_size = 32*gpu_num, verbose=1)
    predict1 = imgMTotal1[0].copy()
    feature = imgMTotal1[1].copy()
    # 3.4保存中间结果