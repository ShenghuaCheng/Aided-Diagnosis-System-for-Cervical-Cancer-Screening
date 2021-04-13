# -*- coding:utf-8 -*-
import os
import numpy as np
import pandas as pd
from utils.read_image.dataset import DataSet
from utils.networks.resnet50_2classes import ResNet


weights_filelist = [
    r'.h5',
]
path_result = r''
os.makedirs(path_result, exist_ok=True)

gpu_num = 1
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

config_path = r''
DataSet = {x: DataSet(config_path,
                      '.xlsx',
                      x,
                      crop_num=1,
                      norm_flag=True,
                      enhance_flag=False,
                      scale_flag=False,
                      include_center=False,
                      mask_flag=False) for x in ['Itest']}

Net = ResNet(input_shape=(512, 512, 3))

t = 'Itest'
classes = DataSet[t].get_read_in_img_dir()
for c in classes:
    label_data = np.stack(DataSet[t].get_label(classes=[c]))
    img_data = np.stack(DataSet[t].get_img(classes=[c]))
    for weights_path in weights_filelist:
        print(c)
        print('Set model %s' % weights_path)
        Net.load_weights(weights_path)
         # 预测及统计
        preds_prob = Net.predict(img_data, batch_size=16*gpu_num, verbose=1)

        if label_data[0]==1:
            acc=np.sum(preds_prob>0.5)/len(preds_prob)
        elif label_data[0]==0:
            acc=np.sum(preds_prob<0.5)/len(preds_prob)
        else:
            print('error!')
        print(acc)
        with open(path_result+'/'+c.split('/')[1]+'.txt', 'a') as f:
            f.write(str(acc)+'\n')
record_col=[]
for c in classes:
    with open(path_result+'/'+c.split('/')[1]+'.txt', 'r') as f:
        acc = f.read()
    record_col.append(acc.split('\n')[:-1])
    
record_col= np.vstack(record_col)
record_df = pd.DataFrame(record_col)
record_df.index = classes
record_df.columns = [a.split('\\')[-1][:-3] for a in weights_filelist]
writer = pd.ExcelWriter(path_result+'/result.xlsx')
record_df.to_excel(writer, 'Sheet1', float_format='%.5f')  
writer.save()    