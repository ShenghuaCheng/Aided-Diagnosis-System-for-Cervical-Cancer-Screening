# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: tmp_check_data.py
@Date: 2020/12/21 
@Time: 18:56
@Desc: 检查sfy6的数据是否存在问题
'''
import os
import numpy as np
import glob2
import random
from others.manu_yjy_code.networks.resnet50_2classes import ResNet
from utils.manu.aux_func import rd_imgs


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    weights_filelist = [
        r'F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\manu_yjy\model1\new_E\m1_pred_Block_333.h5',
        r'F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\manu_yjy\model1\weights\szsq_model1_700_pred.h5',
        r'F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\manu_yjy\model1\weights\Block_330.h5']

    Net = ResNet(input_shape=(512, 512, 3))

    img_dirs = glob2.glob(r'H:\AdaptDATA\train\szsq\sdpc_sfy6\gamma_simple\n\n_0\*tif')
    img_dirs = random.sample(img_dirs, 1000)

    data = rd_imgs(img_dirs, (512, 512, 3))

    for weights_path in weights_filelist:
        print('Set model %s' % weights_path)
        Net.load_weights(weights_path)
         # 预测及统计
        preds_prob = Net.predict(data, batch_size=16, verbose=1)
        np.save(r'I:\20201221_Manu_Fig3\{}.npy'.format(os.path.split(weights_path)[-1]), preds_prob)
        acc = np.sum(preds_prob < 0.5) / len(preds_prob)
        print(acc)
