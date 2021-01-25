# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: tsne_visual.py
@Date: 2020/1/14 
@Time: 15:28
@Desc:
'''
import os
import time
import random
from functools import partial
import numpy as np
from multiprocessing.dummy import Pool
from utils.networks.wsi_related import resnet_encoder
from utils.auxfunc import tsne
from utils.auxfunc.readin_val import readin_val


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '7'
    # create encoder model
    encoder_weight = r'F:\LiuSibo\Exps\200108_paper\weights\AB_enhance\model2_pred\ckpt_complex\Block_68.h5'
    # encoder_weight = r'F:\LiuSibo\Exps\200108_paper\weights\AB_enhance\model2_pred\ckpt_simple\Block_240.h5'
    encoder = resnet_encoder((256, 256, 3))
    encoder.load_weights(encoder_weight)
    # read in imgs
    nb_readin = 1000
    labels = [1] * nb_readin + [0] * nb_readin

    # n_fld = r'H:\AdaptDATA\test\3d\sfy1\nplus'
    # p_fld = r'H:\AdaptDATA\test\3d\sfy1\ASCUS'
    n_list = random.sample([os.path.join(r'H:\AdaptDATA\test\3d\sfy1\n\n_0', d) for d in os.listdir(r'H:\AdaptDATA\test\3d\sfy1\n\n_0') if '.tif' in d], 50) + \
             random.sample([os.path.join(r'H:\AdaptDATA\test\3d\sfy1\n\n_5', d) for d in os.listdir(r'H:\AdaptDATA\test\3d\sfy1\n\n_5') if '.tif' in d], 450) + \
             random.sample([os.path.join(r'H:\AdaptDATA\test\3d\sfy2\n\n_0', d) for d in os.listdir(r'H:\AdaptDATA\test\3d\sfy2\n\n_0') if '.tif' in d], 50) + \
             random.sample([os.path.join(r'H:\AdaptDATA\test\3d\sfy2\n\n_5', d) for d in os.listdir(r'H:\AdaptDATA\test\3d\sfy2\n\n_5') if '.tif' in d], 450)

    p_list = random.sample([os.path.join(r'H:\AdaptDATA\test\3d\sfy1\ASCUS', d) for d in os.listdir(r'H:\AdaptDATA\test\3d\sfy1\ASCUS') if '.tif' in d], 92) + \
             random.sample([os.path.join(r'H:\AdaptDATA\test\3d\sfy1\HSIL', d) for d in os.listdir(r'H:\AdaptDATA\test\3d\sfy1\HSIL') if '.tif' in d], 319) + \
             random.sample([os.path.join(r'H:\AdaptDATA\test\3d\sfy1\LSIL', d) for d in os.listdir(r'H:\AdaptDATA\test\3d\sfy1\LSIL') if '.tif' in d], 89) + \
             random.sample([os.path.join(r'H:\AdaptDATA\test\3d\sfy2\ASCUS', d) for d in os.listdir(r'H:\AdaptDATA\test\3d\sfy2\ASCUS') if '.tif' in d], 260) + \
             random.sample([os.path.join(r'H:\AdaptDATA\test\3d\sfy2\HSIL', d) for d in os.listdir(r'H:\AdaptDATA\test\3d\sfy2\HSIL') if '.tif' in d], 206) + \
             random.sample([os.path.join(r'H:\AdaptDATA\test\3d\sfy2\LSIL', d) for d in os.listdir(r'H:\AdaptDATA\test\3d\sfy2\LSIL') if '.tif' in d], 34)

    readin_func = partial(readin_val, re_size=(1536, 1536), crop_size=(256, 256))
    since = time.time()
    print('read in images')
    p = Pool(16)
    p_img = p.map(readin_func, p_list)
    n_img = p.map(readin_func, n_list)
    p.close()
    p.join()
    print('consume %.2f' % (time.time()-since))
    print('embedding images')
    scores, embedding = tsne.embedding_imgs(np.stack(p_img+n_img), encoder)
    print('plotting')
    tsne.plot_embedding(embedding, scores,  labels, title='AB')

