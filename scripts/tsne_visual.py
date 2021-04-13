# -*- coding:utf-8 -*-

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
    encoder_weight = r'model2.h5'
    encoder = resnet_encoder((256, 256, 3))
    encoder.load_weights(encoder_weight)
    # read in imgs
    nb_readin = 1000
    labels = [1] * nb_readin + [0] * nb_readin

    n_list = random.sample([os.path.join(r'', d) for d in os.listdir(r'') if '.tif' in d], 50)

    p_list = random.sample([os.path.join(r'', d) for d in os.listdir(r'') if '.tif' in d], 92)

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

