# -*- coding:utf-8 -*-
import os
import time
import numpy as np

from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras.callbacks import TensorBoard
import tensorflow as tf

from utils.networks.resnet50_2classes import ResNet
from utils.read_image.dataset import DataSet

# =====================================================================================================================
# DataSet Setting
# =====================================================================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '6, 7'
num_gpu = 2
learning_rate = 1e-4
config_root = r''
path_checkpoints = r''
path_xlsx = {'train':config_root + '.xlsx',
             'test':config_root + '.xlsx'}

file_pretrain_weights = r'.h5'
start = 62
end = 100
Frozen_layers=37

enhance_flag = {'train': True, 'test': False}
scale_flag = {'train': False, 'test': False}
DataSet = {x: DataSet(path_xlsx[x],
                      crop_num=1,
                      norm_flag=True,
                      enhance_flag=enhance_flag[x],
                      scale_flag=scale_flag[x],
                      include_center=False,
                      mask_flag=False) for x in ['train', 'test']}
# =====================================================================================================================
# Model Setting
# =====================================================================================================================


path_log = os.path.join(path_checkpoints, 'logs')
hist_file = 'hist.txt'
if not os.path.exists(path_checkpoints):
    os.mkdir(path_checkpoints)
if not os.path.exists(path_log):
    os.mkdir(path_log)


print('Setting Model')
if num_gpu > 1:
    with tf.device('/cpu:0'):
        model = ResNet(input_shape=(512, 512, 3),frozen_layers=Frozen_layers)  # 37
        model.compile(optimizer=Adam(lr=learning_rate), loss=["binary_crossentropy"], metrics=["binary_accuracy"])
        if file_pretrain_weights is not None:
            print('Load weights %s' % file_pretrain_weights)
            model.load_weights(file_pretrain_weights)
    parallel_model = multi_gpu_model(model, gpus=num_gpu)
    parallel_model.compile(optimizer=Adam(lr=learning_rate), loss=["binary_crossentropy"], metrics=["binary_accuracy"])
else:
    parallel_model = ResNet(input_shape=(512, 512, 3), frozen_layers=37)
    parallel_model.compile(optimizer=Adam(lr=learning_rate), loss=["binary_crossentropy"], metrics=["binary_accuracy"])
    if file_pretrain_weights is not None:
        print('Load weights %s' % file_pretrain_weights)
        parallel_model.load_weights(file_pretrain_weights)
# =====================================================================================================================
# Training
# =====================================================================================================================

for block in range(start, end):
    print('Training with block %d' % block)

    print('Reading ...')
    since = time.time()
    img_data = {}
    label_data = {}
    for data in ['train', 'test']:
        DataSet[data].get_read_in_img_dir()
        label_data[data] = np.stack(DataSet[data].get_label())
        img_data[data] = np.stack(DataSet[data].get_img())

    print("\nTotal train images: %s" % (len(img_data['train'])),
          "\nTotal test images: %s" % (len(img_data['test'])),
          "\n%.2f sec per 1000 image" % ((time.time() - since) * 1000 / (len(img_data['train']) + len(img_data['test']))))

    tensorboard = TensorBoard(log_dir=path_log)

    hist = parallel_model.fit(img_data['train'], label_data['train'], batch_size=16*num_gpu, epochs=1, verbose=1,
                              validation_data=(img_data['test'], label_data['test']), callbacks=[tensorboard])

    with open(os.path.join(path_log, hist_file), 'a') as f:
        f.write('Block' + str(block) + '\n')
        f.write(str(hist.history) + '\n')

    if block % 2 == 0:
        if num_gpu > 1:
            model.save_weights(os.path.join(path_checkpoints, 'Block_%s.h5' % block))
        else:
            parallel_model.save_weights(os.path.join(path_checkpoints, 'Block_%s.h5' % block))
