# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: wsi_related.py
@Date: 2019/12/24 
@Time: 8:49
@Desc: 存放与rnn切片分类相关的权重，包括encoder和simplernn
'''
import os

import keras
from keras.layers import Input, Activation, Dense, Add, Dropout
from keras.layers.recurrent import SimpleRNN
from keras.models import Model
from keras.applications.resnet50 import ResNet50
import keras.backend as K


def resnet_clf(input_shape=None, frozen_layers=79):
    """二分类ResNet50
    :param input_shape: 输入尺寸
    :param frozen_layers: 冻结层数
    :return: 返回建立好的模型
    """
    model_base = ResNet50(include_top=False,
                          input_shape=input_shape,
                          weights=None,
                          pooling='max')
    for layer in model_base.layers[:frozen_layers]:
        layer.trainable = False

    x = model_base.output
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(model_base.input, x)
    return model


def resnet_encoder(input_shape):
    model_base = ResNet50(include_top=False,
                          input_shape=input_shape,
                          weights=None,
                          pooling='max')
    features = model_base.output
    x = Dense(64, activation='relu')(features)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(model_base.input, [output, features])
    return model


def simple_rnn(input_shape, ndims):
    input = Input(input_shape, name='feature_input')
    x = SimpleRNN(ndims, activation='relu', name='simple_rnn')(input)
    output = Dense(1, activation='sigmoid', name='output')(x)
    return Model(input, output)


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]='2'
    # ===============================================================================
    # net = resnet_encoder((256, 256, 3))
    # net = resnet_clf((256, 256, 3))
    # net.load_weights(r'H:\weights\200810_Lean2BD\selected_w\L2BD_model2_272_pred.h5')
    # net.save(r'H:\weights\200810_Lean2BD\selected_w\models\L2BD_model2_272.h5')
    # ===============================================================================
    rnn_w = r'I:\20200810_Lean2BD\rnn_all\select_w'
    rnn_m_save = r'I:\20200810_Lean2BD\rnn_all\select_w\models'
    rnn_dict = {
        10: ['10_0_0_09-0.86-0.97.h5', '10_12_0_03-0.86-0.94.h5'],
        20: ['20_0_2_10-0.85-0.95.h5', '20_1_0_10-0.86-0.97.h5'],
        30: ['30_0_2_03-0.79-0.91.h5', '30_0_13_01-0.78-0.92.h5'],
    }
    for nb_rnn in rnn_dict:
        K.clear_session()
        rnn = simple_rnn((nb_rnn, 2048), 512)
        for w in rnn_dict[nb_rnn]:
            rnn.load_weights(os.path.join(rnn_w, w))
            rnn.save(os.path.join(rnn_m_save, w))
            print(os.path.join(rnn_m_save, w))

