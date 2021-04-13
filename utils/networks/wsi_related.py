# -*- coding:utf-8 -*-
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

