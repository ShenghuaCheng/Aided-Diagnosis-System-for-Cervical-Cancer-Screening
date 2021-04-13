# -*- coding:utf-8 -*-
from __future__ import print_function
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.resnet50 import ResNet50


def ResNet(input_shape=None, frozen_layers=79):
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


def ResNet_f(input_shape=None, frozen_layers=79):
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

    f = model_base.output
    x = Dense(64, activation='relu')(f)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(model_base.input, [x, f])
    return model


     

