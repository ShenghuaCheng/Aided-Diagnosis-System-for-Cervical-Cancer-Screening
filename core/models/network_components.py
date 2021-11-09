# -*- coding: utf-8 -*-
"""
This file is a part of project "Aided-Diagnosis-System-for-Cervical-Cancer-Screening".
See https://github.com/ShenghuaCheng/Aided-Diagnosis-System-for-Cervical-Cancer-Screening for more information.

File name: network_components
Description: Components of model.
"""

from __future__ import print_function
from keras.layers import Input, Conv2D, Activation, Dense, Dropout, add
from keras.layers.recurrent import SimpleRNN
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from keras.applications.resnet50 import ResNet50
from keras import backend as K

__all__ = [
    "resnet50_clf",
    "resnet50_loc",
    "resnet50_det",
    "resnet50_enc",
    "simple_rnn"
]


def atrous_conv_block(input_tensor, kernel_size, filters, stage, block, weight_decay=0., strides=(1, 1),
                      atrous_rate=(2, 2), batch_momentum=0.99):
    filter1, filter2, filter3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filter1, (1, 1), strides=strides,
               name=conv_name_base + '2a', kernel_regularizer=l2(weight_decay))(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a', momentum=batch_momentum)(x)
    x = Activation('relu')(x)

    x = Conv2D(filter2, (kernel_size, kernel_size), padding='same',
               dilation_rate=atrous_rate, name=conv_name_base + '2b', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b', momentum=batch_momentum)(x)
    x = Activation('relu')(x)

    x = Conv2D(filter3, (1, 1), name=conv_name_base + '2c', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c', momentum=batch_momentum)(x)

    shortcut = Conv2D(filter3, (1, 1), strides=strides,
                      name=conv_name_base + '1', kernel_regularizer=l2(weight_decay))(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1', momentum=batch_momentum)(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x


def atrous_identity_block(input_tensor, kernel_size=3, filters=[256, 256, 1024], stage=11, block='a', weight_decay=0.,
                          atrous_rate=(2, 2), batch_momentum=0.99):
    filter1, filter2, filter3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    filter1 = 16  # 降维减少参数量
    x = Conv2D(filter1, (1, 1), name=conv_name_base + '2a', kernel_regularizer=l2(weight_decay))(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a', momentum=batch_momentum)(x)
    x = Activation('relu')(x)

    x = Conv2D(filter2, (kernel_size, kernel_size), dilation_rate=atrous_rate, padding='same',
               name=conv_name_base + '2b', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b', momentum=batch_momentum)(x)
    x = Activation('relu')(x)

    x = Conv2D(filter3, (1, 1), name=conv_name_base + '2c', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c', momentum=batch_momentum)(x)
    x = add([x, input_tensor])

    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, weight_decay=0., strides=(2, 2), batch_momentum=0.99):
    bn_axis = -1
    filter1, filter2, filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filter1, (1, 1), strides=strides, name=conv_name_base + '2a', kernel_regularizer=l2(weight_decay))(
        input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a', momentum=batch_momentum)(x)
    x = Activation('relu')(x)

    x = Conv2D(filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b',
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b', momentum=batch_momentum)(x)
    x = Activation('relu')(x)

    x = Conv2D(filter3, (1, 1), name=conv_name_base + '2c', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c', momentum=batch_momentum)(x)

    shortcut = Conv2D(filter3, (1, 1), strides=strides, name=conv_name_base + '1', kernel_regularizer=l2(weight_decay))(
        input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1', momentum=batch_momentum)(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x


def conv2d_bn(input_tensor, kernel_size, filters, stage, block, padding='same', strides=(1, 1), weight_decay=0.,
              batch_momentum=0.99):
    conv_name_base = 'conv' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters, (kernel_size, kernel_size), name=conv_name_base + '2a', strides=strides, padding=padding,
               kernel_regularizer=l2(weight_decay))(input_tensor)
    x = BatchNormalization(axis=-1, name=bn_name_base + '2a', momentum=batch_momentum)(x)
    x = Activation('relu')(x)
    return x


def resnet50_clf(input_shape=None, frozen_layers=79):
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


def resnet50_loc(input_shape=None):
    # input_shape = (512, 512, 3)
    model_base = resnet50_clf(input_shape)
    x = model_base.get_layer("activation_49").output
    # (16,16,2048)
    x = atrous_conv_block(x, 3, [512, 512, 256], stage=11, block='a', atrous_rate=(2, 2))
    # (16,16,256)
    x = atrous_identity_block(x, 3, [256, 256, 256], stage=11, block='b', atrous_rate=(2, 2))
    # (16,16,256)
    x = atrous_identity_block(x, 3, [256, 256, 256], stage=11, block='c', atrous_rate=(2, 2))
    # (16,16,256)
    x = conv2d_bn(x, 3, 64, stage=11, block='d')
    # (16,16,64)
    x = Conv2D(2, (1, 1), kernel_initializer='he_normal', activation='softmax', padding='same', strides=(1, 1))(x)
    # (16,16,2)
    # 预定义激活函数 权重初始化全部采用 He Normal He正态分布初始化方法，参数由0均值，标准差为sqrt(2 / fan_in) 的正态分布产生
    # softmax：对输入数据的最后一维进行softmax
    model = Model(model_base.input, x)

    for layer in model_base.layers[:173]:  # 79/37
        layer.trainable = False

    return model


def resnet50_det(input_shape):
    # input_shape = (512, 512, 3)
    model_base = ResNet50(include_top=False, input_shape=input_shape, weights='imagenet', pooling='max')
    # feature_model# (16,16,2048)
    x = model_base.get_layer("activation_49").output
    feature = atrous_conv_block(x, 3, [512, 512, 256], stage=11, block='a', atrous_rate=(2, 2))
    feature = atrous_identity_block(feature, 3, [256, 256, 256], stage=11, block='b', atrous_rate=(2, 2))
    feature = atrous_identity_block(feature, 3, [256, 256, 256], stage=11, block='c', atrous_rate=(2, 2))
    feature = conv2d_bn(feature, 3, 64, stage=11, block='d')

    # 该版本的softmax能够与TensorRT保持一致
    # feature = Conv2D(2, (1, 1), kernel_initializer='he_normal', activation=K.softmax, padding='same', strides=(1, 1))(feature)

    feature = Conv2D(2, (1, 1), kernel_initializer='he_normal', activation='softmax', padding='same', strides=(1, 1))(
        feature)

    # predict_model
    predict = model_base.output
    predict = Dense(64, activation='relu')(predict)
    predict = Dropout(0.5)(predict)
    predict = Dense(1, activation='sigmoid')(predict)

    # 预定义激活函数 权重初始化全部采用 He Normal He正态分布初始化方法，参数由0均值，标准差为sqrt(2 / fan_in) 的正态分布产生
    # softmax：对输入数据的最后一维进行softmax
    model = Model(model_base.input, outputs=[predict, feature])
    for layer in model.layers[:173] + model.layers[204:213:2]:  # 79/37
        layer.trainable = False
    return model


def resnet50_enc(input_shape):
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
