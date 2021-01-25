# -*- coding: utf-8 -*-
"""
Created on Sat Jul 08 15:53:01 2017

@author: chengshenghua
"""

from __future__ import print_function
from keras.models import Model
from keras.layers import Activation, add
from keras.regularizers import l2
from keras.layers import Conv2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from others.manu_yjy_code.networks.resnet50_2classes import ResNet


def atrous_conv_block(input_tensor, kernel_size, filters, stage, block, weight_decay=0., strides=(1, 1), atrous_rate=(2, 2), batch_momentum=0.99):
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


def atrous_identity_block(input_tensor, kernel_size=3, filters=[256, 256, 1024], stage=11, block='a', weight_decay=0., atrous_rate=(2, 2), batch_momentum=0.99):
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

    x = Conv2D(filter2, (kernel_size, kernel_size), dilation_rate=atrous_rate, padding='same', name=conv_name_base + '2b', kernel_regularizer=l2(weight_decay))(x)
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

    x = Conv2D(filter1, (1, 1), strides=strides, name=conv_name_base + '2a', kernel_regularizer=l2(weight_decay))(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a', momentum=batch_momentum)(x)
    x = Activation('relu')(x)

    x = Conv2D(filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b', momentum=batch_momentum)(x)
    x = Activation('relu')(x)

    x = Conv2D(filter3, (1, 1), name=conv_name_base + '2c', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c', momentum=batch_momentum)(x)

    shortcut = Conv2D(filter3, (1, 1), strides=strides, name=conv_name_base + '1', kernel_regularizer=l2(weight_decay))(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1', momentum=batch_momentum)(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x


def conv2d_bn(input_tensor, kernel_size, filters, stage, block, padding='same', strides=(1, 1), weight_decay=0., batch_momentum=0.99):
    conv_name_base = 'conv' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    x = Conv2D(filters, (kernel_size, kernel_size), name=conv_name_base + '2a', strides=strides, padding=padding, kernel_regularizer=l2(weight_decay))(input_tensor)
    x = BatchNormalization(axis=-1, name=bn_name_base + '2a', momentum=batch_momentum)(x)
    x = Activation('relu')(x)
    return x


def Resnet_atrous(input_shape=None):
    # input_shape = (512, 512, 3)
    model_base = ResNet(input_shape)
    x = model_base.get_layer("activation_49").output
    # (16,16,2048)
    x = atrous_conv_block(x, 3, [512,512,256], stage=11, block='a', atrous_rate=(2, 2))
    # (16,16,256)
    x = atrous_identity_block(x, 3, [256,256,256], stage=11, block='b',atrous_rate=(2, 2))
    # (16,16,256)
    x = atrous_identity_block(x, 3, [256,256,256], stage=11, block='c', atrous_rate=(2, 2))
    # (16,16,256)
    x = conv2d_bn(x, 3, 64, stage=11, block='d')
    # (16,16,64)
    x = Conv2D(2, (1, 1), kernel_initializer='he_normal', activation='softmax', padding='same', strides=(1, 1))(x)
    # (16,16,2)
    # 预定义激活函数 权重初始化全部采用 He Normal He正态分布初始化方法，参数由0均值，标准差为sqrt(2 / fan_in) 的正态分布产生
    # softmax：对输入数据的最后一维进行softmax
    model = Model(model_base.input, x)
    
    for layer in model_base.layers[:173]: # 79/37
        layer.trainable = False
        
    return model


