# -*- coding: utf-8 -*-
"""
Created on Sat Jul 08 15:53:01 2017

@author: chengshenghua
"""

from __future__ import print_function
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.resnet50 import ResNet50


def ResNet(input_shape = None, classes = 4):
    model_base = ResNet50(include_top=False, input_shape = input_shape, weights='imagenet', pooling='max')
    for layer in model_base.layers[:79]:  #79
        layer.trainable = False
   
    x = model_base.output
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(classes, activation='softmax')(x)
    model = Model(model_base.input, x)
    return model



     

