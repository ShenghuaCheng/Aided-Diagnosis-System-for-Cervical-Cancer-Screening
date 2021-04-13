# -*- coding:utf-8 -*-
from keras.layers import Input, Dense, Conv2D, Concatenate
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.models import Model


def simple_rnn(input_shape, ndims):
    """
    :param input_shape: (nb_dims)
    :param ndims:
    :return:
    """
    input = Input(input_shape, name='feature_input')
    x = SimpleRNN(ndims, activation='relu', name='simple_rnn')(input)
    output = Dense(1, activation='sigmoid', name='output')(x)
    return Model(input, output)


def lstm(input_shape, ndims):
    """
    :param input_shape:
    :param ndims:
    :return:
    """
    input = Input(input_shape, name='feature_input')
    x = LSTM(ndims, name='lstm')(input)
    output = Dense(1, activation='sigmoid', name='output')(x)
    return Model(input, output)


def lstm_exp(input_shape, ndims):
    """
    :param input_shape:
    :param ndims:
    :return:
    """
    input = Input(input_shape, name='feature_input')
    x = LSTM(ndims, name='lstm')(input)
    x = Dense(ndims//4, activation='relu', name='clf_hidden')(x)
    output = Dense(1, activation='sigmoid', name='output')(x)
    return Model(input, output)


def lstm_score(input_shape, sf_shape, ndims):
    """
    :param input_shape:
    :param ndims:
    :return:
    """
    input = Input(input_shape, name='feature_input')
    sf_input = Input(sf_shape, name='score_input')
    x = LSTM(ndims, name='lstm')(input)
    x = Concatenate()([x, sf_input])  # 将分数特征与lstm的整合特征拼接
    x = Dense(ndims//4, activation='relu', name='clf_hidden')(x)
    output = Dense(1, activation='sigmoid', name='output')(x)
    return Model([input, sf_input], output)


def merge_cnn(input_shape):

    input = Input(input_shape, name='feature_input')
    x = Conv2D(64, (3, 3), strides=(1, 2), activation='relu')(input)
    x = Conv2D(64, (3, 3), strides=(1, 2), activation='relu')(x)
    x = Conv2D(64, (3, 3), strides=(1, 2), activation='relu')(x)
    x = Conv2D(64, (3, 3), strides=(1, 2), activation='relu')(x)
    x = Conv2D(64, (3, 3), strides=(1, 2), activation='relu')(x)
    x = Conv2D(64, (3, 3), strides=(1, 2), activation='relu')(x)
    output = Conv2D(64, (3, 3), strides=(1, 2), activation='relu')(x)
    return Model(input, output)
