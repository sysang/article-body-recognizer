import re
import math
import random
import html
import base64
import pickle
import jsonlines
import json
import datetime
import time
import _thread

from pathlib import Path
from lxml import etree
from collections import deque
import numpy as np
import numpy.ma as ma
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adagrad, Nadam
from tensorflow.keras.optimizers.schedules import InverseTimeDecay
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM, Bidirectional, Dropout, LayerNormalization, GRU, BatchNormalization, Dot
from tensorflow.keras.layers import concatenate, Reshape, SpatialDropout1D, Conv1D, Flatten, AveragePooling1D, MaxPool1D, Average, Maximum, Multiply, Add
from tensorflow.keras.models import Model, Sequential

from article_body_recognizer.system_specs import char_emb_training_specs
from article_body_recognizer.ANNs.charemb_network import CharEmbeddingV5


def CharembComparatorV1(cfg):
    emb_trainable = cfg['emb_trainable']
    comparison_norm_trainable = cfg['comparison_norm_trainable']
    num_classes = cfg['num_classes']
    max_length = cfg['max_length']
    lr = cfg['learning_rate']
    embedding_model_class = cfg['embedding_model_class']

    if not emb_trainable:
        comparison_norm_trainable = True

        print('[INFO] Training Comparator')
    else:
        print('[INFO] Training Char Embedding')

    print('[CFG] emb_trainable: ', emb_trainable)
    print('[CFG] comparison_norm_trainable: ', comparison_norm_trainable)
    print('[CFG] optimizer : ', cfg['optimizer'])
    print('[CFG] learning_rate : ', lr)

    conv_activation = 'swish'

    dense_output_1_lv1_size = 11
    dense_output_2_lv1_size = 7
    dense_output_3_lv1_size = 7
    dense_output_1_lv2_size = 3
    dense_output_2_lv2_size = 4
    dense_activation = 'swish'

    char_embedding_layer = CharEmbeddingV5(
        num_classes=num_classes,
        max_length=max_length,
        trainable=emb_trainable,
        name='char_embedding'
    )
    assert isinstance(char_embedding_layer, embedding_model_class), f'{char_embedding_layer} is wrong embedding model!'

    # Encoder 1
    input_1 = Input(shape=(max_length,), dtype='int32', name='input_1')
    embedded_1 = char_embedding_layer(input_1)

    input_2 = Input(shape=(max_length,), dtype='int32', name='input_2')
    embedded_2 = char_embedding_layer(input_2)

    convoluted_11 = Conv1D(5, 3, 1, activation=conv_activation, padding='same', name='convoluted_11')(embedded_1)
    convoluted_21 = Conv1D(5, 5, 1, activation=conv_activation, padding='same', name='convoluted_21')(embedded_1)
    convoluted_31 = Conv1D(5, 7, 1, activation=conv_activation, padding='same', name='convoluted_31')(embedded_1)
    convoluted_41 = Conv1D(5, 11, 1, activation=conv_activation, padding='same', name='convoluted_41')(embedded_1)

    merged_conv_12 = Maximum(name='merged_conv_12')([convoluted_11, convoluted_21, convoluted_31, convoluted_41])
    convoluted_12 = Conv1D(11, 3, 1, activation=conv_activation, padding='same', name='convoluted_12')(merged_conv_12)
    pooled_12 = AveragePooling1D(3, 3, padding='same', name='pooled_12')(convoluted_12)
    convoluted_22 = Conv1D(11, 3, 1, activation=conv_activation, padding='same', name='convoluted_22')(pooled_12)
    pooled_22 = MaxPool1D(11, 1, padding='same', name='pooled_22')(convoluted_22)

    merged_conv_13 = Maximum(name='merged_conv_13')([pooled_12, pooled_22])
    convoluted_13 = Conv1D(17, 3, 3, activation=conv_activation, padding='same', name='convoluted_13')(merged_conv_13)
    pooled_13 = AveragePooling1D(3, 3, padding='same', name='pooled_13')(convoluted_13)

    unit_convoluted_1 = Conv1D(1, 1, 1, activation=conv_activation, padding='same', name='unit_convoluted_1')(pooled_13)
    squeezed_1 = tf.squeeze(unit_convoluted_1, axis=2, name='squeezed_1')

    # Encoder 2

    rnn_1 = GRU(10, return_sequences=True, name='rnn_1')(embedded_2)
    rnn_2 = GRU(10, return_sequences=True, name='rnn_2')(rnn_1)
    rnn_3 = GRU(10, return_sequences=False, name='rnn_3')(rnn_2)

    # distance
    norm_1 = LayerNormalization(trainable=comparison_norm_trainable, name='norm_1')(squeezed_1)
    norm_2 = LayerNormalization(trainable=comparison_norm_trainable, name='norm_2')(rnn_3)

    dense_output_11_lv1 = Dense(dense_output_1_lv1_size, name='dense_output_11_lv1', kernel_regularizer='l2', activation=dense_activation)(norm_1)
    dense_output_11_lv2 = Dense(dense_output_1_lv2_size, name='dense_output_11_lv2', kernel_regularizer='l2', activation=dense_activation)(dense_output_11_lv1)
    dense_output_21_lv1 = Dense(dense_output_2_lv1_size, name='dense_output_21_lv1', kernel_regularizer='l2', activation=dense_activation)(norm_1)
    dense_output_21_lv2 = Dense(dense_output_2_lv2_size, name='dense_output_21_lv2', kernel_regularizer='l2', activation=dense_activation)(dense_output_21_lv1)
    dense_output_31_lv1 = Dense(dense_output_3_lv1_size, name='dense_output_31_lv1', kernel_regularizer='l2', activation=dense_activation)(norm_1)

    dense_output_12_lv1 = Dense(dense_output_1_lv1_size, name='dense_output_12_lv1', kernel_regularizer='l2', activation=dense_activation)(norm_2)
    dense_output_12_lv2 = Dense(dense_output_1_lv2_size, name='dense_output_12_lv2', kernel_regularizer='l2', activation=dense_activation)(dense_output_12_lv1)
    dense_output_22_lv1 = Dense(dense_output_2_lv1_size, name='dense_output_22_lv1', kernel_regularizer='l2', activation=dense_activation)(norm_2)
    dense_output_22_lv2 = Dense(dense_output_2_lv2_size, name='dense_output_22_lv2', kernel_regularizer='l2', activation=dense_activation)(dense_output_22_lv1)
    dense_output_32_lv1 = Dense(dense_output_3_lv1_size, name='dense_output_32_lv1', kernel_regularizer='l2', activation=dense_activation)(norm_2)

    distance_1 = Dot(axes=1, normalize=True, name='distance_1')([dense_output_11_lv2, dense_output_12_lv2])
    distance_2 = Dot(axes=1, normalize=True, name='distance_2')([dense_output_21_lv2, dense_output_22_lv2])
    distance_3 = Dot(axes=1, normalize=True, name='distance_3')([dense_output_31_lv1, dense_output_32_lv1])

    model = Model(inputs=[input_1, input_2], outputs=[distance_1, distance_2, distance_3])

    optimizer = cfg['optimizer'](learning_rate=lr)

    model.compile(
        loss='mae',
        optimizer=optimizer,
    )

    return model
