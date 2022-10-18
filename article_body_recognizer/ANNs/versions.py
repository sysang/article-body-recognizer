import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adagrad, Nadam
from tensorflow.keras.optimizers.schedules import InverseTimeDecay
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM, Bidirectional, Dropout, LayerNormalization, BatchNormalization
from tensorflow.keras.layers import concatenate, Reshape, SpatialDropout1D, Conv1D, Flatten, AveragePooling1D, MaxPool1D, Average, Maximum, Multiply, Add
from tensorflow.keras.models import Model, Sequential

from tensorflow import config as config
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import TensorBoard, Callback

from scraping.models import CharEmbeddingV411

def HierarchyV3_090702(cfg, learning_rate=None):

  print('[ATTENTION!] Models version: v3.9.7.2')

  num_categories = cfg['num_categories']
  num_classes = cfg['num_classes']
  max_length = cfg['max_length']
  chunk_length = cfg['chunk_length']
  dim_embeddings = cfg['dim_embeddings']
  lr = learning_rate if learning_rate else cfg['learning_rate']
  decay_steps = cfg['decay_steps']
  decay_rate = cfg['decay_rate']

  # dense_compressed_1_size = 32
  # dense_compressed_2_size = 16
  # dense_compressed_3_size = 8
  dense_compressed_4_size = 3
  dense_compressed_activation = 'sigmoid'

  dense_1_dropout_ratio = 0.25
  dense_1_size = 128
  dense_2_dropout_ratio = 0.15
  dense_2_size = 64
  dense_3_dropout_ratio = 0.12
  dense_3_size = 32
  dense_4_dropout_ratio = 0.12
  dense_4_size = 16
  dense_5_dropout_ratio = 0.10
  dense_5_size = 8
  dense_activation = 'swish'

  detail_content_dendrop_ratio = 0.25
  detail_content_dense_size_1 = 32
  detail_content_dense_size_2 = 16
  detail_content_dense_activation = 'swish'

  detail_title_dendrop_ratio = 0.25
  detail_title_dense_size_1 = 32
  detail_title_dense_size_2 = 16
  detail_title_dense_activation = 'swish'

  abstract_content_dendrop_ratio = 0.15
  abstract_content_dense_size = 16
  abstract_content_dense_activation = 'swish'


  assert max_length % chunk_length == 0, 'max_length is no multiple of chunk_length'
  num_chunks = int(max_length / chunk_length)

  # embedded_layer = Embedding(num_classes, dim_embeddings, input_length=chunk_length, trainable=True, name='embedding')
  # dense_compressed_layer_4 = Dense(dense_compressed_4_size, name='dense_compressed_4', trainable=True, activation=dense_compressed_activation)

  input = Input(shape=(num_chunks, chunk_length), dtype='int32', name='input')

  input_by_node = tf.transpose(input, [1, 0, 2])
  char_sequence_encoded = []
  for i in range(num_chunks):
    # embedded = embedded_layer(input_by_node[i])
    # dense_compressed_4 = dense_compressed_layer_4(embedded)
    one_hot = tf.one_hot(input_by_node[i], num_classes)
    char_sequence_encoded.append(one_hot)

  concatenated_0 = concatenate(char_sequence_encoded, axis=1, name='concatenated_0')

  # block 1
  convoluted_11 = Conv1D(16, 3, 3, padding='same', name='convoluted_11')(concatenated_0)
  # pooled_11 = AveragePooling1D(3, 3, padding='same', name='pooled_11')(convoluted_11)
  convoluted_21 = Conv1D(16, 5, 3, padding='same', name='convoluted_21')(concatenated_0)
  # pooled_21 = AveragePooling1D(5, 3, padding='same', name='pooled_21')(convoluted_21)
  convoluted_31 = Conv1D(16, 7, 3, padding='same', name='convoluted_31')(concatenated_0)
  # pooled_31 = AveragePooling1D(7, 3, padding='same', name='pooled_31')(convoluted_31)
  convoluted_41 = Conv1D(16, 11, 3, padding='same', name='convoluted_41')(concatenated_0)
  # pooled_41 = AveragePooling1D(11, 3, padding='same', name='pooled_41')(convoluted_41)

  # block 2
  # merged_conv_12 = Average(name='merged_conv_12')([pooled_11, pooled_21, pooled_31, pooled_41])
  merged_conv_12 = Average(name='merged_conv_12')([convoluted_11, convoluted_21, convoluted_31, convoluted_41])
  convoluted_12 = Conv1D(32, 2, 2, padding='same', name='convoluted_12')(merged_conv_12)
  # pooled_12 = MaxPool1D(2, 2, padding='same', name='pooled_12')(convoluted_12)
  convoluted_22 = Conv1D(32, 3, 2, padding='same', name='convoluted_22')(merged_conv_12)
  # pooled_22 = MaxPool1D(3, 2, padding='same', name='pooled_22')(convoluted_22)
  convoluted_32 = Conv1D(32, 5, 2, padding='same', name='convoluted_32')(merged_conv_12)
  # pooled_32 = MaxPool1D(5, 2, padding='same', name='pooled_32')(convoluted_32)

  # block 3
  # merged_conv_13 = Maximum(name='merged_conv_13')([pooled_12, pooled_22, pooled_32])
  merged_conv_13 = Maximum(name='merged_conv_13')([convoluted_12, convoluted_22, convoluted_32])
  convoluted_13 = Conv1D(64, 3, 1, padding='same', name='convoluted_13')(merged_conv_13)
  pooled_13 = AveragePooling1D(3, 3, padding='same', name='pooled_13')(convoluted_13)
  convoluted_23 = Conv1D(128, 3, 5, padding='same', name='convoluted_23')(pooled_13)
  pooled_23 = AveragePooling1D(3, 5, padding='same', name='pooled_23')(convoluted_23)
  convoluted_33 = Conv1D(128, 3, 5, padding='same', name='convoluted_33')(pooled_23)
  pooled_33 = AveragePooling1D(3, 5, padding='same', name='pooled_33')(convoluted_33)

  unit_convoluted_23 = Conv1D(1, 1, 1, padding='same', name='unit_convoluted_23')(pooled_23)
  squeezed_23 = tf.squeeze(unit_convoluted_23, axis=2)

  unit_convoluted_33 = Conv1D(1, 1, 1, padding='same', name='unit_convoluted_33')(pooled_33)
  squeezed_33 = tf.squeeze(unit_convoluted_33, axis=2)

  # merged_conv_43 = Maximum(name='merged_conv_43')([pooled_12, pooled_22, pooled_32])
  merged_conv_43 = Multiply(name='merged_conv_43')([convoluted_12, convoluted_22, convoluted_32])
  convoluted_43 = Conv1D(64, 3, 3, padding='same', name='convoluted_43')(merged_conv_43)
  pooled_43 = AveragePooling1D(3, 3, padding='same', name='pooled_43')(convoluted_43)
  convoluted_53 = Conv1D(128, 3, 5, padding='same', name='convoluted_53')(pooled_43)
  pooled_53 = AveragePooling1D(3, 5, padding='same', name='pooled_53')(convoluted_53)
  convoluted_63 = Conv1D(128, 3, 5, padding='same', name='convoluted_63')(pooled_53)
  pooled_63 = AveragePooling1D(3, 5, padding='same', name='pooled_63')(convoluted_63)

  unit_convoluted_63 = Conv1D(1, 1, 1, padding='same', name='unit_convoluted_63')(pooled_63)
  squeezed_63 = tf.squeeze(unit_convoluted_63, axis=2)

  # block 4
  # convoluted_14 = Conv1D(4, 3, 3, padding='same', name='convoluted_14')(concatenated_0)
  # pooled_14 = AveragePooling1D(3, 3, padding='same', name='pooled_14')(convoluted_14)
  # convoluted_24 = Conv1D(4, 5, 3, padding='same', name='convoluted_24')(concatenated_0)
  # pooled_24 = AveragePooling1D(5, 3, padding='same', name='pooled_24')(convoluted_24)

  # block 5
  # merged_conv_15 = Maximum(name='merged_conv_15')([pooled_14, pooled_24])
  merged_conv_15 = Maximum(name='merged_conv_15')([convoluted_11, convoluted_21])
  convoluted_25 = Conv1D(32, 3, 1, padding='same', name='convoluted_25')(merged_conv_15)
  pooled_25 = AveragePooling1D(3, 3, padding='same', name='pooled_25')(convoluted_25)
  convoluted_35 = Conv1D(64, 3, 5, padding='same', name='convoluted_35')(pooled_25)
  pooled_35 = AveragePooling1D(3, 5, padding='same', name='pooled_35')(convoluted_35)
  convoluted_45 = Conv1D(128, 3, 5, padding='same', name='convoluted_45')(pooled_35)
  pooled_45 = AveragePooling1D(3, 5, padding='same', name='pooled_45')(convoluted_45)

  unit_convoluted_35 = Conv1D(1, 1, 1, padding='same', name='unit_convoluted_35')(pooled_35)
  squeezed_35 = tf.squeeze(unit_convoluted_35, axis=2)

  unit_convoluted_45 = Conv1D(1, 1, 1, padding='same', name='unit_convoluted_45')(pooled_45)
  squeezed_45 = tf.squeeze(unit_convoluted_45, axis=2)

  # block 6
  merged_convs = concatenate([squeezed_23, squeezed_33, squeezed_63, squeezed_35, squeezed_45], name='merged_convs')

  dense_1_dropout = Dropout(rate=dense_1_dropout_ratio, name='dense_1_dropout')(merged_convs)
  dense_1 = Dense(dense_1_size, name='dense_1', activation=dense_activation)(dense_1_dropout)
  dense_2_dropout = Dropout(rate=dense_2_dropout_ratio, name='dense_2_dropout')(dense_1)
  dense_2 = Dense(dense_2_size, name='dense_2', activation=dense_activation)(dense_2_dropout)
  dense_3_dropout = Dropout(rate=dense_3_dropout_ratio, name='dense_3_dropout')(dense_2)
  dense_3 = Dense(dense_3_size, name='dense_3', activation=dense_activation)(dense_3_dropout)
  dense_4_dropout = Dropout(rate=dense_4_dropout_ratio, name='dense_4_dropout')(dense_3)
  content_dense_4 = Dense(dense_4_size, name='content_dense_4', activation=dense_activation)(dense_4_dropout)
  title_dense_4 = Dense(dense_4_size, name='title_dense_4', activation=dense_activation)(dense_4_dropout)
  dense_5_dropout = Dropout(rate=dense_5_dropout_ratio, name='dense_5_dropout')(title_dense_4)
  dense_5 = Dense(dense_5_size, name='dense_5', activation=dense_activation)(dense_5_dropout)

  concatenated_1 = concatenate([dense_1, dense_2, dense_3, content_dense_4, dense_5], name='concatenated_1')
  detail_content_dendrop_1 = Dropout(rate=detail_content_dendrop_ratio, name='detail_content_dendrop_1')(concatenated_1)
  detail_content_dense_1 = Dense(detail_content_dense_size_1, name='detail_content_dense_1', activation=detail_content_dense_activation)(detail_content_dendrop_1)
  detail_content_dendrop_2 = Dropout(rate=detail_content_dendrop_ratio, name='detail_content_dendrop_2')(detail_content_dense_1)
  detail_content_dense_2 = Dense(detail_content_dense_size_2, name='detail_content_dense_2', activation=detail_content_dense_activation)(detail_content_dendrop_2)
  detail_content_output = Dense(num_categories, name='detail_content_output', activation='softmax')(detail_content_dense_2)

  concatenated_2 = concatenate([dense_1, dense_2, dense_3, title_dense_4], name='concatenated_2')
  detail_title_dendrop_1 = Dropout(rate=detail_title_dendrop_ratio, name='detail_title_dendrop_1')(concatenated_2)
  detail_title_dense_1 = Dense(detail_title_dense_size_1, name='detail_title_dense_1', activation=detail_title_dense_activation)(detail_title_dendrop_1)
  detail_title_dendrop_2 = Dropout(rate=detail_title_dendrop_ratio, name='detail_title_dendrop_2')(detail_title_dense_1)
  detail_title_dense_2 = Dense(detail_title_dense_size_2, name='detail_title_dense_2', activation=detail_title_dense_activation)(detail_title_dendrop_2)
  detail_title_output = Dense(num_categories, name='detail_title_output', activation='softmax')(detail_title_dense_2)

  merged_abstract_content = concatenate([content_dense_4, dense_5], name='merged_abstract_content')
  abstract_content_dendrop = Dropout(rate=abstract_content_dendrop_ratio, name='abstract_content_dendrop')(merged_abstract_content)
  abstract_content_dense = Dense(abstract_content_dense_size, name='abstract_content_dense', activation=abstract_content_dense_activation)(abstract_content_dendrop)
  abstract_content_output = Dense(num_categories, name='abstract_content_output', activation='softmax')(abstract_content_dense)

  model = Model(inputs=[input], outputs=[abstract_content_output, detail_content_output, detail_title_output])


  optimizer = RMSprop(learning_rate=lr)

  # learning_rate_fn = InverseTimeDecay(
  #     initial_learning_rate=lr,
  #     decay_steps=decay_steps,
  #     decay_rate=decay_rate,
  #     staircase=True
  #   )
  # optimizer = Nadam(learning_rate=lr)
  # optimizer = Adam(learning_rate=learning_rate_fn)
  # optimizer = RMSprop(learning_rate=learning_rate_fn)
  # optimizer = Adagrad(learning_rate=learning_rate_fn)

  model.compile(
      loss='categorical_crossentropy',
      optimizer=optimizer,
      metrics=['accuracy'],
    )

  return model


def HierarchyV3_080601(cfg, dim_embeddings=8, learning_rate=5e-4):

  print('[ATTENTION!] Models version: v3.8.6.1')

  num_categories = cfg['num_categories']
  num_classes = cfg['num_classes']
  max_length = cfg['max_length']
  chunk_length = cfg['chunk_length']
  dem = dim_embeddings if dim_embeddings else cfg['dim_embeddings']
  lr = learning_rate if learning_rate else cfg['learning_rate']

  # dense_compressed_1_size = 32
  # dense_compressed_2_size = 16
  # dense_compressed_3_size = 8
  # dense_compressed_4_size = 4
  # dense_compressed_activation = 'relu'

  dense_1_dropout_ratio = 0.15
  dense_1_size = 128
  dense_2_dropout_ratio = 0.12
  dense_2_size = 64
  dense_3_dropout_ratio = 0.12
  dense_3_size = 32
  dense_4_dropout_ratio = 0.10
  dense_4_size = 16
  dense_5_dropout_ratio = 0.10
  dense_5_size = 8

  detail_content_dendrop_ratio = 0.15
  detail_content_dense_size_1 = 32
  detail_content_dense_size_2 = 16
  detail_content_dense_activation = 'swish'

  detail_title_dendrop_ratio = 0.15
  detail_title_dense_size_1 = 32
  detail_title_dense_size_2 = 16
  detail_title_dense_activation = 'swish'

  abstract_content_dendrop_ratio = 0.10
  abstract_content_dense_size = 16
  abstract_content_dense_activation = 'swish'


  assert max_length % chunk_length == 0, 'max_length is no multiple of chunk_length'
  num_chunks = int(max_length / chunk_length)

  embedded_layer = Embedding(num_classes, dim_embeddings, input_length=chunk_length, trainable=True, name='embedding')

  input = Input(shape=(num_chunks, chunk_length), dtype='int32', name='input')

  input_by_node = tf.transpose(input, [1, 0, 2])
  char_sequence_encoded = []
  for i in range(num_chunks):
    embedded = embedded_layer(input_by_node[i])
    char_sequence_encoded.append(embedded)

  concatenated_0 = concatenate(char_sequence_encoded, axis=1, name='concatenated_0')

  # layer 1
  convoluted_11 = Conv1D(107, 3, 1, padding='same', name='convoluted_11')(concatenated_0)
  pooled_11 = AveragePooling1D(3, 1, padding='same', name='pooled_11')(convoluted_11)
  convoluted_21 = Conv1D(107, 5, 1, padding='same', name='convoluted_21')(concatenated_0)
  pooled_21 = AveragePooling1D(5, 1, padding='same', name='pooled_21')(convoluted_21)
  convoluted_31 = Conv1D(107, 7, 1, padding='same', name='convoluted_31')(concatenated_0)
  pooled_31 = AveragePooling1D(7, 1, padding='same', name='pooled_31')(convoluted_31)
  convoluted_41 = Conv1D(107, 9, 1, padding='same', name='convoluted_41')(concatenated_0)
  pooled_41 = AveragePooling1D(9, 1, padding='same', name='pooled_41')(convoluted_41)

  # layer 2
  merged_conv_12 = Add(name='merged_conv_12')([pooled_11, pooled_21, pooled_31, pooled_41])
  convoluted_12 = Conv1D(71, 2, 1, padding='same', name='convoluted_12')(merged_conv_12)
  pooled_12 = AveragePooling1D(2, 1, padding='same', name='pooled_12')(convoluted_12)
  convoluted_22 = Conv1D(71, 3, 1, padding='same', name='convoluted_22')(merged_conv_12)
  pooled_22 = AveragePooling1D(3, 1, padding='same', name='pooled_22')(convoluted_22)
  convoluted_32 = Conv1D(71, 5, 1, padding='same', name='convoluted_32')(merged_conv_12)
  pooled_32 = AveragePooling1D(5, 1, padding='same', name='pooled_32')(convoluted_32)

  # layer 3
  merged_conv_13 = Add(name='merged_conv_13')([pooled_12, pooled_22, pooled_32])
  convoluted_13 = Conv1D(19, 1, 3, padding='same', name='convoluted_13')(merged_conv_13)
  pooled_13 = AveragePooling1D(1, 3, padding='same', name='pooled_13')(convoluted_13)
  convoluted_23 = Conv1D(19, 2, 3, padding='same', name='convoluted_23')(merged_conv_13)
  pooled_23 = AveragePooling1D(2, 3, padding='same', name='pooled_23')(convoluted_23)
  convoluted_33 = Conv1D(19, 3, 3, padding='same', name='convoluted_33')(merged_conv_13)
  pooled_33 = AveragePooling1D(3, 3, padding='same', name='pooled_33')(convoluted_33)

  # layer 4
  merged_conv_14 = Average(name='merged_conv_14')([pooled_13, pooled_23, pooled_33])
  convoluted_14 = Conv1D(3, 1, 2, padding='same', name='convoluted_14')(merged_conv_14)
  pooled_14 = AveragePooling1D(1, 2, padding='same', name='pooled_14')(convoluted_14)
  convoluted_24 = Conv1D(3, 2, 2, padding='same', name='convoluted_24')(merged_conv_14)
  pooled_24 = AveragePooling1D(2, 2, padding='same', name='pooled_24')(convoluted_24)
  convoluted_34 = Conv1D(3, 3, 2, padding='same', name='convoluted_34')(merged_conv_14)
  pooled_34 = AveragePooling1D(3, 2, padding='same', name='pooled_34')(convoluted_34)

  # layer 5
  merged_conv_15 = Average(name='merged_conv_15')([pooled_14, pooled_24, pooled_34])
  convoluted_15 = Conv1D(1, 1, 1, padding='same', name='convoluted_15')(merged_conv_15)

  flatten = Flatten()(convoluted_15)

  dense_1_dropout = Dropout(rate=dense_1_dropout_ratio, name='dense_1_dropout')(flatten)
  dense_1 = Dense(dense_1_size, name='dense_1', activation='swish')(dense_1_dropout)
  dense_2_dropout = Dropout(rate=dense_2_dropout_ratio, name='dense_2_dropout')(dense_1)
  dense_2 = Dense(dense_2_size, name='dense_2', activation='swish')(dense_2_dropout)
  dense_3_dropout = Dropout(rate=dense_3_dropout_ratio, name='dense_3_dropout')(dense_2)
  dense_3 = Dense(dense_3_size, name='dense_3', activation='relu')(dense_3_dropout)
  dense_4_dropout = Dropout(rate=dense_4_dropout_ratio, name='dense_4_dropout')(dense_3)
  content_dense_4 = Dense(dense_4_size, name='content_dense_4', activation='swish')(dense_4_dropout)
  title_dense_4 = Dense(dense_4_size, name='title_dense_4', activation='swish')(dense_4_dropout)
  dense_5_dropout = Dropout(rate=dense_5_dropout_ratio, name='dense_5_dropout')(dense_3)
  dense_5 = Dense(dense_5_size, name='dense_5', activation='relu')(dense_5_dropout)

  concatenated_1 = concatenate([dense_1, dense_2, dense_3, content_dense_4, dense_5], name='concatenated_1')
  detail_content_dendrop_1 = Dropout(rate=detail_content_dendrop_ratio, name='detail_content_dendrop_1')(concatenated_1)
  detail_content_dense_1 = Dense(detail_content_dense_size_1, name='detail_content_dense_1', activation=detail_content_dense_activation)(detail_content_dendrop_1)
  detail_content_dendrop_2 = Dropout(rate=detail_content_dendrop_ratio, name='detail_content_dendrop_2')(detail_content_dense_1)
  detail_content_dense_2 = Dense(detail_content_dense_size_2, name='detail_content_dense_2', activation=detail_content_dense_activation)(detail_content_dendrop_2)
  detail_content_output = Dense(num_categories, name='detail_content_output', activation='softmax')(detail_content_dense_2)

  concatenated_2 = concatenate([dense_1, dense_2, dense_3, title_dense_4, dense_5], name='concatenated_2')
  detail_title_dendrop_1 = Dropout(rate=detail_title_dendrop_ratio, name='detail_title_dendrop_1')(concatenated_2)
  detail_title_dense_1 = Dense(detail_title_dense_size_1, name='detail_title_dense_1', activation=detail_title_dense_activation)(detail_title_dendrop_1)
  detail_title_dendrop_2 = Dropout(rate=detail_title_dendrop_ratio, name='detail_title_dendrop_2')(detail_title_dense_1)
  detail_title_dense_2 = Dense(detail_title_dense_size_2, name='detail_title_dense_2', activation=detail_title_dense_activation)(detail_title_dendrop_2)
  detail_title_output = Dense(num_categories, name='detail_title_output', activation='softmax')(detail_title_dense_2)

  merged_abstract_content = concatenate([content_dense_4, title_dense_4], name='merged_abstract_content')
  abstract_content_dendrop = Dropout(rate=abstract_content_dendrop_ratio, name='abstract_content_dendrop')(merged_abstract_content)
  abstract_content_dense = Dense(abstract_content_dense_size, name='abstract_content_dense', activation=abstract_content_dense_activation)(abstract_content_dendrop)
  abstract_content_output = Dense(num_categories, name='abstract_content_output', activation='softmax')(abstract_content_dense)

  model = Model(inputs=[input], outputs=[abstract_content_output, detail_content_output, detail_title_output])

  optimizer = RMSprop(learning_rate=lr)

  model.compile(
      loss='categorical_crossentropy',
      optimizer=optimizer,
      metrics=['accuracy'],
    )

  return model


def HierarchyV3_090200(cfg, learning_rate=2e-3):

  print('[ATTENTION!] Models version: v3.8.6.1')

  num_categories = cfg['num_categories']
  num_classes = cfg['num_classes']
  max_length = cfg['max_length']
  lr = learning_rate if learning_rate else cfg['learning_rate']

  # dense_compressed_1_size = 32
  # dense_compressed_2_size = 16
  # dense_compressed_3_size = 8
  dim_embeddings = 8
  dense_compressed_4_size = 4
  dense_compressed_activation = 'sigmoid'

  # dense_1_dropout_ratio = 0.08
  # dense_1_size = 64
  dense_2_dropout_ratio = 0.15
  dense_2_size = 32
  dense_3_dropout_ratio = 0.12
  dense_3_size = 16
  dense_4_dropout_ratio = 0.12
  dense_4_size = 8
  # dense_5_dropout_ratio = 0.03
  # dense_5_size = 4

  detail_content_dendrop_ratio = 0.20
  detail_content_dense_size_1 = 16
  detail_content_dense_size_2 = 8
  detail_content_dense_activation = 'swish'

  detail_title_dendrop_ratio = 0.25
  detail_title_dense_size_1 = 16
  detail_title_dense_size_2 = 8
  detail_title_dense_activation = 'swish'

  abstract_content_dendrop_ratio = 0.15
  abstract_content_dense_size = 8
  abstract_content_dense_activation = 'swish'

  embedded_layer = Embedding(num_classes, dim_embeddings, input_length=max_length, trainable=True, name='embedding')
  dense_compressed_layer_4 = Dense(dense_compressed_4_size, name='dense_compressed_4', trainable=True, activation=dense_compressed_activation)

  input = Input(shape=(max_length,), dtype='int32', name='input')

  embedded = embedded_layer(input)
  dense_compressed_4 = dense_compressed_layer_4(embedded)

  # block 1
  convoluted_11 = Conv1D(16, 3, 1, padding='same', name='convoluted_11')(dense_compressed_4)
  # pooled_11 = AveragePooling1D(3, 1, padding='same', name='pooled_11')(convoluted_11)
  convoluted_21 = Conv1D(16, 5, 1, padding='same', name='convoluted_21')(dense_compressed_4)
  # pooled_21 = AveragePooling1D(5, 1, padding='same', name='pooled_21')(convoluted_21)
  convoluted_31 = Conv1D(16, 7, 1, padding='same', name='convoluted_31')(dense_compressed_4)
  # pooled_31 = AveragePooling1D(7, 1, padding='same', name='pooled_31')(convoluted_31)
  convoluted_41 = Conv1D(16, 11, 1, padding='same', name='convoluted_41')(dense_compressed_4)
  # pooled_41 = AveragePooling1D(11, 1, padding='same', name='pooled_41')(convoluted_41)

  # block 2
  merged_conv_12 = Average(name='merged_conv_12')([convoluted_11, convoluted_21, convoluted_31, convoluted_41])
  # merged_conv_12 = Average(name='merged_conv_12')([pooled_11, pooled_21, pooled_31, pooled_41])
  convoluted_12 = Conv1D(16, 2, 1, padding='same', name='convoluted_12')(merged_conv_12)
  # pooled_12 = AveragePooling1D(2, 1, padding='same', name='pooled_12')(convoluted_12)
  convoluted_22 = Conv1D(16, 3, 1, padding='same', name='convoluted_22')(merged_conv_12)
  # pooled_22 = AveragePooling1D(3, 1, padding='same', name='pooled_22')(convoluted_22)
  convoluted_32 = Conv1D(16, 5, 1, padding='same', name='convoluted_32')(merged_conv_12)
  # pooled_32 = AveragePooling1D(5, 1, padding='same', name='pooled_32')(convoluted_32)

  # block 3
  merged_conv_13 = Average(name='merged_conv_13')([convoluted_12, convoluted_22, convoluted_32])
  # merged_conv_13 = Average(name='merged_conv_13')([pooled_12, pooled_22, pooled_32])
  convoluted_13 = Conv1D(32, 3, 3, padding='same', name='convoluted_13')(merged_conv_13)
  pooled_13 = AveragePooling1D(3, 3, padding='same', name='pooled_13')(convoluted_13)
  convoluted_23 = Conv1D(64, 3, 3, padding='same', name='convoluted_23')(pooled_13)
  pooled_23 = AveragePooling1D(3, 3, padding='same', name='pooled_23')(convoluted_23)
  convoluted_33 = Conv1D(1, 1, 1, padding='same', name='convoluted_33')(pooled_23)
  squeezed_43 = tf.squeeze(convoluted_33, axis=2)

  # block 4
  convoluted_14 = Conv1D(4, 5, 1, padding='same', name='convoluted_14')(dense_compressed_4)
  # pooled_14 = AveragePooling1D(5, 1, padding='same', name='pooled_14')(convoluted_14)
  convoluted_24 = Conv1D(8, 3, 3, padding='same', name='convoluted_24')(convoluted_14)
  pooled_24 = MaxPool1D(3, 3, padding='same', name='pooled_24')(convoluted_24)
  convoluted_34 = Conv1D(16, 3, 3, padding='same', name='convoluted_34')(pooled_24)
  pooled_34 = MaxPool1D(3, 3, padding='same', name='pooled_34')(convoluted_34)
  convoluted_44 = Conv1D(1, 1, 1, padding='same', name='convoluted_44')(pooled_34)
  squeezed_54 = tf.squeeze(convoluted_44, axis=2)

  # block 5
  merged_15 = concatenate([squeezed_43, squeezed_54], name='merged_15')

  # dense_1_dropout = Dropout(rate=dense_1_dropout_ratio, name='dense_1_dropout')(flatten)
  # dense_1 = Dense(dense_1_size, name='dense_1', activation='swish')(dense_1_dropout)
  dense_2_dropout = Dropout(rate=dense_2_dropout_ratio, name='dense_2_dropout')(merged_15)
  dense_2 = Dense(dense_2_size, name='dense_2', activation='swish')(dense_2_dropout)
  dense_3_dropout = Dropout(rate=dense_3_dropout_ratio, name='dense_3_dropout')(dense_2)
  dense_3 = Dense(dense_3_size, name='dense_3', activation='relu')(dense_3_dropout)
  dense_4_dropout = Dropout(rate=dense_4_dropout_ratio, name='dense_4_dropout')(dense_3)
  content_dense_4 = Dense(dense_4_size, name='content_dense_4', activation='swish')(dense_4_dropout)
  title_dense_4 = Dense(dense_4_size, name='title_dense_4', activation='swish')(dense_4_dropout)
  # dense_5_dropout = Dropout(rate=dense_5_dropout_ratio, name='dense_5_dropout')(dense_3)
  # dense_5 = Dense(dense_5_size, name='dense_5', activation='relu')(dense_5_dropout)

  concatenated_1 = concatenate([dense_2, dense_3, content_dense_4], name='concatenated_1')
  detail_content_dendrop_1 = Dropout(rate=detail_content_dendrop_ratio, name='detail_content_dendrop_1')(concatenated_1)
  detail_content_dense_1 = Dense(detail_content_dense_size_1, name='detail_content_dense_1', activation=detail_content_dense_activation)(detail_content_dendrop_1)
  detail_content_dendrop_2 = Dropout(rate=detail_content_dendrop_ratio, name='detail_content_dendrop_2')(detail_content_dense_1)
  detail_content_dense_2 = Dense(detail_content_dense_size_2, name='detail_content_dense_2', activation=detail_content_dense_activation)(detail_content_dendrop_2)
  detail_content_output = Dense(num_categories, name='detail_content_output', activation='softmax')(detail_content_dense_2)

  concatenated_2 = concatenate([dense_2, dense_3, title_dense_4], name='concatenated_2')
  detail_title_dendrop_1 = Dropout(rate=detail_title_dendrop_ratio, name='detail_title_dendrop_1')(concatenated_2)
  detail_title_dense_1 = Dense(detail_title_dense_size_1, name='detail_title_dense_1', activation=detail_title_dense_activation)(detail_title_dendrop_1)
  detail_title_dendrop_2 = Dropout(rate=detail_title_dendrop_ratio, name='detail_title_dendrop_2')(detail_title_dense_1)
  detail_title_dense_2 = Dense(detail_title_dense_size_2, name='detail_title_dense_2', activation=detail_title_dense_activation)(detail_title_dendrop_2)
  detail_title_output = Dense(num_categories, name='detail_title_output', activation='softmax')(detail_title_dense_2)

  merged_abstract_content = concatenate([content_dense_4, title_dense_4], name='merged_abstract_content')
  abstract_content_dendrop = Dropout(rate=abstract_content_dendrop_ratio, name='abstract_content_dendrop')(merged_abstract_content)
  abstract_content_dense = Dense(abstract_content_dense_size, name='abstract_content_dense', activation=abstract_content_dense_activation)(abstract_content_dendrop)
  abstract_content_output = Dense(num_categories, name='abstract_content_output', activation='softmax')(abstract_content_dense)

  model = Model(inputs=[input], outputs=[abstract_content_output, detail_content_output, detail_title_output])


  optimizer = Adam(learning_rate=lr)

  # learning_rate_fn = InverseTimeDecay(
  #     initial_learning_rate=lr,
  #     decay_steps=decay_steps,
  #     decay_rate=decay_rate,
  #     staircase=True
  #   )
  # optimizer = Nadam(learning_rate=lr)
  # optimizer = Adam(learning_rate=learning_rate_fn)
  # optimizer = RMSprop(learning_rate=learning_rate_fn)
  # optimizer = Adagrad(learning_rate=learning_rate_fn)

  model.compile(
      loss='categorical_crossentropy',
      optimizer=optimizer,
      metrics=['accuracy'],
    )

  return model


def HierarchyV03x091103(cfg, learning_rate=5e-4):
  num_categories = cfg['num_categories']
  num_classes = cfg['num_classes']
  max_length = cfg['max_length']
  lr = learning_rate if learning_rate else cfg['learning_rate']

  dense_1_dropout_ratio = 0.15
  dense_1_size = 79
  dense_2_dropout_ratio = 0.15
  dense_2_size = 41
  dense_3_dropout_ratio = 0.12
  dense_3_size = 19
  dense_4_dropout_ratio = 0.10
  dense_4_size = 7
  dense_5_dropout_ratio = 0.05
  dense_5_size = 3
  dense_activation = 'swish'

  detail_content_dendrop_ratio = 0.15
  detail_content_dense_size_2 = 29
  detail_content_dense_activation = 'tanh'
  detail_content_outdrop_ratio = 0.15

  detail_title_dendrop_ratio = 0.15
  detail_title_dense_size_2 = 29
  detail_title_dense_activation = 'tanh'
  detail_title_outdrop_ratio = 0.15

  abstract_content_outdrop_ratio = 0.25

  char_embedding_layer = CharEmbeddingV01x02(cfg, trainable=False, name='char_embedding')

  inputs = Input(shape=(max_length,), dtype='int32', name='inputs')

  embedded = char_embedding_layer(inputs)

  # block 1
  convoluted_11 = Conv1D(31, 3, 3, padding='same', name='convoluted_11')(embedded)
  pooled_11 = AveragePooling1D(3, 3, padding='same', name='pooled_11')(convoluted_11)
  convoluted_21 = Conv1D(31, 7, 3, padding='same', name='convoluted_21')(embedded)
  pooled_21 = AveragePooling1D(7, 3, padding='same', name='pooled_21')(convoluted_21)
  convoluted_31 = Conv1D(31, 13, 3, padding='same', name='convoluted_31')(embedded)
  pooled_31 = AveragePooling1D(11, 3, padding='same', name='pooled_31')(convoluted_31)
  convoluted_41 = Conv1D(31, 17, 3, padding='same', name='convoluted_41')(embedded)
  pooled_41 = AveragePooling1D(13, 3, padding='same', name='pooled_41')(convoluted_41)

  merged_conv_43 = Maximum(name='merged_conv_43')([pooled_11, pooled_21, pooled_31, pooled_41])
  convoluted_43 = Conv1D(19, 7, 7, padding='same', name='convoluted_43')(merged_conv_43)
  pooled_43 = AveragePooling1D(7, 7, padding='same', name='pooled_43')(convoluted_43)
  convoluted_53 = Conv1D(7, 5, 5, padding='same', name='convoluted_53')(pooled_43)
  pooled_53 = AveragePooling1D(5, 5, padding='same', name='pooled_53')(convoluted_53)
  convoluted_63 = Conv1D(3, 3, 3, padding='same', name='convoluted_63')(pooled_53)
  pooled_63 = AveragePooling1D(3, 3, padding='same', name='pooled_63')(convoluted_63)

  unit_convoluted_53 = Conv1D(1, 1, 1, padding='same', name='unit_convoluted_53')(pooled_53)
  squeezed_53 = tf.squeeze(unit_convoluted_53, axis=2)

  unit_convoluted_63 = Conv1D(1, 1, 1, padding='same', name='unit_convoluted_63')(pooled_63)
  squeezed_63 = tf.squeeze(unit_convoluted_63, axis=2)

  merged_conv_73 = Multiply(name='merged_conv_73')([pooled_11, pooled_21, pooled_31, pooled_41])
  convoluted_73 = Conv1D(19, 7, 7, padding='same', name='convoluted_73')(merged_conv_73)
  pooled_83 = AveragePooling1D(7, 7, padding='same', name='pooled_83')(convoluted_73)
  convoluted_93 = Conv1D(7, 5, 5, padding='same', name='convoluted_93')(pooled_83)
  pooled_93 = AveragePooling1D(5, 5, padding='same', name='pooled_93')(convoluted_93)
  convoluted_103 = Conv1D(3, 3, 3, padding='same', name='convoluted_103')(pooled_93)
  pooled_103 = AveragePooling1D(3, 3, padding='same', name='pooled_103')(convoluted_103)

  unit_convoluted_103 = Conv1D(1, 1, 1, padding='same', name='unit_convoluted_103')(pooled_103)
  squeezed_103 = tf.squeeze(unit_convoluted_103, axis=2)

  # block 4
  merged_conv_14 = Maximum(name='merged_conv_14')([convoluted_11, convoluted_31])
  convoluted_24 = Conv1D(13, 5, 5, padding='same', name='convoluted_24')(merged_conv_14)
  pooled_24 = MaxPool1D(7, 5, padding='same', name='pooled_24')(convoluted_24)
  convoluted_34 = Conv1D(3, 3, 3, padding='same', name='convoluted_34')(pooled_24)
  pooled_34 = MaxPool1D(3, 3, padding='same', name='pooled_34')(convoluted_34)

  unit_convoluted_24 = Conv1D(1, 1, 1, padding='same', name='unit_convoluted_24')(pooled_24)
  squeezed_24 = tf.squeeze(unit_convoluted_24, axis=2)

  unit_convoluted_34 = Conv1D(1, 1, 1, padding='same', name='unit_convoluted_34')(pooled_34)
  squeezed_34 = tf.squeeze(unit_convoluted_34, axis=2)

  # block 5
  merged_convs = concatenate([squeezed_53, squeezed_63, squeezed_103, squeezed_24, squeezed_34], name='merged_convs')

  dense_1_dropout = Dropout(rate=dense_1_dropout_ratio, name='dense_1_dropout')(merged_convs)
  dense_1 = Dense(dense_1_size, name='dense_1', kernel_regularizer='l2', activation=dense_activation)(dense_1_dropout)
  dense_2_dropout = Dropout(rate=dense_2_dropout_ratio, name='dense_2_dropout')(dense_1)
  dense_2 = Dense(dense_2_size, name='dense_2', kernel_regularizer='l2', activation=dense_activation)(dense_2_dropout)
  dense_2_norm = LayerNormalization(axis=-1, name='dense_2_norm')(dense_2)
  dense_3_dropout = Dropout(rate=dense_3_dropout_ratio, name='dense_3_dropout')(dense_2_norm)
  dense_3 = Dense(dense_3_size, name='dense_3', kernel_regularizer='l2', activation=dense_activation)(dense_3_dropout)
  dense_3_norm = LayerNormalization(axis=-1, name='dense_3_norm')(dense_3)
  dense_4_dropout = Dropout(rate=dense_4_dropout_ratio, name='dense_4_dropout')(dense_3_norm)
  content_dense_4 = Dense(dense_4_size, name='content_dense_4', kernel_regularizer='l2', activation=dense_activation)(dense_4_dropout)
  content_dense_4_norm = LayerNormalization(axis=-1, name='content_dense_4_norm')(content_dense_4)
  title_dense_4 = Dense(dense_4_size, name='title_dense_4', kernel_regularizer='l2', activation=dense_activation)(dense_4_dropout)
  title_dense_4_norm = LayerNormalization(axis=-1, name='title_dense_4_norm')(title_dense_4)
  dense_5_dropout = Dropout(rate=dense_5_dropout_ratio, name='dense_5_dropout')(title_dense_4_norm)
  dense_5 = Dense(dense_5_size, name='dense_5', kernel_regularizer='l2', activation=dense_activation)(dense_5_dropout)
  dense_5_norm = LayerNormalization(axis=-1, name='dense_5_norm')(dense_5)

  concatenated_1 = concatenate([dense_2_norm, dense_3_norm, content_dense_4_norm, dense_5_norm], name='concatenated_1')
  detail_content_dendrop_2 = Dropout(rate=detail_content_dendrop_ratio, name='detail_content_dendrop_2')(concatenated_1)
  detail_content_dense_2 = Dense(detail_content_dense_size_2, name='detail_content_dense_2', kernel_regularizer='l2', activation=detail_content_dense_activation)(detail_content_dendrop_2)
  detail_content_dense_2_norm = LayerNormalization(axis=-1, name='detail_content_dense_2_norm')(detail_content_dense_2)
  detail_content_outdrop_2 = Dropout(rate=detail_content_outdrop_ratio, name='detail_content_outdrop_2')(detail_content_dense_2_norm)
  detail_content_output = Dense(num_categories, name='detail_content_output', activation='softmax')(detail_content_outdrop_2)

  concatenated_2 = concatenate([dense_2_norm, dense_3_norm, title_dense_4_norm], name='concatenated_2')
  detail_title_dendrop_2 = Dropout(rate=detail_title_dendrop_ratio, name='detail_title_dendrop_2')(concatenated_2)
  detail_title_dense_2 = Dense(detail_title_dense_size_2, name='detail_title_dense_2', kernel_regularizer='l2', activation=detail_title_dense_activation)(detail_title_dendrop_2)
  detail_title_dense_2_norm = LayerNormalization(axis=-1, name='detail_title_dense_2_norm')(detail_title_dense_2)
  detail_title_outdrop = Dropout(rate=detail_title_outdrop_ratio, name='detail_title_outdrop')(detail_title_dense_2_norm)
  detail_title_output = Dense(num_categories, name='detail_title_output', activation='softmax')(detail_title_outdrop)

  merged_abstract_content = concatenate([content_dense_4_norm, dense_5_norm], name='merged_abstract_content')
  abstract_content_outdrop = Dropout(rate=abstract_content_outdrop_ratio, name='abstract_content_outdrop')(merged_abstract_content)
  abstract_content_output = Dense(num_categories, name='abstract_content_output', activation='softmax')(abstract_content_outdrop)

  model = Model(inputs=[inputs], outputs=[abstract_content_output, detail_content_output, detail_title_output])


  optimizer = Adam(learning_rate=lr)

  model.compile(
      loss='categorical_crossentropy',
      optimizer=optimizer,
      metrics=['accuracy'],
    )

  return model


def HierarchyV3_170000(cfg, learning_rate=None):
  num_categories = cfg['num_categories']
  num_classes = cfg['num_classes']
  max_length = cfg['max_length']
  lr = learning_rate if learning_rate else cfg['learning_rate']

  conv_activation = 'swish'
  dense_activation = 'swish'
  detail_content_dense_activation = 'swish'
  detail_title_dense_activation = 'swish'
  abstract_content_dense_activation = 'swish'

  dense_1_size = 13
  dense_2_size = 71
  detail_content_dense_3_size = 11
  title_dense_3_size = 11
  abs_content_dense_3_size = 19
  dense_4_size = 7

  dense_1_dropout_ratio = 0.71
  dense_2_dropout_ratio = 0.11
  detail_content_outdrop_3_ratio = 0.29
  title_outdrop_3_ratio = 0.29
  abstract_content_outdrop_3_ratio = 0.29
  dense_4_dropout_ratio = 0.17

  print('[INFO] dense_1_dropout_ratio: ', dense_1_dropout_ratio)
  print('[INFO] dense_2_dropout_ratio: ', dense_2_dropout_ratio)
  print('[INFO] detail_content_outdrop_3_ratio: ', detail_content_outdrop_3_ratio)
  print('[INFO] title_outdrop_3_ratio: ', title_outdrop_3_ratio)
  print('[INFO] abstract_content_outdrop_3_ratio: ', abstract_content_outdrop_3_ratio)
  print('[INFO] dense_4_dropout_ratio: ', dense_4_dropout_ratio)

  char_embedding_layer = CharEmbeddingV04(cfg, trainable=False, name='char_embedding')

  input = Input(shape=(max_length,), dtype='int32', name='input')

  embedded = char_embedding_layer(input)

  # block 1
  convoluted_11 = Conv1D(3, 3, 2, activation=conv_activation, padding='same', name='convoluted_11')(embedded)
  convoluted_21 = Conv1D(3, 5, 2, activation=conv_activation, padding='same', name='convoluted_21')(embedded)
  convoluted_31 = Conv1D(3, 7, 2, activation=conv_activation, padding='same', name='convoluted_31')(embedded)
  convoluted_41 = Conv1D(3, 11, 2, activation=conv_activation, padding='same', name='convoluted_41')(embedded)
  merged_conv_1 = Maximum(name='merged_conv_1')([convoluted_11, convoluted_21, convoluted_31, convoluted_41])

  # block 2
  convoluted_12 = Conv1D(7, 3, 3, activation=conv_activation, padding='same', name='convoluted_12')(merged_conv_1)
  convoluted_22 = Conv1D(11, 7, 5, activation=conv_activation, padding='same', name='convoluted_22')(convoluted_12)
  convoluted_32 = Conv1D(17, 7, 5, activation=conv_activation, padding='same', name='convoluted_32')(convoluted_22)

  unit_convoluted_42 = Conv1D(1, 1, 1, activation=conv_activation, padding='same', name='unit_convoluted_42')(convoluted_32)
  conv_squeezed_42 = tf.squeeze(unit_convoluted_42, axis=2, name='conv_squeezed_42')

  aver_pooled_42 = AveragePooling1D(3, 3, padding='same', name='aver_pooled_42')(convoluted_32)
  aver_unit_convoluted_42 = Conv1D(1, 1, 1, activation=conv_activation, padding='same', name='aver_unit_convoluted_42')(aver_pooled_42)
  aver_squeezed_42 = tf.squeeze(aver_unit_convoluted_42, axis=2, name='aver_squeezed_42')

  max_max_pooled_42 = MaxPool1D(3, 3, padding='same', name='max_max_pooled_42')(convoluted_32)
  max_unit_convoluted_42 = Conv1D(1, 1, 1, activation=conv_activation, padding='same', name='max_unit_convoluted_42')(max_max_pooled_42)
  max_squeezed_42 = tf.squeeze(max_unit_convoluted_42, axis=2, name='max_squeezed_42')

  # block 3
  convoluted_13 = Conv1D(7, 3, 3, activation=conv_activation, padding='same', name='convoluted_13')(merged_conv_1)
  pooled_13 = AveragePooling1D(3, 3, padding='same', name='pooled_13')(convoluted_13)
  convoluted_23 = Conv1D(11, 3, 3, activation=conv_activation, padding='same', name='convoluted_23')(pooled_13)
  pooled_23 = AveragePooling1D(3, 3, padding='same', name='pooled_23')(convoluted_23)
  convoluted_33 = Conv1D(17, 7, 5, activation=conv_activation, padding='same', name='convoluted_33')(pooled_23)
  pooled_33 = AveragePooling1D(7, 5, padding='same', name='pooled_33')(convoluted_33)

  aver_pooled_43 = AveragePooling1D(3, 3, padding='same', name='aver_pooled_43')(pooled_33)
  aver_unit_convoluted_43 = Conv1D(1, 1, 1, activation=conv_activation, padding='same', name='aver_unit_convoluted_43')(aver_pooled_43)
  aver_squeezed_43 = tf.squeeze(aver_unit_convoluted_43, axis=2, name='aver_squeezed_43')

  max_pooled_43 = MaxPool1D(3, 3, padding='same', name='max_pooled_43')(pooled_33)
  max_unit_convoluted_43 = Conv1D(1, 1, 1, activation=conv_activation, padding='same', name='max_unit_convoluted_43')(max_pooled_43)
  max_squeezed_43 = tf.squeeze(max_unit_convoluted_43, axis=2, name='max_squeezed_43')

  # block 4
  convoluted_14 = Conv1D(7, 7, 5, activation=None, padding='same', name='convoluted_14')(merged_conv_1)
  convoluted_24 = Conv1D(11, 7, 5, activation=None, padding='same', name='convoluted_24')(convoluted_14)
  convoluted_34 = Conv1D(17, 7, 5, activation=None, padding='same', name='convoluted_34')(convoluted_24)

  linear_unit_convoluted_64 = Conv1D(1, 1, 1, activation=None, padding='same', name='linear_unit_convoluted_64')(convoluted_34)
  linear_squeezed_64 = tf.squeeze(linear_unit_convoluted_64, axis=2, name='linear_squeezed_64')

  activated_unit_convoluted_64 = Conv1D(1, 1, 1, activation=conv_activation, padding='same', name='activated_unit_convoluted_64')(convoluted_34)
  activated_squeezed_64 = tf.squeeze(activated_unit_convoluted_64, axis=2, name='activated_squeezed_64')

  max_pooled_64 = MaxPool1D(3, 3, padding='same', name='max_pooled_64')(convoluted_34)
  max_unit_convoluted_64 = Conv1D(1, 1, 1, activation=conv_activation, padding='same', name='max_unit_convoluted_64')(max_pooled_64)
  max_squeezed_64 = tf.squeeze(max_unit_convoluted_64, axis=2, name='max_squeezed_64')

  merged_convs = concatenate([
      conv_squeezed_42, aver_squeezed_42, max_squeezed_42,
      aver_squeezed_43, max_squeezed_43,
      activated_squeezed_64, max_squeezed_64, linear_squeezed_64,
    ], name='merged_convs')

  # Depth lv1
  det_content_decoder_dendrop_1 = Dropout(rate=dense_1_dropout_ratio, name='det_content_decoder_dendrop_1')(merged_convs)
  det_content_decoder_dense_1 = Dense(dense_1_size, name='det_content_decoder_dense_1', kernel_regularizer='l2', activation=dense_activation)(det_content_decoder_dendrop_1)
  det_content_decoder_dense_1_norm = LayerNormalization(trainable=True, name='det_content_decoder_dense_1_norm')(det_content_decoder_dense_1)

  title_decoder_dendrop_1 = Dropout(rate=dense_1_dropout_ratio, name='title_decoder_dendrop_1')(merged_convs)
  title_decoder_dense_1 = Dense(dense_1_size, name='title_decoder_dense_1', kernel_regularizer='l2', activation=dense_activation)(title_decoder_dendrop_1)
  title_decoder_dense_1_norm = LayerNormalization(trainable=True, name='title_decoder_dense_1_norm')(title_decoder_dense_1)

  abs_content_decoder_dendrop_1 = Dropout(rate=dense_1_dropout_ratio, name='abs_content_decoder_dendrop_1')(merged_convs)
  abs_content_decoder_dense_1 = Dense(dense_1_size, name='abs_content_decoder_dense_1', kernel_regularizer='l2', activation=dense_activation)(abs_content_decoder_dendrop_1)
  abs_content_decoder_dense_1_norm = LayerNormalization(trainable=True, name='abs_content_decoder_dense_1_norm')(abs_content_decoder_dense_1)

  # Depth lv2
  det_content_dendrop_2 = Dropout(rate=dense_2_dropout_ratio, name='det_content_dendrop_2')(det_content_decoder_dense_1_norm)
  det_content_dense_2 = Dense(dense_2_size, name='det_content_dense_2', kernel_regularizer='l2', activation=dense_activation)(det_content_dendrop_2)
  det_content_dense_2_norm = LayerNormalization(trainable=True, name='det_content_dense_2_norm')(det_content_dense_2)

  title_dendrop_2 = Dropout(rate=dense_2_dropout_ratio, name='title_dendrop_2')(title_decoder_dense_1_norm)
  title_dense_2 = Dense(dense_2_size, name='title_dense_2', kernel_regularizer='l2', activation=dense_activation)(title_dendrop_2)
  title_dense_2_norm = LayerNormalization(trainable=True, name='title_dense_2_norm')(title_dense_2)

  abs_content_dendrop_2 = Dropout(rate=dense_2_dropout_ratio, name='abs_content_dendrop_2')(abs_content_decoder_dense_1_norm)
  abs_content_dense_2 = Dense(dense_2_size, name='abs_content_dense_2', kernel_regularizer='l2', activation=dense_activation)(abs_content_dendrop_2)
  abs_content_dense_2_norm = LayerNormalization(trainable=True, name='abs_content_dense_2_norm')(abs_content_dense_2)

  # Depth lv3
  content_concatenated_3 = concatenate([det_content_decoder_dense_1_norm, det_content_dense_2_norm], name='content_concatenated_3')
  detail_content_dendrop_3 = Dropout(rate=detail_content_outdrop_3_ratio, name='detail_content_dendrop_3')(content_concatenated_3)
  detail_content_dense_3 = Dense(detail_content_dense_3_size, name='detail_content_dense_3', kernel_regularizer='l2', activation=detail_content_dense_activation)(detail_content_dendrop_3)
  detail_content_dense_norm_3 = LayerNormalization(trainable=True, name='detail_content_dense_norm_3')(detail_content_dense_3)
  detail_content_output = Dense(num_categories, name='detail_content_output', activation='softmax')(detail_content_dense_norm_3)

  title_concatenated_3 = concatenate([title_decoder_dense_1_norm, title_dense_2_norm], name='title_concatenated_3')
  detail_title_dendrop_3 = Dropout(rate=title_outdrop_3_ratio, name='detail_title_dendrop_3')(title_concatenated_3)
  detail_title_dense_3 = Dense(title_dense_3_size, name='detail_title_dense_3', kernel_regularizer='l2', activation=detail_title_dense_activation)(detail_title_dendrop_3)
  detail_title_dense_norm_3 = LayerNormalization(trainable=True, name='detail_title_dense_norm_3')(detail_title_dense_3)
  detail_title_output = Dense(num_categories, name='detail_title_output', activation='softmax')(detail_title_dense_norm_3)

  abs_content_dendrop_3 = Dropout(rate=abstract_content_outdrop_3_ratio, name='abs_content_dendrop_3')(abs_content_dense_2_norm)
  abs_content_dense_3 = Dense(abs_content_dense_3_size, name='abs_content_dense_3', kernel_regularizer='l2', activation=dense_activation)(abs_content_dendrop_3)
  abs_dense_3_norm = LayerNormalization(trainable=True, name='abs_dense_3_norm')(abs_content_dense_3)
  abstract_content_dendrop = Dropout(rate=dense_4_dropout_ratio, name='abstract_content_dendrop')(abs_dense_3_norm)
  abstract_content_dense_4 = Dense(dense_4_size, name='abstract_content_dense_4', kernel_regularizer='l2', activation=abstract_content_dense_activation)(abstract_content_dendrop)
  abstract_content_dense_norm_4 = LayerNormalization(trainable=True, name='abstract_content_dense_norm_4')(abstract_content_dense_4)
  abstract_content_output = Dense(num_categories, name='abstract_content_output', activation='softmax')(abstract_content_dense_norm_4)

  model = Model(inputs=[input], outputs=[abstract_content_output, detail_content_output, detail_title_output])

  optimizer = RMSprop(learning_rate=lr)

  # learning_rate_fn = InverseTimeDecay(
  #     initial_learning_rate=lr,
  #     decay_steps=decay_steps,
  #     decay_rate=decay_rate,
  #     staircase=True
  #   )
  # optimizer = Nadam(learning_rate=lr)
  # optimizer = Adam(learning_rate=learning_rate_fn)
  # optimizer = RMSprop(learning_rate=learning_rate_fn)
  # optimizer = Adagrad(learning_rate=learning_rate_fn)

  model.compile(
      loss='categorical_crossentropy',
      optimizer=optimizer,
      metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy')],
    )

  return model


def HierarchyV3_130100(cfg, learning_rate=None):
  num_categories = cfg['num_categories']
  num_classes = cfg['num_classes']
  max_length = cfg['max_length']
  lr = learning_rate if learning_rate else cfg['learning_rate']

  dense_1_dropout_ratio = 0.05
  dense_1_size = 17
  dense_2_dropout_ratio = 0.12
  dense_2_size = 17
  dense_3_dropout_ratio = 0.10
  dense_3_size = 11
  # dense_4_dropout_ratio = 0.10
  dense_4_size = 7

  conv_activation = 'swish'
  dense_activation = 'swish'
  detail_content_dense_activation = 'swish'
  detail_title_dense_activation = 'swish'
  abstract_content_dense_activation = 'swish'

  detail_content_outdrop_ratio = 0.15
  detail_title_outdrop_ratio = 0.12
  abstract_content_outdrop_ratio = 0.10

  print('[ATTENTION!] Models version: v3.13.1.0')
  print('[INFO] detail_content_outdrop_ratio: ', detail_content_outdrop_ratio)
  print('[INFO] detail_title_outdrop_ratio: ', detail_title_outdrop_ratio)
  print('[INFO] abstract_content_outdrop_ratio: ', abstract_content_outdrop_ratio)

  char_embedding_layer = CharEmbeddingV4117(cfg, trainable=False, name='char_embedding')

  input = Input(shape=(max_length,), dtype='int32', name='input')

  embedded = char_embedding_layer(input)

  # block 1
  convoluted_11_s1 = Conv1D(7, 1, 1, activation=None, padding='same', name='convoluted_11_s1')(embedded)
  convoluted_11 = Conv1D(11, 3, 1, activation=conv_activation, padding='same', name='convoluted_11')(convoluted_11_s1)
  convoluted_21_s1 = Conv1D(7, 1, 1, activation=None, padding='same', name='convoluted_21_s1')(embedded)
  convoluted_21 = Conv1D(11, 5, 1, activation=conv_activation, padding='same', name='convoluted_21')(convoluted_21_s1)
  convoluted_31_s1 = Conv1D(7, 1, 1, activation=None, padding='same', name='convoluted_31_s1')(embedded)
  convoluted_31 = Conv1D(11, 7, 1, activation=conv_activation, padding='same', name='convoluted_31')(convoluted_31_s1)
  convoluted_41_s1 = Conv1D(7, 1, 1, activation=None, padding='same', name='convoluted_41_s1')(embedded)
  convoluted_41 = Conv1D(11, 11, 1, activation=conv_activation, padding='same', name='convoluted_41')(convoluted_41_s1)

  # block 2
  merged_conv_12 = Average(name='merged_conv_12')([convoluted_11, convoluted_21, convoluted_31, convoluted_41])
  convoluted_12_s1 = Conv1D(11, 3, 1, padding='same', activation=None, name='convoluted_12_s1')(merged_conv_12)
  convoluted_12 = Conv1D(17, 3, 3, activation=conv_activation, padding='same', name='convoluted_12')(convoluted_12_s1)
  convoluted_22_s1 = Conv1D(11, 3, 1, padding='same', activation=None, name='convoluted_22_s1')(convoluted_12)
  convoluted_22 = Conv1D(29, 3, 3, activation=conv_activation, padding='same', name='convoluted_22')(convoluted_22_s1)
  convoluted_32_s1 = Conv1D(11, 3, 1, padding='same', activation=None, name='convoluted_32_s1')(convoluted_22)
  convoluted_32 = Conv1D(41, 7, 5, activation=conv_activation, padding='same', name='convoluted_32')(convoluted_32_s1)
  convoluted_42 = Conv1D(53, 11, 7, activation=None, padding='same', name='convoluted_42')(convoluted_32)

  unit_convoluted_42 = Conv1D(1, 1, 1, activation=conv_activation, padding='same', name='unit_convoluted_42')(convoluted_42)
  conv_squeezed_42 = tf.squeeze(unit_convoluted_42, axis=2, name='conv_squeezed_42')

  aver_pooled_42 = AveragePooling1D(3, 1, padding='same', name='aver_pooled_42')(convoluted_42)
  aver_unit_convoluted_42 = Conv1D(1, 1, 1, activation=conv_activation, padding='same', name='aver_unit_convoluted_42')(aver_pooled_42)
  aver_squeezed_42 = tf.squeeze(aver_unit_convoluted_42, axis=2, name='aver_squeezed_42')

  max_max_pooled_42 = MaxPool1D(3, 3, padding='same', name='max_max_pooled_42')(convoluted_42)
  max_unit_convoluted_42 = Conv1D(1, 1, 1, activation=conv_activation, padding='same', name='max_unit_convoluted_42')(max_max_pooled_42)
  max_squeezed_42 = tf.squeeze(max_unit_convoluted_42, axis=2, name='max_squeezed_42')

  # block 3
  merged_conv_13 = Maximum(name='merged_conv_13')([convoluted_11, convoluted_21, convoluted_31, convoluted_41])
  convoluted_13_s1 = Conv1D(11, 3, 1, activation=None, padding='same', name='convoluted_13_s1')(merged_conv_13)
  convoluted_13 = Conv1D(17, 3, 3, activation=conv_activation, padding='same', name='convoluted_13')(convoluted_13_s1)
  convoluted_23_s1 = Conv1D(11, 3, 1, activation=None, padding='same', name='convoluted_23_s1')(convoluted_13)
  convoluted_23 = Conv1D(29, 3, 3, activation=conv_activation, padding='same', name='convoluted_23')(convoluted_23_s1)
  convoluted_33_s1 = Conv1D(11, 3, 1, activation=None, padding='same', name='convoluted_33_s1')(convoluted_23)
  convoluted_33 = Conv1D(41, 7, 5, activation=conv_activation, padding='same', name='convoluted_33')(convoluted_33_s1)
  convoluted_43 = Conv1D(53, 11, 7, activation=None, padding='same', name='convoluted_43')(convoluted_33)

  aver_pooled_43 = AveragePooling1D(3, 1, padding='same', name='aver_pooled_43')(convoluted_43)
  aver_unit_convoluted_43 = Conv1D(1, 1, 1, activation=conv_activation, padding='same', name='aver_unit_convoluted_43')(aver_pooled_43)
  aver_squeezed_43 = tf.squeeze(aver_unit_convoluted_43, axis=2, name='aver_squeezed_43')

  max_pooled_43 = MaxPool1D(3, 3, padding='same', name='max_pooled_43')(convoluted_43)
  max_unit_convoluted_43 = Conv1D(1, 1, 1, activation=conv_activation, padding='same', name='max_unit_convoluted_43')(max_pooled_43)
  max_squeezed_43 = tf.squeeze(max_unit_convoluted_43, axis=2, name='max_squeezed_43')

  # block 4
  merged_conv_14 = Maximum(name='merged_conv_14')([convoluted_11, convoluted_21, convoluted_31, convoluted_41])
  convoluted_14 = Conv1D(11, 3, 3, activation=None, padding='same', name='convoluted_14')(merged_conv_14)
  convoluted_24 = Conv1D(17, 3, 3, activation=None, padding='same', name='convoluted_24')(convoluted_14)
  convoluted_34 = Conv1D(29, 3, 3, activation=None, padding='same', name='convoluted_34')(convoluted_24)
  convoluted_44 = Conv1D(41, 3, 3, activation=None, padding='same', name='convoluted_44')(convoluted_34)
  convoluted_54 = Conv1D(53, 5, 3, activation=None, padding='same', name='convoluted_54')(convoluted_44)
  convoluted_64 = Conv1D(79, 7, 5, activation=None, padding='same', name='convoluted_64')(convoluted_54)

  linear_unit_convoluted_64 = Conv1D(1, 1, 1, padding='same', name='linear_unit_convoluted_64')(convoluted_64)
  linear_squeezed_64 = tf.squeeze(linear_unit_convoluted_64, axis=2, name='linear_squeezed_64')

  activated_unit_convoluted_64 = Conv1D(1, 1, 1, activation=conv_activation, padding='same', name='activated_unit_convoluted_64')(convoluted_64)
  activated_squeezed_64 = tf.squeeze(activated_unit_convoluted_64, axis=2, name='activated_squeezed_64')

  max_pooled_64 = MaxPool1D(3, 1, padding='same', name='max_pooled_64')(convoluted_64)
  max_unit_convoluted_64 = Conv1D(1, 1, 1, activation=conv_activation, padding='same', name='max_unit_convoluted_64')(max_pooled_64)
  max_squeezed_64 = tf.squeeze(max_unit_convoluted_64, axis=2, name='max_squeezed_64')

  # block 5
  merged_convs = concatenate([
      conv_squeezed_42, aver_squeezed_42, max_squeezed_42,
      aver_squeezed_43, max_squeezed_43,
      linear_squeezed_64, activated_squeezed_64, max_squeezed_64,
    ], name='merged_convs')

  # merged_convs_norm = BatchNormalization(trainable=True, name='merged_convs_norm')(merged_convs)
  dense_1_dropout = Dropout(rate=dense_1_dropout_ratio, name='dense_1_dropout')(merged_convs)
  dense_1 = Dense(dense_1_size, name='dense_1', kernel_regularizer='l2', activation=dense_activation)(dense_1_dropout)
  dense_2_dropout = Dropout(rate=dense_2_dropout_ratio, name='dense_2_dropout')(dense_1)
  dense_2 = Dense(dense_2_size, name='dense_2', kernel_regularizer='l2', activation=dense_activation)(dense_2_dropout)
  dense_2_norm = BatchNormalization(trainable=True, name='dense_2_norm')(dense_2)
  content_dendrop_3 = Dropout(rate=dense_3_dropout_ratio, name='content_dendrop_3')(dense_2_norm)
  content_dense_3 = Dense(dense_3_size, name='content_dense_3', kernel_regularizer='l2', activation=dense_activation)(content_dendrop_3)
  content_dense_3_norm = BatchNormalization(trainable=True, name='content_dense_3_norm')(content_dense_3)
  title_dendrop_3 = Dropout(rate=dense_3_dropout_ratio, name='title_dendrop_3')(dense_2_norm)
  title_dense_3 = Dense(dense_3_size, name='title_dense_3', kernel_regularizer='l2', activation=dense_activation)(title_dendrop_3)
  title_dense_3_norm = BatchNormalization(trainable=True, name='title_dense_3_norm')(title_dense_3)

  content_concatenated_4 = concatenate([dense_2_norm, content_dense_3_norm], name='content_concatenated_4')
  detail_content_dendrop = Dropout(rate=detail_content_outdrop_ratio, name='detail_content_dendrop')(content_concatenated_4)
  detail_content_dense = Dense(dense_4_size, name='detail_content_dense', kernel_regularizer='l2', activation=detail_content_dense_activation)(detail_content_dendrop)
  detail_content_dense_norm = BatchNormalization(trainable=True, name='detail_content_dense_norm')(detail_content_dense)
  detail_content_output = Dense(num_categories, name='detail_content_output', activation='softmax')(detail_content_dense_norm)

  title_concatenated_4 = concatenate([dense_2_norm, title_dense_3_norm], name='title_concatenated_4')
  detail_title_dendrop = Dropout(rate=detail_title_outdrop_ratio, name='detail_title_dendrop')(title_concatenated_4)
  detail_title_dense = Dense(dense_4_size, name='detail_title_dense', kernel_regularizer='l2', activation=detail_title_dense_activation)(detail_title_dendrop)
  detail_title_dense_norm = BatchNormalization(trainable=True, name='detail_title_dense_norm')(detail_title_dense)
  detail_title_output = Dense(num_categories, name='detail_title_output', activation='softmax')(detail_title_dense_norm)

  content_title_concatenated = concatenate([title_dense_3_norm, content_dense_3_norm], name='content_title_concatenated')
  abstract_content_dendrop = Dropout(rate=abstract_content_outdrop_ratio, name='abstract_content_dendrop')(content_title_concatenated)
  abstract_content_dense = Dense(dense_4_size, name='abstract_content_dense', kernel_regularizer='l2', activation=abstract_content_dense_activation)(abstract_content_dendrop)
  abstract_content_dense_norm = BatchNormalization(trainable=True, name='abstract_content_dense_norm')(abstract_content_dense)
  abstract_content_output = Dense(num_categories, name='abstract_content_output', activation='softmax')(abstract_content_dense_norm)

  model = Model(inputs=[input], outputs=[abstract_content_output, detail_content_output, detail_title_output])


  optimizer = Nadam(learning_rate=lr)

  # learning_rate_fn = InverseTimeDecay(
  #     initial_learning_rate=lr,
  #     decay_steps=decay_steps,
  #     decay_rate=decay_rate,
  #     staircase=True
  #   )
  # optimizer = Nadam(learning_rate=lr)
  # optimizer = Adam(learning_rate=learning_rate_fn)
  # optimizer = RMSprop(learning_rate=learning_rate_fn)
  # optimizer = Adagrad(learning_rate=learning_rate_fn)

  model.compile(
      loss='categorical_crossentropy',
      optimizer=optimizer,
      metrics=['accuracy'],
    )

  return model

def HierarchyV3_100100(cfg, learning_rate=None):
  num_categories = cfg['num_categories']
  num_classes = cfg['num_classes']
  max_length = cfg['max_length']
  lr = learning_rate if learning_rate else cfg['learning_rate']

  dense_1_dropout_ratio = 0.15
  dense_1_size = 41
  dense_2_dropout_ratio = 0.12
  dense_2_size = 29
  dense_3_dropout_ratio = 0.12
  dense_3_size = 17
  dense_4_dropout_ratio = 0.10
  dense_4_size = 11
  dense_5_dropout_ratio = 0.08
  dense_5_size = 7
  dense_activation = 'swish'

  detail_content_dendrop_ratio = 0.12
  detail_content_dense_size = 11
  detail_content_dense_activation = 'tanh'
  detail_content_outdrop_ratio = 0.10

  # detail_title_dendrop_ratio = 0.15
  # detail_title_dense_size_2 = 29
  # detail_title_dense_activation = 'tanh'

  detail_title_outdrop_ratio = 0.15

  abstract_content_outdrop_ratio = 0.15

  print('[ATTENTION!] Model version: v3.10.1.0')
  print('[INFO] detail_content_dendrop_ratio', detail_content_dendrop_ratio)
  print('[INFO] detail_title_outdrop_ratio', detail_title_outdrop_ratio)
  print('[INFO] abstract_content_outdrop_ratio', abstract_content_outdrop_ratio)

  char_embedding_layer = CharEmbeddingV411(cfg, trainable=False, name='char_embedding')

  input = Input(shape=(max_length,), dtype='int32', name='input')
  # one_hot = tf.one_hot(input, num_classes)

  embedded = char_embedding_layer(input)

  # block 1
  convoluted_11 = Conv1D(3, 3, 1, padding='same', name='convoluted_11')(embedded)
  pooled_11 = AveragePooling1D(3, 1, padding='same', name='pooled_11')(convoluted_11)
  convoluted_21 = Conv1D(3, 5, 1, padding='same', name='convoluted_21')(embedded)
  pooled_21 = AveragePooling1D(5, 1, padding='same', name='pooled_21')(convoluted_21)
  convoluted_31 = Conv1D(3, 7, 1, padding='same', name='convoluted_31')(embedded)
  pooled_31 = AveragePooling1D(7, 1, padding='same', name='pooled_31')(convoluted_31)
  convoluted_41 = Conv1D(3, 11, 1, padding='same', name='convoluted_41')(embedded)
  pooled_41 = AveragePooling1D(11, 1, padding='same', name='pooled_41')(convoluted_41)

  # block 2
  merged_conv_12 = Add(name='merged_conv_12')([pooled_11, pooled_21, pooled_31, pooled_41])
  convoluted_12 = Conv1D(7, 3, 3, padding='same', name='convoluted_12')(merged_conv_12)
  pooled_12 = AveragePooling1D(3, 3, padding='same', name='pooled_12')(convoluted_12)
  convoluted_22 = Conv1D(13, 3, 3, padding='same', name='convoluted_22')(pooled_12)
  pooled_22 = AveragePooling1D(3, 3, padding='same', name='pooled_22')(convoluted_22)

  convoluted_32 = Conv1D(19, 3, 3, padding='same', name='convoluted_32')(pooled_22)
  pooled_32 = AveragePooling1D(3, 3, padding='same', name='pooled_32')(convoluted_32)
  unit_convoluted_32 = Conv1D(1, 1, 1, padding='same', name='unit_convoluted_32')(pooled_32)
  squeezed_32 = tf.squeeze(unit_convoluted_32, axis=2)

  convoluted_42 = Conv1D(31, 3, 3, padding='same', name='convoluted_42')(pooled_32)
  pooled_42 = AveragePooling1D(3, 3, padding='same', name='pooled_42')(convoluted_42)
  unit_convoluted_42 = Conv1D(1, 1, 1, padding='same', name='unit_convoluted_42')(pooled_42)
  squeezed_42 = tf.squeeze(unit_convoluted_42, axis=2)


  # block 3
  merged_conv_13 = Multiply(name='merged_conv_13')([pooled_11, pooled_21, pooled_31, pooled_41])
  convoluted_13 = Conv1D(7, 3, 3, padding='same', name='convoluted_13')(merged_conv_13)
  pooled_13 = AveragePooling1D(3, 3, padding='same', name='pooled_13')(convoluted_13)
  convoluted_23 = Conv1D(13, 3, 3, padding='same', name='convoluted_23')(pooled_13)
  pooled_23 = AveragePooling1D(3, 3, padding='same', name='pooled_23')(convoluted_23)
  convoluted_33 = Conv1D(19, 3, 3, padding='same', name='convoluted_33')(pooled_23)
  pooled_33 = AveragePooling1D(3, 3, padding='same', name='pooled_33')(convoluted_33)
  convoluted_43 = Conv1D(31, 3, 3, padding='same', name='convoluted_43')(pooled_33)
  pooled_43 = AveragePooling1D(3, 3, padding='same', name='pooled_43')(convoluted_43)

  unit_convoluted_43 = Conv1D(1, 1, 1, padding='same', name='unit_convoluted_43')(pooled_43)
  squeezed_43 = tf.squeeze(unit_convoluted_43, axis=2)

  # block 4
  merged_conv_14 = Add(name='merged_conv_14')([pooled_21, pooled_41])
  convoluted_14 = Conv1D(7, 3, 3, padding='same', name='convoluted_14')(merged_conv_14)
  pooled_14 = MaxPool1D(3, 3, padding='same', name='pooled_14')(convoluted_14)
  convoluted_24 = Conv1D(13, 5, 5, padding='same', name='convoluted_24')(pooled_14)
  pooled_24 = MaxPool1D(5, 5, padding='same', name='pooled_24')(convoluted_24)
  convoluted_34 = Conv1D(19, 5, 5, padding='same', name='convoluted_34')(pooled_24)
  pooled_34 = MaxPool1D(5, 5, padding='same', name='pooled_34')(convoluted_34)

  unit_convoluted_34 = Conv1D(1, 1, 1, padding='same', name='unit_convoluted_34')(pooled_34)
  squeezed_34 = tf.squeeze(unit_convoluted_34, axis=2)

  # block 5
  merged_convs = concatenate([squeezed_32, squeezed_42, squeezed_43, squeezed_34], name='merged_convs')

  dense_1_dropout = Dropout(rate=dense_1_dropout_ratio, name='dense_1_dropout')(merged_convs)
  dense_1 = Dense(dense_1_size, name='dense_1', kernel_regularizer='l2', activation=dense_activation)(dense_1_dropout)
  dense_2_dropout = Dropout(rate=dense_2_dropout_ratio, name='dense_2_dropout')(dense_1)
  dense_2 = Dense(dense_2_size, name='dense_2', kernel_regularizer='l2', activation=dense_activation)(dense_2_dropout)
  dense_2_norm = BatchNormalization(trainable=False, name='dense_2_norm')(dense_2)
  dense_3_dropout = Dropout(rate=dense_3_dropout_ratio, name='dense_3_dropout')(dense_2)
  dense_3 = Dense(dense_3_size, name='dense_3', kernel_regularizer='l2', activation=dense_activation)(dense_3_dropout)
  dense_3_norm = BatchNormalization(trainable=False, name='dense_3_norm')(dense_3)
  title_dendrop_3 = Dropout(rate=dense_3_dropout_ratio, name='title_dendrop_3')(dense_2)
  title_dense_3 = Dense(dense_3_size, name='title_dense_3', kernel_regularizer='l2', activation=dense_activation)(title_dendrop_3)
  title_dense_3_norm = BatchNormalization(trainable=True, name='title_dense_3_norm')(title_dense_3)
  content_dendrop_4 = Dropout(rate=dense_4_dropout_ratio, name='content_dendrop_4')(dense_3)
  content_dense_4 = Dense(dense_4_size, name='content_dense_4', kernel_regularizer='l2', activation=dense_activation)(content_dendrop_4)
  content_dense_4_norm = BatchNormalization(trainable=True, name='content_dense_4_norm')(content_dense_4)
  dense_5_dropout = Dropout(rate=dense_5_dropout_ratio, name='dense_5_dropout')(content_dense_4_norm)
  dense_5 = Dense(dense_5_size, name='dense_5', kernel_regularizer='l2', activation=dense_activation)(dense_5_dropout)
  dense_5_norm = BatchNormalization(trainable=True, name='dense_5_norm')(dense_5)

  concatenated_1 = concatenate([dense_2_norm, dense_3_norm], name='concatenated_1')
  detail_content_dendrop = Dropout(rate=detail_content_dendrop_ratio, name='detail_content_dendrop')(concatenated_1)
  detail_content_dense = Dense(detail_content_dense_size, name='detail_content_dense', kernel_regularizer='l2', activation=detail_content_dense_activation)(detail_content_dendrop)
  detail_content_dense_norm = BatchNormalization(trainable=True, name='detail_content_dense_norm')(detail_content_dense)
  detail_content_outdrop = Dropout(rate=detail_content_outdrop_ratio, name='detail_content_outdrop')(detail_content_dense_norm)
  detail_content_output = Dense(num_categories, name='detail_content_output', activation='softmax')(detail_content_outdrop)

  detail_title_outdrop = Dropout(rate=detail_title_outdrop_ratio, name='detail_title_outdrop')(title_dense_3_norm)
  detail_title_output = Dense(num_categories, name='detail_title_output', activation='softmax')(detail_title_outdrop)

  abstract_content_outdrop = Dropout(rate=abstract_content_outdrop_ratio, name='abstract_content_outdrop')(dense_5_norm)
  abstract_content_output = Dense(num_categories, name='abstract_content_output', activation='softmax')(abstract_content_outdrop)

  model = Model(inputs=[input], outputs=[abstract_content_output, detail_content_output, detail_title_output])

  Optimizer = cfg['optimizer']
  print('[INFO] optimizer: ', Optimizer)

  optimizer = Optimizer(learning_rate=lr)

  model.compile(
      loss='categorical_crossentropy',
      optimizer=optimizer,
      metrics=['accuracy'],
    )

  return model

def HierarchyV3_170200(cfg, learning_rate=None):
  num_categories = cfg['num_categories']
  num_classes = cfg['num_classes']
  max_length = cfg['max_length']
  lr = learning_rate if learning_rate else cfg['learning_rate']

  conv_activation = 'swish'
  dense_activation = 'swish'
  detail_content_dense_activation = 'swish'
  detail_title_dense_activation = 'swish'
  abstract_content_dense_activation = 'swish'

  dense_1_size = 19
  dense_2_size = 79
  detail_content_dense_3_size = 17
  title_dense_3_size = 17
  abs_content_dense_3_size = 43
  dense_4_size = 17

  dense_1_dropout_ratio = 0.83
  dense_2_dropout_ratio = 0.47
  detail_content_outdrop_3_ratio = 0.47
  title_outdrop_3_ratio = 0.47
  abstract_content_outdrop_3_ratio = 0.47
  dense_4_dropout_ratio = 0.23

  print('[ATTENTION!] Model version: v3.17.2.0')
  print('[INFO] dense_1_dropout_ratio: ', dense_1_dropout_ratio)
  print('[INFO] dense_2_dropout_ratio: ', dense_2_dropout_ratio)
  print('[INFO] detail_content_outdrop_3_ratio: ', detail_content_outdrop_3_ratio)
  print('[INFO] title_outdrop_3_ratio: ', title_outdrop_3_ratio)
  print('[INFO] abstract_content_outdrop_3_ratio: ', abstract_content_outdrop_3_ratio)
  print('[INFO] dense_4_dropout_ratio: ', dense_4_dropout_ratio)

  char_embedding_layer = CharEmbeddingV411(cfg, trainable=False, name='char_embedding')

  input = Input(shape=(max_length,), dtype='int32', name='input')

  embedded = char_embedding_layer(input)

  # block 1
  convoluted_11 = Conv1D(3, 3, 2, activation=conv_activation, padding='same', name='convoluted_11')(embedded)
  convoluted_21 = Conv1D(3, 5, 2, activation=conv_activation, padding='same', name='convoluted_21')(embedded)
  convoluted_31 = Conv1D(3, 7, 2, activation=conv_activation, padding='same', name='convoluted_31')(embedded)
  convoluted_41 = Conv1D(3, 11, 2, activation=conv_activation, padding='same', name='convoluted_41')(embedded)
  convoluted_51 = Conv1D(3, 17, 2, activation=conv_activation, padding='same', name='convoluted_41')(embedded)
  merged_conv_1 = Average(name='merged_conv_1')([convoluted_11, convoluted_21, convoluted_31, convoluted_51])

  # block 2
  convoluted_12 = Conv1D(7, 3, 3, activation=conv_activation, padding='same', name='convoluted_12')(merged_conv_1)
  convoluted_22 = Conv1D(13, 3, 3, activation=conv_activation, padding='same', name='convoluted_22')(convoluted_12)
  convoluted_32 = Conv1D(31, 3, 3, activation=conv_activation, padding='same', name='convoluted_32')(convoluted_22)
  convoluted_2 = Conv1D(43, 3, 3, activation=conv_activation, padding='same', name='convoluted_2')(convoluted_32)

  unit_convoluted_2 = Conv1D(1, 1, 1, activation=conv_activation, padding='same', name='unit_convoluted_2')(convoluted_2)
  conv_squeezed_2 = tf.squeeze(unit_convoluted_2, axis=2, name='conv_squeezed_2')

  aver_pooled_2 = AveragePooling1D(3, 3, padding='same', name='aver_pooled_2')(convoluted_2)
  aver_unit_convoluted_2 = Conv1D(1, 1, 1, activation=conv_activation, padding='same', name='aver_unit_convoluted_2')(aver_pooled_2)
  aver_squeezed_2 = tf.squeeze(aver_unit_convoluted_2, axis=2, name='aver_squeezed_2')

  max_max_pooled_2 = MaxPool1D(3, 3, padding='same', name='max_max_pooled_2')(convoluted_2)
  max_unit_convoluted_2 = Conv1D(1, 1, 1, activation=conv_activation, padding='same', name='max_unit_convoluted_2')(max_max_pooled_2)
  max_squeezed_2 = tf.squeeze(max_unit_convoluted_2, axis=2, name='max_squeezed_2')

  # block 3
  convoluted_13 = Conv1D(7, 3, 3, activation=conv_activation, padding='same', name='convoluted_13')(merged_conv_1)
  pooled_13 = AveragePooling1D(3, 3, padding='same', name='pooled_13')(convoluted_13)
  convoluted_23 = Conv1D(13, 3, 3, activation=conv_activation, padding='same', name='convoluted_23')(pooled_13)
  pooled_23 = AveragePooling1D(3, 3, padding='same', name='pooled_23')(convoluted_23)
  convoluted_3 = Conv1D(31, 3, 3, activation=conv_activation, padding='same', name='convoluted_3')(pooled_23)

  aver_pooled_3 = AveragePooling1D(3, 3, padding='same', name='aver_pooled_3')(convoluted_3)
  aver_unit_convoluted_3 = Conv1D(1, 1, 1, activation=conv_activation, padding='same', name='aver_unit_convoluted_3')(aver_pooled_3)
  aver_squeezed_3 = tf.squeeze(aver_unit_convoluted_3, axis=2, name='aver_squeezed_3')

  max_pooled_3 = MaxPool1D(3, 3, padding='same', name='max_pooled_3')(convoluted_3)
  max_unit_convoluted_3 = Conv1D(1, 1, 1, activation=conv_activation, padding='same', name='max_unit_convoluted_3')(max_pooled_3)
  max_squeezed_3 = tf.squeeze(max_unit_convoluted_3, axis=2, name='max_squeezed_3')

  # block 4
  convoluted_14 = Conv1D(7, 3, 3, activation=None, padding='same', name='convoluted_14')(merged_conv_1)
  convoluted_24 = Conv1D(13, 3, 3, activation=None, padding='same', name='convoluted_24')(convoluted_14)
  convoluted_34 = Conv1D(31, 3, 3, activation=None, padding='same', name='convoluted_34')(convoluted_24)
  convoluted_4 = Conv1D(43, 7, 5, activation=None, padding='same', name='convoluted_4')(convoluted_34)

  linear_unit_convoluted_4 = Conv1D(1, 1, 1, activation=None, padding='same', name='linear_unit_convoluted_4')(convoluted_4)
  linear_squeezed_4 = tf.squeeze(linear_unit_convoluted_4, axis=2, name='linear_squeezed_4')

  activated_unit_convoluted_4 = Conv1D(1, 1, 1, activation=conv_activation, padding='same', name='activated_unit_convoluted_4')(convoluted_4)
  activated_squeezed_4 = tf.squeeze(activated_unit_convoluted_4, axis=2, name='activated_squeezed_4')

  max_pooled_4 = MaxPool1D(3, 3, padding='same', name='max_pooled_4')(convoluted_4)
  max_unit_convoluted_4 = Conv1D(1, 1, 1, activation=conv_activation, padding='same', name='max_unit_convoluted_4')(max_pooled_4)
  max_squeezed_4 = tf.squeeze(max_unit_convoluted_4, axis=2, name='max_squeezed_4')

  merged_convs = concatenate([
      conv_squeezed_2, aver_squeezed_2, max_squeezed_2,
      aver_squeezed_3, max_squeezed_3,
      linear_squeezed_4, activated_squeezed_4, max_squeezed_4,
    ], name='merged_convs')

  # Depth lv1
  det_content_decoder_dendrop_1 = Dropout(rate=dense_1_dropout_ratio, name='det_content_decoder_dendrop_1')(merged_convs)
  det_content_decoder_dense_1 = Dense(dense_1_size, name='det_content_decoder_dense_1', kernel_regularizer='l2', activation=dense_activation)(det_content_decoder_dendrop_1)
  det_content_decoder_dense_1_norm = LayerNormalization(trainable=True, name='det_content_decoder_dense_1_norm')(det_content_decoder_dense_1)

  title_decoder_dendrop_1 = Dropout(rate=dense_1_dropout_ratio, name='title_decoder_dendrop_1')(merged_convs)
  title_decoder_dense_1 = Dense(dense_1_size, name='title_decoder_dense_1', kernel_regularizer='l2', activation=dense_activation)(title_decoder_dendrop_1)
  title_decoder_dense_1_norm = LayerNormalization(trainable=True, name='title_decoder_dense_1_norm')(title_decoder_dense_1)

  abs_content_decoder_dendrop_1 = Dropout(rate=dense_1_dropout_ratio, name='abs_content_decoder_dendrop_1')(merged_convs)
  abs_content_decoder_dense_1 = Dense(dense_1_size, name='abs_content_decoder_dense_1', kernel_regularizer='l2', activation=dense_activation)(abs_content_decoder_dendrop_1)
  abs_content_decoder_dense_1_norm = LayerNormalization(trainable=True, name='abs_content_decoder_dense_1_norm')(abs_content_decoder_dense_1)

  # Depth lv2
  det_content_dendrop_2 = Dropout(rate=dense_2_dropout_ratio, name='det_content_dendrop_2')(det_content_decoder_dense_1_norm)
  det_content_dense_2 = Dense(dense_2_size, name='det_content_dense_2', kernel_regularizer='l2', activation=dense_activation)(det_content_dendrop_2)
  det_content_dense_2_norm = LayerNormalization(trainable=True, name='det_content_dense_2_norm')(det_content_dense_2)

  title_dendrop_2 = Dropout(rate=dense_2_dropout_ratio, name='title_dendrop_2')(title_decoder_dense_1_norm)
  title_dense_2 = Dense(dense_2_size, name='title_dense_2', kernel_regularizer='l2', activation=dense_activation)(title_dendrop_2)
  title_dense_2_norm = LayerNormalization(trainable=True, name='title_dense_2_norm')(title_dense_2)

  abs_content_dendrop_2 = Dropout(rate=dense_2_dropout_ratio, name='abs_content_dendrop_2')(abs_content_decoder_dense_1_norm)
  abs_content_dense_2 = Dense(dense_2_size, name='abs_content_dense_2', kernel_regularizer='l2', activation=dense_activation)(abs_content_dendrop_2)
  abs_content_dense_2_norm = LayerNormalization(trainable=True, name='abs_content_dense_2_norm')(abs_content_dense_2)

  # Depth lv3
  content_concatenated_3 = concatenate([det_content_decoder_dense_1_norm, det_content_dense_2_norm], name='content_concatenated_3')
  detail_content_dendrop_3 = Dropout(rate=detail_content_outdrop_3_ratio, name='detail_content_dendrop_3')(content_concatenated_3)
  detail_content_dense_3 = Dense(detail_content_dense_3_size, name='detail_content_dense_3', kernel_regularizer='l2', activation=detail_content_dense_activation)(detail_content_dendrop_3)
  detail_content_dense_norm_3 = LayerNormalization(trainable=True, name='detail_content_dense_norm_3')(detail_content_dense_3)
  detail_content_output = Dense(num_categories, name='detail_content_output', activation='softmax')(detail_content_dense_norm_3)

  title_concatenated_3 = concatenate([title_decoder_dense_1_norm, title_dense_2_norm], name='title_concatenated_3')
  detail_title_dendrop_3 = Dropout(rate=title_outdrop_3_ratio, name='detail_title_dendrop_3')(title_concatenated_3)
  detail_title_dense_3 = Dense(title_dense_3_size, name='detail_title_dense_3', kernel_regularizer='l2', activation=detail_title_dense_activation)(detail_title_dendrop_3)
  detail_title_dense_norm_3 = LayerNormalization(trainable=True, name='detail_title_dense_norm_3')(detail_title_dense_3)
  detail_title_output = Dense(num_categories, name='detail_title_output', activation='softmax')(detail_title_dense_norm_3)

  abs_content_dendrop_3 = Dropout(rate=abstract_content_outdrop_3_ratio, name='abs_content_dendrop_3')(abs_content_dense_2_norm)
  abs_content_dense_3 = Dense(abs_content_dense_3_size, name='abs_content_dense_3', kernel_regularizer='l2', activation=dense_activation)(abs_content_dendrop_3)
  abs_dense_3_norm = LayerNormalization(trainable=True, name='abs_dense_3_norm')(abs_content_dense_3)
  abstract_content_dendrop = Dropout(rate=dense_4_dropout_ratio, name='abstract_content_dendrop')(abs_dense_3_norm)
  abstract_content_dense_4 = Dense(dense_4_size, name='abstract_content_dense_4', kernel_regularizer='l2', activation=abstract_content_dense_activation)(abstract_content_dendrop)
  abstract_content_dense_norm_4 = LayerNormalization(trainable=True, name='abstract_content_dense_norm_4')(abstract_content_dense_4)
  abstract_content_output = Dense(num_categories, name='abstract_content_output', activation='softmax')(abstract_content_dense_norm_4)

  model = Model(inputs=[input], outputs=[abstract_content_output, detail_content_output, detail_title_output])

  Optimizer = cfg['optimizer']
  print('[INFO] optimizer: ', Optimizer)

  optimizer = Optimizer(learning_rate=lr)

  # learning_rate_fn = InverseTimeDecay(
  #     initial_learning_rate=lr,
  #     decay_steps=decay_steps,
  #     decay_rate=decay_rate,
  #     staircase=True
  #   )
  # optimizer = Nadam(learning_rate=lr)
  # optimizer = Adam(learning_rate=learning_rate_fn)
  # optimizer = RMSprop(learning_rate=learning_rate_fn)
  # optimizer = Adagrad(learning_rate=learning_rate_fn)

  model.compile(
      loss='categorical_crossentropy',
      optimizer=optimizer,
      metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy')],
    )

  return model


def HierarchyV3_150200(cfg, learning_rate=None):
  num_categories = cfg['num_categories']
  num_classes = cfg['num_classes']
  max_length = cfg['max_length']
  lr = learning_rate if learning_rate else cfg['learning_rate']

  conv_activation = 'swish'
  dense_activation = 'swish'
  detail_content_dense_activation = 'swish'
  detail_title_dense_activation = 'swish'
  abstract_content_dense_activation = 'swish'

  decoder_dense_size_1 = 29

  dense_2_dropout_ratio = 0.43
  dense_2_size = 17
  dense_3_dropout_ratio = 0.29
  dense_3_size = 11

  detail_content_outdrop_4_ratio = 0.20
  detail_title_outdrop_4_ratio = 0.18
  abstract_content_outdrop_4_ratio = 0.15
  dense_4_size = 7

  print('[ATTENTION!] Model version: v3.15.2.0')
  print('[INFO] dense_2_dropout_ratio: ', dense_2_dropout_ratio)
  print('[INFO] dense_3_dropout_ratio: ', dense_3_dropout_ratio)
  print('[INFO] detail_content_outdrop_4_ratio: ', detail_content_outdrop_4_ratio)
  print('[INFO] detail_title_outdrop_4_ratio: ', detail_title_outdrop_4_ratio)
  print('[INFO] abstract_content_outdrop_4_ratio: ', abstract_content_outdrop_4_ratio)

  char_embedding_layer = CharEmbeddingV411(cfg, trainable=False, name='char_embedding')

  input = Input(shape=(max_length,), dtype='int32', name='input')

  embedded = char_embedding_layer(input)

  # block 1
  convoluted_11 = Conv1D(3, 3, 2, activation=conv_activation, padding='same', name='convoluted_11')(embedded)
  convoluted_21 = Conv1D(3, 5, 2, activation=conv_activation, padding='same', name='convoluted_21')(embedded)
  convoluted_31 = Conv1D(3, 7, 2, activation=conv_activation, padding='same', name='convoluted_31')(embedded)
  convoluted_41 = Conv1D(3, 11, 2, activation=conv_activation, padding='same', name='convoluted_41')(embedded)
  merged_conv_1 = Maximum(name='merged_conv_1')([convoluted_11, convoluted_21, convoluted_31, convoluted_41])

  # block 2
  convoluted_12 = Conv1D(7, 3, 3, activation=conv_activation, padding='same', name='convoluted_12')(merged_conv_1)
  convoluted_22 = Conv1D(11, 7, 5, activation=conv_activation, padding='same', name='convoluted_22')(convoluted_12)
  convoluted_32 = Conv1D(17, 7, 5, activation=conv_activation, padding='same', name='convoluted_32')(convoluted_22)

  unit_convoluted_42 = Conv1D(1, 1, 1, activation=conv_activation, padding='same', name='unit_convoluted_42')(convoluted_32)
  conv_squeezed_42 = tf.squeeze(unit_convoluted_42, axis=2, name='conv_squeezed_42')

  aver_pooled_42 = AveragePooling1D(3, 3, padding='same', name='aver_pooled_42')(convoluted_32)
  aver_unit_convoluted_42 = Conv1D(1, 1, 1, activation=conv_activation, padding='same', name='aver_unit_convoluted_42')(aver_pooled_42)
  aver_squeezed_42 = tf.squeeze(aver_unit_convoluted_42, axis=2, name='aver_squeezed_42')

  max_max_pooled_42 = MaxPool1D(3, 3, padding='same', name='max_max_pooled_42')(convoluted_32)
  max_unit_convoluted_42 = Conv1D(1, 1, 1, activation=conv_activation, padding='same', name='max_unit_convoluted_42')(max_max_pooled_42)
  max_squeezed_42 = tf.squeeze(max_unit_convoluted_42, axis=2, name='max_squeezed_42')

  # block 3
  convoluted_13 = Conv1D(7, 3, 3, activation=conv_activation, padding='same', name='convoluted_13')(merged_conv_1)
  pooled_13 = AveragePooling1D(3, 3, padding='same', name='pooled_13')(convoluted_13)
  convoluted_23 = Conv1D(11, 3, 3, activation=conv_activation, padding='same', name='convoluted_23')(pooled_13)
  pooled_23 = AveragePooling1D(3, 3, padding='same', name='pooled_23')(convoluted_23)
  convoluted_33 = Conv1D(17, 7, 5, activation=conv_activation, padding='same', name='convoluted_33')(pooled_23)
  pooled_33 = AveragePooling1D(7, 5, padding='same', name='pooled_33')(convoluted_33)

  aver_pooled_43 = AveragePooling1D(3, 3, padding='same', name='aver_pooled_43')(pooled_33)
  aver_unit_convoluted_43 = Conv1D(1, 1, 1, activation=conv_activation, padding='same', name='aver_unit_convoluted_43')(aver_pooled_43)
  aver_squeezed_43 = tf.squeeze(aver_unit_convoluted_43, axis=2, name='aver_squeezed_43')

  max_pooled_43 = MaxPool1D(3, 3, padding='same', name='max_pooled_43')(pooled_33)
  max_unit_convoluted_43 = Conv1D(1, 1, 1, activation=conv_activation, padding='same', name='max_unit_convoluted_43')(max_pooled_43)
  max_squeezed_43 = tf.squeeze(max_unit_convoluted_43, axis=2, name='max_squeezed_43')

  # block 4
  convoluted_14 = Conv1D(7, 7, 5, activation=None, padding='same', name='convoluted_14')(merged_conv_1)
  convoluted_24 = Conv1D(11, 7, 5, activation=None, padding='same', name='convoluted_24')(convoluted_14)
  convoluted_34 = Conv1D(17, 7, 5, activation=None, padding='same', name='convoluted_34')(convoluted_24)

  # linear_unit_convoluted_64 = Conv1D(1, 1, 1, activation=None, padding='same', name='linear_unit_convoluted_64')(convoluted_34)
  # linear_squeezed_64 = tf.squeeze(linear_unit_convoluted_64, axis=2, name='linear_squeezed_64')

  activated_unit_convoluted_64 = Conv1D(1, 1, 1, activation=conv_activation, padding='same', name='activated_unit_convoluted_64')(convoluted_34)
  activated_squeezed_64 = tf.squeeze(activated_unit_convoluted_64, axis=2, name='activated_squeezed_64')

  max_pooled_64 = MaxPool1D(3, 3, padding='same', name='max_pooled_64')(convoluted_34)
  max_unit_convoluted_64 = Conv1D(1, 1, 1, activation=conv_activation, padding='same', name='max_unit_convoluted_64')(max_pooled_64)
  max_squeezed_64 = tf.squeeze(max_unit_convoluted_64, axis=2, name='max_squeezed_64')

  merged_convs = concatenate([
      conv_squeezed_42, aver_squeezed_42, max_squeezed_42,
      aver_squeezed_43, max_squeezed_43,
      activated_squeezed_64, max_squeezed_64,
    ], name='merged_convs')

  decoder_dense_1 = Dense(decoder_dense_size_1, name='decoder_dense_1', kernel_regularizer='l2', activation=dense_activation)(merged_convs)

  content_dendrop_2 = Dropout(rate=dense_2_dropout_ratio, name='content_dendrop_2')(decoder_dense_1)
  content_dense_2 = Dense(dense_2_size, name='content_dense_2', kernel_regularizer='l2', activation=dense_activation)(content_dendrop_2)
  content_dendrop_3 = Dropout(rate=dense_3_dropout_ratio, name='content_dendrop_3')(content_dense_2)
  content_dense_3 = Dense(dense_3_size, name='content_dense_3', kernel_regularizer='l2', activation=dense_activation)(content_dendrop_3)
  dense_3_norm = LayerNormalization(trainable=True, name='dense_3_norm')(content_dense_3)

  title_dendrop_2 = Dropout(rate=dense_2_dropout_ratio, name='title_dendrop_2')(decoder_dense_1)
  title_dense_2 = Dense(dense_2_size, name='title_dense_2', kernel_regularizer='l2', activation=dense_activation)(title_dendrop_2)
  title_dendrop_3 = Dropout(rate=dense_3_dropout_ratio, name='title_dendrop_3')(title_dense_2)
  title_dense_3 = Dense(dense_3_size, name='title_dense_3', kernel_regularizer='l2', activation=dense_activation)(title_dendrop_3)

  content_concatenated_4 = concatenate([content_dense_2, content_dense_3], name='content_concatenated_4')
  detail_content_dendrop_4 = Dropout(rate=detail_content_outdrop_4_ratio, name='detail_content_dendrop_4')(content_concatenated_4)
  detail_content_dense_4 = Dense(dense_4_size, name='detail_content_dense_4', kernel_regularizer='l2', activation=detail_content_dense_activation)(detail_content_dendrop_4)
  detail_content_dense_norm_4 = LayerNormalization(trainable=True, name='detail_content_dense_norm_4')(detail_content_dense_4)
  detail_content_output = Dense(num_categories, name='detail_content_output', activation='softmax')(detail_content_dense_norm_4)

  abstract_content_dendrop = Dropout(rate=abstract_content_outdrop_4_ratio, name='abstract_content_dendrop')(dense_3_norm)
  abstract_content_dense_4 = Dense(dense_4_size, name='abstract_content_dense_4', kernel_regularizer='l2', activation=abstract_content_dense_activation)(abstract_content_dendrop)
  abstract_content_dense_norm_4 = LayerNormalization(trainable=True, name='abstract_content_dense_norm_4')(abstract_content_dense_4)
  abstract_content_output = Dense(num_categories, name='abstract_content_output', activation='softmax')(abstract_content_dense_norm_4)

  title_concatenated_4 = concatenate([title_dense_2, title_dense_3], name='title_concatenated_4')
  detail_title_dendrop_4 = Dropout(rate=detail_title_outdrop_4_ratio, name='detail_title_dendrop_4')(title_concatenated_4)
  detail_title_dense_4 = Dense(dense_4_size, name='detail_title_dense_4', kernel_regularizer='l2', activation=detail_title_dense_activation)(detail_title_dendrop_4)
  detail_title_dense_norm_4 = LayerNormalization(trainable=True, name='detail_title_dense_norm_4')(detail_title_dense_4)
  detail_title_output = Dense(num_categories, name='detail_title_output', activation='softmax')(detail_title_dense_norm_4)

  model = Model(inputs=[input], outputs=[abstract_content_output, detail_content_output, detail_title_output])

  Optimizer = cfg['optimizer']
  print('[INFO] optimizer: ', Optimizer)

  optimizer = Optimizer(learning_rate=lr)

  # learning_rate_fn = InverseTimeDecay(
  #     initial_learning_rate=lr,
  #     decay_steps=decay_steps,
  #     decay_rate=decay_rate,
  #     staircase=True
  #   )
  # optimizer = Nadam(learning_rate=lr)
  # optimizer = Adam(learning_rate=learning_rate_fn)
  # optimizer = RMSprop(learning_rate=learning_rate_fn)
  # optimizer = Adagrad(learning_rate=learning_rate_fn)

  model.compile(
      loss='categorical_crossentropy',
      optimizer=optimizer,
      metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy')],
    )

  return model
