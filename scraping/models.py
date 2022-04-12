import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adagrad, Nadam
from tensorflow.keras.optimizers.schedules import InverseTimeDecay
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM, Bidirectional, Dropout, LayerNormalization, Layer, BatchNormalization
from tensorflow.keras.layers import concatenate, Reshape, SpatialDropout1D, Conv1D, Flatten, AveragePooling1D, MaxPool1D, Average, Maximum, Multiply, Add, Minimum
from tensorflow.keras.models import Model, Sequential

from tensorflow import config as config
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import TensorBoard, Callback


def Hierarchy(cfg, learning_rate=None):
  num_categories = cfg['num_categories']
  num_classes = cfg['num_classes']
  max_length = cfg['max_length']
  lr = learning_rate if learning_rate else cfg['learning_rate']
  emb_trainable = cfg['emb_trainable']
  decoder_trainable = cfg['decoder_trainable']

  conv_activation = 'swish'
  dense_activation = 'swish'
  detail_content_dense_activation = 'swish'
  detail_title_dense_activation = 'swish'
  abstract_content_dense_activation = 'swish'

  dense_1_size = 64
  dense_2_size = 32
  detail_content_dense_3_size = 8
  title_dense_3_size = 8
  abs_content_dense_3_size = 16
  dense_4_size = 8

  fine_tuning = cfg.get('dropout_fine_tuning', 0)
  dense_1_dropout_ratio = 0.75 - fine_tuning
  dense_2_dropout_ratio = 0.50 - fine_tuning
  detail_content_outdrop_3_ratio = 0.50 - fine_tuning
  title_outdrop_3_ratio = 0.50 - fine_tuning
  abstract_content_outdrop_3_ratio = 0.25 - fine_tuning
  dense_4_dropout_ratio = 0.25 - fine_tuning

  print('[ATTENTION!] Model version: v3.22.0.0')
  print('[ATTENTION!] emb_trainable: ', emb_trainable)
  print('[ATTENTION!] decoder_trainable: ', decoder_trainable)
  print('[INFO] learning_rate: ', lr)
  print('[INFO] dropout_fine_tuning: ', fine_tuning)
  print('[INFO] dense_1_dropout_ratio: ', dense_1_dropout_ratio)
  print('[INFO] dense_2_dropout_ratio: ', dense_2_dropout_ratio)
  print('[INFO] detail_content_outdrop_3_ratio: ', detail_content_outdrop_3_ratio)
  print('[INFO] title_outdrop_3_ratio: ', title_outdrop_3_ratio)
  print('[INFO] abstract_content_outdrop_3_ratio: ', abstract_content_outdrop_3_ratio)
  print('[INFO] dense_4_dropout_ratio: ', dense_4_dropout_ratio)

  char_embedding_layer = CharEmbeddingV5(cfg, trainable=emb_trainable, name='char_embedding')

  input = Input(shape=(max_length,), dtype='int32', name='input')

  one = tf.one_hot(input, num_classes)
  embedded = char_embedding_layer(input)

  # block 1
  convoluted_one_11 = Conv1D(7, 3, 1, activation=conv_activation, padding='same', name='convoluted_one_11')(one)
  pooled_one_11 = AveragePooling1D(3, 3, padding='same', name='pooled_one_11')(convoluted_one_11)
  convoluted_11 = Conv1D(7, 3, 1, activation=conv_activation, padding='same', name='convoluted_11')(embedded)
  pooled_11 = AveragePooling1D(3, 3, padding='same', name='pooled_11')(convoluted_11)
  convoluted_one_21 = Conv1D(7, 5, 1, activation=conv_activation, padding='same', name='convoluted_one_21')(one)
  pooled_one_21 = AveragePooling1D(5, 3, padding='same', name='pooled_one_21')(convoluted_one_21)
  convoluted_21 = Conv1D(7, 5, 1, activation=conv_activation, padding='same', name='convoluted_21')(embedded)
  pooled_21 = AveragePooling1D(5, 3, padding='same', name='pooled_21')(convoluted_21)
  # convoluted_31 = Conv1D(7, 7, 1, activation=conv_activation, padding='same', name='convoluted_31')(embedded)
  # pooled_31 = AveragePooling1D(5, 3, padding='same', name='pooled_31')(convoluted_31)
  # convoluted_41 = Conv1D(7, 11, 1, activation=conv_activation, padding='same', name='convoluted_41')(embedded)
  # pooled_41 = AveragePooling1D(5, 3, padding='same', name='pooled_41')(convoluted_41)

  # block 2
  merged_conv_12 = Maximum(name='merged_conv_12')([pooled_one_11, pooled_11, pooled_one_21, pooled_21])
  convoluted_12 = Conv1D(13, 3, 3, activation=conv_activation, padding='same', name='convoluted_12')(merged_conv_12)
  pooled_12 = MaxPool1D(3, 3, padding='same', name='pooled_12')(convoluted_12)

  convoluted_22 = Conv1D(19, 3, 3, activation=conv_activation, padding='same', name='convoluted_22')(pooled_12)
  pooled_22 = MaxPool1D(3, 3, padding='same', name='pooled_22')(convoluted_22)
  unit_convoluted_22 = Conv1D(1, 1, 1, activation=None, padding='same', name='unit_convoluted_22')(pooled_22)
  squeezed_22 = tf.squeeze(unit_convoluted_22, axis=2)

  convoluted_32 = Conv1D(25, 3, 3, activation=conv_activation, padding='same', name='convoluted_32')(pooled_22)
  pooled_32 = MaxPool1D(3, 3, padding='same', name='pooled_32')(convoluted_32)
  unit_convoluted_32 = Conv1D(1, 1, 1, activation=None, padding='same', name='unit_convoluted_32')(pooled_32)
  squeezed_32 = tf.squeeze(unit_convoluted_32, axis=2)

  convoluted_42 = Conv1D(31, 3, 3, activation=conv_activation, padding='same', name='convoluted_42')(pooled_32)
  pooled_42 = MaxPool1D(3, 3, padding='same', name='pooled_42')(convoluted_42)
  unit_convoluted_42 = Conv1D(1, 1, 1, activation=None, padding='same', name='unit_convoluted_42')(pooled_42)
  squeezed_42 = tf.squeeze(unit_convoluted_42, axis=2)

  # block 3
  merged_conv_13 = Average(name='merged_conv_13')([pooled_one_11, pooled_11, pooled_one_21, pooled_21])
  convoluted_13 = Conv1D(13, 3, 3, activation=conv_activation, padding='same', name='convoluted_13')(merged_conv_13)
  pooled_13 = AveragePooling1D(3, 3, padding='same', name='pooled_13')(convoluted_13)
  convoluted_23 = Conv1D(19, 3, 3, activation=conv_activation, padding='same', name='convoluted_23')(pooled_13)
  pooled_23 = AveragePooling1D(3, 3, padding='same', name='pooled_23')(convoluted_23)

  convoluted_33 = Conv1D(25, 3, 3, activation=conv_activation, padding='same', name='convoluted_33')(pooled_23)
  pooled_33 = AveragePooling1D(3, 3, padding='same', name='pooled_33')(convoluted_33)
  unit_convoluted_33 = Conv1D(1, 1, 1, activation=None, padding='same', name='unit_convoluted_33')(pooled_33)
  squeezed_33 = tf.squeeze(unit_convoluted_33, axis=2)

  convoluted_43 = Conv1D(31, 3, 3, activation=conv_activation, padding='same', name='convoluted_43')(pooled_33)
  pooled_43 = AveragePooling1D(3, 3, padding='same', name='pooled_43')(convoluted_43)
  unit_convoluted_43 = Conv1D(1, 1, 1, activation=None, padding='same', name='unit_convoluted_43')(pooled_43)
  squeezed_43 = tf.squeeze(unit_convoluted_43, axis=2)

  # block 4
  merged_conv_14 = Average(name='merged_conv_14')([pooled_one_11, pooled_one_21])
  convoluted_14 = Conv1D(13, 7, 1, activation=conv_activation, padding='same', name='convoluted_14')(merged_conv_14)
  pooled_14 = MaxPool1D(7, 7, padding='same', name='pooled_14')(convoluted_14)
  convoluted_24 = Conv1D(19, 7, 7, activation=conv_activation, padding='same', name='convoluted_24')(pooled_14)
  pooled_24 = MaxPool1D(7, 7, padding='same', name='pooled_24')(convoluted_24)
  # convoluted_34 = Conv1D(31, 5, 5, activation=conv_activation, padding='same', name='convoluted_34')(pooled_24)
  # pooled_34 = MaxPool1D(5, 5, padding='same', name='pooled_34')(convoluted_34)

  unit_convoluted_34 = Conv1D(1, 1, 1, activation=None, padding='same', name='unit_convoluted_34')(pooled_24)
  squeezed_34 = tf.squeeze(unit_convoluted_34, axis=2)

  # block 5
  merged_convs = concatenate([
    squeezed_22, squeezed_32, squeezed_42,
    squeezed_33, squeezed_43,
    squeezed_34], name='merged_convs')

  # Depth lv1
  det_content_decoder_dendrop_1 = Dropout(rate=dense_1_dropout_ratio, name='det_content_decoder_dendrop_1')(merged_convs)
  det_content_decoder_dense_1 = Dense(
      dense_1_size,
      name='det_content_decoder_dense_1',
      kernel_regularizer='l2',
      activation=dense_activation,
      trainable=decoder_trainable,
    )(det_content_decoder_dendrop_1)
  det_content_decoder_dense_1_norm = LayerNormalization(
      trainable=decoder_trainable,
      name='det_content_decoder_dense_1_norm'
    )(det_content_decoder_dense_1)

  abs_content_decoder_dendrop_1 = Dropout(rate=dense_1_dropout_ratio, name='abs_content_decoder_dendrop_1')(merged_convs)
  abs_content_decoder_dense_1 = Dense(
      dense_1_size,
      name='abs_content_decoder_dense_1',
      kernel_regularizer='l2',
      activation=dense_activation,
      trainable=decoder_trainable,
    )(abs_content_decoder_dendrop_1)
  abs_content_decoder_dense_1_norm = LayerNormalization(
      trainable=decoder_trainable,
      name='abs_content_decoder_dense_1_norm'
    )(abs_content_decoder_dense_1)

  # Depth lv2
  det_content_dendrop_2 = Dropout(rate=dense_2_dropout_ratio, name='det_content_dendrop_2')(det_content_decoder_dense_1_norm)
  det_content_dense_2 = Dense(
      dense_2_size,
      name='det_content_dense_2',
      kernel_regularizer='l2',
      activation=dense_activation,
      trainable=decoder_trainable,
    )(det_content_dendrop_2)
  det_content_dense_2_norm = LayerNormalization(
      trainable=decoder_trainable,
      name='det_content_dense_2_norm',
    )(det_content_dense_2)

  abs_content_dendrop_2 = Dropout(rate=dense_2_dropout_ratio, name='abs_content_dendrop_2')(abs_content_decoder_dense_1_norm)
  abs_content_dense_2 = Dense(dense_2_size,
      name='abs_content_dense_2',
      kernel_regularizer='l2',
      activation=dense_activation,
      trainable=decoder_trainable,
    )(abs_content_dendrop_2)
  abs_content_dense_2_norm = LayerNormalization(
      trainable=decoder_trainable,
      name='abs_content_dense_2_norm'
    )(abs_content_dense_2)

  # Depth lv3
  content_concatenated_3 = concatenate([det_content_decoder_dense_1_norm, det_content_dense_2_norm], name='content_concatenated_3')
  detail_content_dendrop_3 = Dropout(rate=detail_content_outdrop_3_ratio, name='detail_content_dendrop_3')(content_concatenated_3)
  detail_content_dense_3 = Dense(
      detail_content_dense_3_size,
      name='detail_content_dense_3',
      kernel_regularizer='l2',
      activation=detail_content_dense_activation,
      trainable=decoder_trainable,
    )(detail_content_dendrop_3)
  detail_content_dense_norm_3 = LayerNormalization(
      trainable=decoder_trainable,
      name='detail_content_dense_norm_3',
    )(detail_content_dense_3)
  detail_content_output = Dense(
      num_categories,
      name='detail_content_output',
      activation='softmax',
      trainable=decoder_trainable,
    )(detail_content_dense_norm_3)
  detail_title_dense_norm_3 = LayerNormalization(
      trainable=decoder_trainable,
      name='detail_title_dense_norm_3',
    )(detail_content_dense_3)
  detail_title_output = Dense(
      num_categories,
      name='detail_title_output',
      activation='softmax',
      trainable=decoder_trainable,
    )(detail_title_dense_norm_3)


  abs_content_dendrop_3 = Dropout(rate=abstract_content_outdrop_3_ratio, name='abs_content_dendrop_3')(abs_content_dense_2_norm)
  abs_content_dense_3 = Dense(
      abs_content_dense_3_size,
      name='abs_content_dense_3',
      kernel_regularizer='l2',
      activation=dense_activation,
      trainable=decoder_trainable,
    )(abs_content_dendrop_3)
  abs_dense_3_norm = LayerNormalization(
      trainable=decoder_trainable,
      name='abs_dense_3_norm',
    )(abs_content_dense_3)
  abstract_content_dendrop = Dropout(rate=dense_4_dropout_ratio, name='abstract_content_dendrop')(abs_dense_3_norm)
  abstract_content_dense_4 = Dense(
      dense_4_size,
      name='abstract_content_dense_4',
      kernel_regularizer='l2',
      activation=abstract_content_dense_activation,
      trainable=decoder_trainable,
    )(abstract_content_dendrop)
  abstract_content_dense_norm_4 = LayerNormalization(
      trainable=decoder_trainable,
      name='abstract_content_dense_norm_4')(abstract_content_dense_4)
  abstract_content_output = Dense(
      num_categories,
      name='abstract_content_output',
      activation='softmax',
      trainable=decoder_trainable,
    )(abstract_content_dense_norm_4)

  model = Model(inputs=[input], outputs=[abstract_content_output, detail_content_output, detail_title_output])

  Optimizer = cfg['optimizer']
  print('[INFO] optimizer: ', Optimizer)

  optimizer = Optimizer(learning_rate=lr)

  model.compile(
      loss='categorical_crossentropy',
      optimizer=optimizer,
      metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy')],
    )

  return model

class CharEmbeddingV411(Layer):
  def __init__(self, cfg, trainable=True, **kwargs):
    print('[INFO] Using CharEmbeddingV411')

    num_classes = cfg['num_classes']
    print('[DEBUG] num_classes: ', num_classes)
    max_length = cfg['max_length']
    dim_embeddings = 11

    self.trainable = trainable
    self.output_size = 11
    self.activation = 'swish'
    self.dense_output_activation = 'tanh'

    self.embedding_layer = Embedding(num_classes, dim_embeddings, input_length=max_length, trainable=self.trainable, name='embedding')

    self.dense_112_layer = Dense(2, name='dense_112', trainable=self.trainable, activation=self.activation)
    self.dense_113_layer = Dense(3, name='dense_113', trainable=self.trainable, activation=self.activation)
    self.dense_115_layer = Dense(5, name='dense_115', trainable=self.trainable, activation=self.activation)
    self.dense_117_layer = Dense(7, name='dense_117', trainable=self.trainable, activation=self.activation)

    self.dense_21_layer = Dense(1, name='dense_21', trainable=self.trainable, activation=self.activation)
    self.dense_31_layer = Dense(1, name='dense_31', trainable=self.trainable, activation=self.activation)
    self.dense_51_layer = Dense(1, name='dense_51', trainable=self.trainable, activation=self.activation)
    self.dense_71_layer = Dense(1, name='dense_71', trainable=self.trainable, activation=self.activation)
    self.dense_111_layer = Dense(1, name='dense_111', trainable=self.trainable, activation=self.activation)

    self.add_23_layer = Add(name='add_23')
    self.add_25_layer = Add(name='add_25')
    self.add_27_layer = Add(name='add_27')
    self.add_211_layer = Add(name='add_211')
    self.add_35_layer = Add(name='add_35')
    self.add_37_layer = Add(name='add_37')
    self.add_311_layer = Add(name='add_117')
    self.add_57_layer = Add(name='add_57')
    self.add_511_layer = Add(name='add_511')
    self.add_711_layer = Add(name='add_711')

    self.mul_23_layer = Multiply(name='mul_23')
    self.mul_25_layer = Multiply(name='mul_25')
    self.mul_27_layer = Multiply(name='mul_27')
    self.mul_211_layer = Multiply(name='mul_211')
    self.mul_35_layer = Multiply(name='mul_35')
    self.mul_37_layer = Multiply(name='mul_37')
    self.mul_311_layer = Multiply(name='mul_117')
    self.mul_57_layer = Multiply(name='mul_57')
    self.mul_511_layer = Multiply(name='mul_511')
    self.mul_711_layer = Multiply(name='mul_711')

    self.min_23_layer = Minimum(name='min_23')
    self.min_25_layer = Minimum(name='min_25')
    self.min_27_layer = Minimum(name='min_27')
    self.min_211_layer = Minimum(name='min_211')
    self.min_35_layer = Minimum(name='min_35')
    self.min_37_layer = Minimum(name='min_37')
    self.min_311_layer = Minimum(name='min_117')
    self.min_57_layer = Minimum(name='min_57')
    self.min_511_layer = Minimum(name='min_511')
    self.min_711_layer = Minimum(name='min_711')

    self.max_23_layer = Maximum(name='max_23')
    self.max_25_layer = Maximum(name='max_25')
    self.max_27_layer = Maximum(name='max_27')
    self.max_211_layer = Maximum(name='max_211')
    self.max_35_layer = Maximum(name='max_35')
    self.max_37_layer = Maximum(name='max_37')
    self.max_311_layer = Maximum(name='max_117')
    self.max_57_layer = Maximum(name='max_57')
    self.max_511_layer = Maximum(name='max_511')
    self.max_711_layer = Maximum(name='max_711')

    self.dense_output_layer = Dense(self.output_size, name='dense_output', trainable=self.trainable, activation=self.dense_output_activation)

    super(CharEmbeddingV411, self).__init__(**kwargs)

  def call(self, inputs):
    embedded = self.embedding_layer(inputs)

    dense_112 = self.dense_112_layer(embedded)
    dense_113 = self.dense_113_layer(embedded)
    dense_115 = self.dense_115_layer(embedded)
    dense_117 = self.dense_117_layer(embedded)

    dense_21 = self.dense_21_layer(dense_112)
    dense_31 = self.dense_31_layer(dense_113)
    dense_51 = self.dense_51_layer(dense_115)
    dense_71 = self.dense_71_layer(dense_117)
    dense_111 = self.dense_111_layer(embedded)

    add_23 = self.add_23_layer([dense_21, dense_31])
    add_25 = self.add_25_layer([dense_21, dense_51])
    add_27 = self.add_27_layer([dense_21, dense_71])
    add_211 = self.add_211_layer([dense_21, dense_111])
    add_35 = self.add_35_layer([dense_31, dense_51])
    add_37 = self.add_37_layer([dense_31, dense_71])
    add_311 = self.add_311_layer([dense_31, dense_111])
    add_57 = self.add_57_layer([dense_51, dense_71])
    add_511 = self.add_511_layer([dense_51, dense_111])
    add_711 = self.add_711_layer([dense_71, dense_111])

    mul_23 = self.mul_23_layer([dense_21, dense_31])
    mul_25 = self.mul_25_layer([dense_21, dense_51])
    mul_27 = self.mul_27_layer([dense_21, dense_71])
    mul_211 = self.mul_211_layer([dense_21, dense_111])
    mul_35 = self.mul_35_layer([dense_31, dense_51])
    mul_37 = self.mul_37_layer([dense_31, dense_71])
    mul_311 = self.mul_311_layer([dense_31, dense_111])
    mul_57 = self.mul_57_layer([dense_51, dense_71])
    mul_511 = self.mul_511_layer([dense_51, dense_111])
    mul_711 = self.mul_711_layer([dense_71, dense_111])

    min_23 = self.min_23_layer([dense_21, dense_31])
    min_25 = self.min_25_layer([dense_21, dense_51])
    min_27 = self.min_27_layer([dense_21, dense_71])
    min_211 = self.min_211_layer([dense_21, dense_111])
    min_35 = self.min_35_layer([dense_31, dense_51])
    min_37 = self.min_37_layer([dense_31, dense_71])
    min_311 = self.min_311_layer([dense_31, dense_111])
    min_57 = self.min_57_layer([dense_51, dense_71])
    min_511 = self.min_511_layer([dense_51, dense_111])
    min_711 = self.min_711_layer([dense_71, dense_111])

    max_23 = self.max_23_layer([dense_21, dense_31])
    max_25 = self.max_25_layer([dense_21, dense_51])
    max_27 = self.max_27_layer([dense_21, dense_71])
    max_211 = self.max_211_layer([dense_21, dense_111])
    max_35 = self.max_35_layer([dense_31, dense_51])
    max_37 = self.max_37_layer([dense_31, dense_71])
    max_311 = self.max_311_layer([dense_31, dense_111])
    max_57 = self.max_57_layer([dense_51, dense_71])
    max_511 = self.max_511_layer([dense_51, dense_111])
    max_711 = self.max_711_layer([dense_71, dense_111])

    concatenated_output = concatenate([
        embedded, dense_112, dense_113, dense_115, dense_117, dense_21, dense_31, dense_51, dense_71, dense_111,
        add_23, add_25, add_27, add_211, add_35, add_37, add_311, add_57, add_511, add_711,
        mul_23, mul_25, mul_27, mul_211, mul_35, mul_37, mul_311, mul_57, mul_511, mul_711,
        min_23, min_25, min_27, min_211, min_35, min_37, min_311, min_57, min_511, min_711,
        max_23, max_25, max_27, max_211, max_35, max_37, max_311, max_57, max_511, max_711,
      ],
      name='concatenated_output')

    dense_output = self.dense_output_layer(concatenated_output)

    return dense_output

  def get_config(self):
    cfg = super().get_config()
    return cfg


class CharEmbeddingV03x01(Layer):
  def __init__(self, cfg, trainable=True, **kwargs):
    num_classes = cfg['num_classes']
    max_length = cfg['max_length']
    dim_embeddings = 7

    self.trainable = trainable
    self.output_size = 7
    self.activation = 'tanh'

    self.embedding_layer = Embedding(num_classes, dim_embeddings, input_length=max_length, trainable=self.trainable, name='embedding')
    self.dense_72_layer = Dense(2, name='dense_72', trainable=self.trainable, activation=self.activation)
    self.dense_73_layer = Dense(3, name='dense_73', trainable=self.trainable, activation=self.activation)
    self.dense_75_layer = Dense(5, name='dense_75', trainable=self.trainable, activation=self.activation)

    self.dense_21_layer = Dense(1, name='dense_21', trainable=self.trainable, activation=self.activation)
    self.dense_31_layer = Dense(1, name='dense_31', trainable=self.trainable, activation=self.activation)
    self.dense_51_layer = Dense(1, name='dense_51', trainable=self.trainable, activation=self.activation)
    self.dense_71_layer = Dense(1, name='dense_71', trainable=self.trainable, activation=self.activation)

    self.add_23_layer = Add(name='add_23')
    self.add_25_layer = Add(name='add_25')
    self.add_27_layer = Add(name='add_27')
    self.add_35_layer = Add(name='add_35')
    self.add_37_layer = Add(name='add_37')
    self.add_57_layer = Add(name='add_57')

    self.mul_23_layer = Multiply(name='mul_23')
    self.mul_25_layer = Multiply(name='mul_25')
    self.mul_27_layer = Multiply(name='mul_27')
    self.mul_35_layer = Multiply(name='mul_35')
    self.mul_37_layer = Multiply(name='mul_37')
    self.mul_57_layer = Multiply(name='mul_57')

    self.max_23_layer = Maximum(name='max_23')
    self.max_25_layer = Maximum(name='max_25')
    self.max_27_layer = Maximum(name='max_27')
    self.max_35_layer = Maximum(name='max_35')
    self.max_37_layer = Maximum(name='max_37')
    self.max_57_layer = Maximum(name='max_57')

    self.min_23_layer = Minimum(name='min_23')
    self.min_25_layer = Minimum(name='min_25')
    self.min_27_layer = Minimum(name='min_27')
    self.min_35_layer = Minimum(name='min_35')
    self.min_37_layer = Minimum(name='min_37')
    self.min_57_layer = Minimum(name='min_57')

    self.output_layer = Dense(self.output_size, name='output', trainable=self.trainable, activation=self.activation)

    super(CharEmbeddingV03x01, self).__init__(**kwargs)

  def call(self, inputs):
    embedded = self.embedding_layer(inputs)

    dense_72 = self.dense_72_layer(embedded)
    dense_73 = self.dense_73_layer(embedded)
    dense_75 = self.dense_75_layer(embedded)

    dense_21 = self.dense_21_layer(dense_72)
    dense_31 = self.dense_31_layer(dense_73)
    dense_51 = self.dense_51_layer(dense_75)
    dense_71 = self.dense_71_layer(embedded)

    add_23 = self.add_23_layer([dense_21, dense_31])
    add_25 = self.add_25_layer([dense_21, dense_51])
    add_27 = self.add_27_layer([dense_21, dense_71])
    add_35 = self.add_35_layer([dense_31, dense_51])
    add_37 = self.add_37_layer([dense_31, dense_71])
    add_57 = self.add_57_layer([dense_51, dense_71])

    mul_23 = self.mul_23_layer([dense_21, dense_31])
    mul_25 = self.mul_25_layer([dense_21, dense_51])
    mul_27 = self.mul_27_layer([dense_21, dense_71])
    mul_35 = self.mul_35_layer([dense_31, dense_51])
    mul_37 = self.mul_37_layer([dense_31, dense_71])
    mul_57 = self.mul_57_layer([dense_51, dense_71])

    max_23 = self.max_23_layer([dense_21, dense_31])
    max_25 = self.max_25_layer([dense_21, dense_51])
    max_27 = self.max_27_layer([dense_21, dense_71])
    max_35 = self.max_35_layer([dense_31, dense_51])
    max_37 = self.max_37_layer([dense_31, dense_71])
    max_57 = self.max_57_layer([dense_51, dense_71])

    min_23 = self.min_23_layer([dense_21, dense_31])
    min_25 = self.min_25_layer([dense_21, dense_51])
    min_27 = self.min_27_layer([dense_21, dense_71])
    min_35 = self.min_35_layer([dense_31, dense_51])
    min_37 = self.min_37_layer([dense_31, dense_71])
    min_57 = self.min_57_layer([dense_51, dense_71])

    concatenated_output = concatenate([
        embedded, dense_73, dense_71, dense_51, dense_31, dense_21,
        mul_23, mul_25, mul_27, mul_35, mul_37, mul_57,
        add_23, add_25, add_27, add_35, add_37, add_57,
        max_23, max_25, max_27, max_35, max_37, max_57,
        min_23, min_25, min_27, min_35, min_37, min_57,
      ],
      name='concatenated_output')

    output = self.output_layer(concatenated_output)

    return output

  def get_config(self):
    cfg = super().get_config()
    return cfg


class CharEmbeddingV01x04(Layer):
  def __init__(self, cfg, trainable=True, **kwargs):
    num_classes = cfg['num_classes']
    max_length = cfg['max_length']

    dim_embeddings = 2
    dense_compressed_1_size = 1
    dense_compressed_2_size = 1
    dense_compressed_3_size = 2
    dense_compressed_4_size = 2

    self.embedding_layer = Embedding(num_classes, dim_embeddings, input_length=max_length, trainable=trainable, name='embedding')
    self.dense_compressed_layer_1 = Dense(dense_compressed_1_size, name='dense_compressed_layer_1', trainable=trainable, activation='selu')
    self.dense_compressed_layer_2 = Dense(dense_compressed_2_size, name='dense_compressed_layer_2', trainable=trainable, activation='exponential')
    self.dense_compressed_layer_3 = Dense(dense_compressed_3_size, name='dense_compressed_layer_3', trainable=trainable, activation='selu')
    self.dense_compressed_layer_4 = Dense(dense_compressed_4_size, name='dense_compressed_layer_4', trainable=trainable, activation='linear')

    super(CharEmbeddingV01x04, self).__init__(**kwargs)

  def call(self, inputs):
    embedded = self.embedding_layer(inputs)
    dense_compressed_1 = self.dense_compressed_layer_1(embedded)
    dense_compressed_2 = self.dense_compressed_layer_2(embedded)
    dense_compressed_3 = self.dense_compressed_layer_3(embedded)
    multipled_compressed = Multiply(name='multipled_compressed')([dense_compressed_1, dense_compressed_2])
    concatenated_compressed = concatenate([embedded, dense_compressed_1, dense_compressed_2, dense_compressed_3, multipled_compressed], name='concatenated_compressed')
    dense_compressed_4 = self.dense_compressed_layer_4(concatenated_compressed)


    return dense_compressed_4

  def get_config(self):
    cfg = super().get_config()
    return cfg


class CharEmbeddingV01x02(Layer):
  def __init__(self, cfg, trainable=True, **kwargs):
    num_classes = cfg['num_classes']
    max_length = cfg['max_length']

    dim_embeddings = 4
    dense_compressed_1_size = 1
    dense_compressed_2_size = 1
    dense_compressed_3_size = 3
    dense_compressed_4_size = 2

    self.embedded_layer = Embedding(num_classes, dim_embeddings, input_length=max_length, trainable=trainable, name='embedding')
    self.dense_compressed_layer_1 = Dense(dense_compressed_1_size, name='dense_compressed_layer_1', trainable=trainable, activation='linear')
    self.dense_compressed_layer_2 = Dense(dense_compressed_2_size, name='dense_compressed_layer_2', trainable=trainable, activation='sigmoid')
    self.dense_compressed_layer_3 = Dense(dense_compressed_3_size, name='dense_compressed_layer_3', trainable=trainable, activation='swish')
    self.dense_compressed_layer_4 = Dense(dense_compressed_4_size, name='dense_compressed_layer_4', trainable=trainable, activation='swish')

    super(CharEmbeddingV01x02, self).__init__(**kwargs)

  def call(self, inputs):
    embedded = self.embedded_layer(inputs)
    dense_compressed_1 = self.dense_compressed_layer_1(embedded)
    dense_compressed_2 = self.dense_compressed_layer_2(embedded)
    dense_compressed_3 = self.dense_compressed_layer_3(embedded)
    concatenated_compressed = concatenate([dense_compressed_1, dense_compressed_2, dense_compressed_3], name='concatenated_compressed')
    dense_compressed_4 = self.dense_compressed_layer_4(concatenated_compressed)

    return dense_compressed_4

  def get_config(self):
    cfg = super().get_config()
    return cfg


class CharEmbeddingV01x03(Layer):
  def __init__(self, cfg, trainable=True, **kwargs):
    num_classes = cfg['num_classes']
    max_length = cfg['max_length']

    dim_embeddings = 4
    dense_compressed_1_size = 1
    dense_compressed_2_size = 1
    dense_compressed_3_size = 1

    self.embedding_layer = Embedding(num_classes, dim_embeddings, input_length=max_length, trainable=trainable, name='embedding')
    self.dense_compressed_layer_1 = Dense(dense_compressed_1_size, name='dense_compressed_layer_1', trainable=trainable, activation='selu')
    self.dense_compressed_layer_2 = Dense(dense_compressed_2_size, name='dense_compressed_layer_2', trainable=trainable, activation='exponential')
    self.dense_compressed_layer_3 = Dense(dense_compressed_3_size, name='dense_compressed_layer_3', trainable=trainable, activation='linear')

    super(CharEmbeddingV01x03, self).__init__(**kwargs)

  def call(self, inputs):
    embedded = self.embedding_layer(inputs)
    dense_compressed_1 = self.dense_compressed_layer_1(embedded)
    dense_compressed_2 = self.dense_compressed_layer_2(embedded)
    multipled_compressed = Multiply(name='multipled_compressed')([dense_compressed_1, dense_compressed_2])
    concatenated_compressed = concatenate([embedded, multipled_compressed, dense_compressed_2], name='concatenated_compressed')
    dense_compressed_3 = self.dense_compressed_layer_3(concatenated_compressed)


    return dense_compressed_3

  def get_config(self):
    cfg = super().get_config()
    return cfg


class CharEmbeddingV5(Layer):
  def __init__(self, cfg, trainable=True, **kwargs):
    num_classes = cfg['num_classes']
    max_length = cfg['max_length']

    dim_embeddings = 11
    dense_compressed_1_size = 5

    self.embedding_layer = Embedding(num_classes, dim_embeddings, input_length=max_length, trainable=trainable, name='embedding')
    self.dense_compressed_layer_1 = Dense(dense_compressed_1_size, name='dense_compressed_layer_1', trainable=trainable, activation='tanh')

    super(CharEmbeddingV5, self).__init__(**kwargs)

  def call(self, inputs):
    embedded = self.embedding_layer(inputs)
    dense_compressed_1 = self.dense_compressed_layer_1(embedded)

    return dense_compressed_1

  def get_config(self):
    cfg = super().get_config()
    return cfg

