import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adagrad, Nadam
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM, Bidirectional, Dropout, LayerNormalization, Layer, BatchNormalization
from tensorflow.keras.layers import concatenate, Reshape, SpatialDropout1D, Conv1D, Flatten, AveragePooling1D, MaxPool1D, Average, Maximum, Multiply, Add, Minimum
from tensorflow.keras.models import Model, Sequential

from article_body_recognizer.ANNs.charemb_networks import CharEmbeddingV5


def HierarchyV3x24x00(cfg, learning_rate=None):
  num_categories = cfg['num_categories']
  num_classes = cfg['num_classes']
  max_length = cfg['max_length']
  lr = learning_rate if learning_rate else cfg['learning_rate']
  emb_trainable = cfg['emb_trainable']
  decoder_trainable = cfg['decoder_trainable']

  fine_tuning = cfg.get('dropout_fine_tuning', 0)

  Optimizer = cfg['optimizer']
  print('[INFO] optimizer: ', Optimizer)

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

  char_embedding_layer = CharEmbeddingV5(num_classes=num_classes, max_length=max_length, trainable=emb_trainable, name='char_embedding')

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

  optimizer = Optimizer(learning_rate=lr)

  model.compile(
      loss='categorical_crossentropy',
      optimizer=optimizer,
      metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy')],
    )

  return model
