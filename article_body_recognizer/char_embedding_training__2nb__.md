---
jupyter:
  jupytext:
    cell_metadata_filter: -all
    formats: md,ipynb
    main_language: python
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
---

```python
import os
os.chdir('/workspace/upwork/martien_brouver/mylapi/scraping/')
```
```python
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

from tensorflow import config as config
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import TensorBoard, Callback

from article_body_recognizer.ANNs.charemb_comparator import CharembComparatorV1
from article_body_recognizer.ANNs.charemb_network import CharEmbeddingV5

```
```python
cfg = {
  'tictoc': True,
  'emb_trainable': True,
  'pretrained_emb_vers': None, # str or None
  'new_emb_vers': 'v5x10u03',
  'pretrained_trainer_vers': 'trainer_v5x10u03_re04_tictoc', # str or None
  'new_trainer_version' : 'trainer_v5x10u03_re04_tictoc',
  'lean_dataset': True,
  'pretrained_model_vers': None,  # if set this will get higher priority than pretrained_trainer_vers
  'embedding_model_class': CharEmbeddingV5,
  'comparison_norm_trainable': False,
  'max_length': char_emb_training_specs['MAX_LENGTH'],
  'min_length': char_emb_training_specs['MIN_LENGTH'],
  'num_classes': char_emb_training_specs['NUM_CLASSES'],
  # 'close_masking_ratio': 0.15,    # Many words are suitable to describe this idea:
  # 'neutral_masking_ratio': 0.35,  # It's fundamentally flaw, naive, bad because it reveal obviously to model to easily predict which pair is simimlar and which one disimilar
  'masking_ratio': 0.15,
  'close_distance_scale': 1.0,
  'neutral_distance_scale': -0.55,
  'learning_rate': 1e-4,
  'optimizer': RMSprop,
  'batch_size': 4096,
  'epochs': 21,
  'buffer_size': 32,
  'pribuf_looping': True,  # If is True then buffer_size makes no affect and is set to steps_per_epoch
}

print(cfg)

SLEEP_TIME = 0.05
```
```python
def design_model(cfg):

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

  char_embedding_layer = CharEmbeddingV5(cfg, trainable=emb_trainable, name='char_embedding')
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

def create_model(cfg):
  model = CharembComparatorV1(cfg)
  model.summary()

  emb_trainable = cfg['emb_trainable']
  new_emb_vers = cfg['new_emb_vers']
  pretrained_emb_vers = cfg['pretrained_emb_vers']
  pretrained_trainer_vers = cfg['pretrained_trainer_vers']
  new_trainer_version = cfg['new_trainer_version']

  assert new_trainer_version, "new_trainer_version must be set."

  if pretrained_trainer_vers == new_trainer_version:
    assert pretrained_emb_vers is None, "Retrain the current trainer, set pretrained_emb_vers to None to mitigate potential of mistake."

  if emb_trainable:
    matched = re.search(r'(v\d+x\d\du\d\d)', new_trainer_version)
    detected_embedding_vers = matched.group(1)
    assert new_emb_vers == detected_embedding_vers, f"new_emb_vers: {new_emb_vers} does not match with new_trainer_version: {new_trainer_version} -> detected_embedding_vers"

  if not emb_trainable and pretrained_emb_vers:
    matched = re.search(r'(v\d+x\d\du\d\d)', new_trainer_version)
    detected_embedding_vers = matched.group(1)
    assert pretrained_emb_vers == detected_embedding_vers, f"pretrained_emb_vers: {pretrained_emb_vers} does not match with new_trainer_version: {new_trainer_version} -> {detected_embedding_vers}"

  if pretrained_emb_vers:
    print(f'[INFO] Load embedding layer weights from {pretrained_emb_vers}')
    objectRep = open(f"parser/pretrained_embedding/{pretrained_emb_vers}.pickle", "rb")
    char_embedding_layer_weights = pickle.load(objectRep)
    char_embedding_layer = model.get_layer('char_embedding')
    char_embedding_layer.set_weights(char_embedding_layer_weights)
    objectRep.close()
  else:
    print(f'[INFO] pretrained_emb_vers is not set, embedding layer weights will be initialized.')

  if isinstance(pretrained_trainer_vers, str):
    print(f'[INFO] Load trainer weights from {pretrained_trainer_vers}')
    model.load_weights(f"parser/pretrained_embedding/trainers/{pretrained_trainer_vers}.h5")
  else:
    print(f'[INFO] pretrained_trainer_vers is not set, trainer weights will be initialized.')

  return model
```
```python
def is_jsonline_file(fname):
  is_jlext_p = re.compile('\.jl$')

  return is_jlext_p.search(fname) is not None

def load_dataset(source_path, split=None, loop=True):
  path_obj = Path(source_path)
  dataset = []

  for file in path_obj.iterdir():

    with open(file.absolute()) as reader:
      lines = reader.readlines()
      for line in lines:
        dataset.append(line)

  return dataset

emb_trainable = cfg['emb_trainable']
validating_dataset_file = 'parser/charembdataset/tmp/validating_dataset.pickle'
training1_dataset_file = 'parser/charembdataset/tmp/training1_dataset.pickle'
training2_dataset_file = 'parser/charembdataset/tmp/training2_dataset.pickle'

# validating_dataset = load_dataset('parser/charembdataset/validating', cfg)
# with open(validating_dataset_file, 'wb') as f:
#     pickle.dump(validating_dataset, f, pickle.HIGHEST_PROTOCOL)

# training1_dataset = load_dataset('parser/charembdataset/training1', cfg)
# with open(training1_dataset_file, 'wb') as f:
#     pickle.dump(training1_dataset, f, pickle.HIGHEST_PROTOCOL)

# training2_dataset = load_dataset('parser/charembdataset/training2', cfg)
# with open(training2_dataset_file, 'wb') as f:
#     pickle.dump(training2_dataset, f, pickle.HIGHEST_PROTOCOL)

with open(validating_dataset_file, 'rb') as f:
  validating_dataset = pickle.load(f)

with open(training1_dataset_file, 'rb') as f:
  training1_dataset = pickle.load(f)

with open(training2_dataset_file, 'rb') as f:
  training2_dataset = pickle.load(f)

if cfg['emb_trainable']:
  training_dataset = training1_dataset
else:
  training_dataset = training2_dataset
```
```python
from scraping.char_dict import vocabularies as vocab
print('Vocabuary index: ', vocab)


def mask_squence(sequence, masking_ratio, fill_value):
  lgth = len(sequence)
  masking_lgth = math.floor(lgth * masking_ratio)
  mask = np.full(lgth, 0)
  mask[:masking_lgth] = 1
  np.random.shuffle(mask)

  masked = ma.masked_array(sequence, mask=mask, fill_value=fill_value)

  return masked.filled()


def charrnn_encode_sequence(text, vocab, maxlen, masking_ratio=None):
    '''
    Encodes a text into the corresponding encoding for prediction with
    the model.
    '''

    oov = vocab['oov']
    masking_code = 0
    encoded = np.array([vocab.get(x, oov) for x in text])
    if masking_ratio:
      encoded = mask_squence(encoded, masking_ratio, masking_code)
    return sequence.pad_sequences([encoded], padding='post', maxlen=maxlen)
```
```python
def is_jsonline_file(fname):
  is_jlext_p = re.compile('\.jl$')

  return is_jlext_p.search(fname) is not None


def count_dataset_size(data_source, cfg):
  max_length = cfg['max_length']
  min_length = cfg['min_length']

  dataset = load_dataset(data_source, loop=False)

  counter = 0
  for s in dataset:
    lgth = len(s)
    if min_length > lgth or lgth > max_length:
      continue

    counter += 1

  return counter


def load_dataset(source_path, split=None, loop=True):
  path_obj = Path(source_path)

  while True:
    for file in path_obj.iterdir():

      with open(file.absolute()) as reader:
        lines = reader.readlines()
        for line in lines:
          yield line

    if not loop:
      return None


class ThreadBreaker:
  _terminated = False
  def might_to_exit(self):
    if self._terminated:
      print('\n[THREADING] exited!\n')
      _thread.exit()
    else:
      return True
  def exit(self):
    self._terminated = True

def buffer_data(cfg, dataset, pri_buffer, split_type, breaker, model):
  assert isinstance(breaker, ThreadBreaker), "breaker must be an instance of ThreadBreaker"
  batch_size = cfg['batch_size']
  max_length = cfg['max_length']
  min_length = cfg['min_length']
  masking_ratio = cfg['masking_ratio']
  neutral_distance_scale = cfg['neutral_distance_scale']
  close_distance_scale = cfg['close_distance_scale']
  lean_dataset = cfg['lean_dataset']
  pribuf_looping = cfg['pribuf_looping']

  if pribuf_looping:
    BUFFER_SIZE = pri_buffer.maxlen - 1
  else:
    BUFFER_SIZE = pri_buffer.maxlen / 2

  if lean_dataset:
   assert model is not None, "model must be pre-trained in lean_dataset mode."

  trial_times = 100
  max_prev_text_quantity = 1000
  prev_texts_queue = deque(['<p>This implementation of RMSprop uses plain momentum, not Nesterov momentum.</p>']*max_prev_text_quantity, maxlen=max_prev_text_quantity)

  def is_training_split():
    assert split_type in ['training', 'validating'], "Invalid split name."
    return 'training' == split_type

  def measure_distance(inputs, labels):
    # print('[DEBUG] Generating truth values ...')
    preds = model.predict_on_batch(inputs)
    distance_1 = np.array(preds[0]).squeeze()
    distance_2 = np.array(preds[1]).squeeze()
    distance_3 = np.array(preds[2]).squeeze()

    mask = np.array(labels) == close_distance_scale
    distance_1[mask] = close_distance_scale
    distance_2[mask] = close_distance_scale
    distance_3[mask] = close_distance_scale

    _labels = {
      'distance_1': distance_1,
      'distance_2': distance_2,
      'distance_3': distance_3,
    }

    # print('[DEBUG] Done.')

    return _labels

  def will_happen_with_respect_to(probability):
    to_happen = random.random() <= probability
    return to_happen

  def get_prev_text(_text):
    _prev = random.choice(prev_texts_queue)

    _counter = 1
    while _prev == _text and _counter <= trial_times:
      _prev = random.choice(prev_texts_queue)
      _counter += 1

    return _prev

  # def _queue_or_count(_X1, _X2, _Y1, _Y2, _count_batch):
  def _queue_or_count(_X1, _X2, _Y1, _count_batch):

    # Firstly, count batch's volume
    _count_batch += 1

    if _count_batch % BATCH_SIZE == 0:
      _X1 = np.array(_X1)
      _X2 = np.array(_X2)
      _X = {
        'input_1': _X1,
        'input_2': _X2,
      }

      if lean_dataset and is_training_split():
        _Y = measure_distance(_X, _Y1)
      else:
        _Y1 = np.array(_Y1)
        _Y = {
          'distance_1': _Y1,
          'distance_2': _Y1,
          'distance_3': _Y1,
        }

      batch = (_X, _Y)

      pri_buffer.appendleft(batch)

      # if the primary buffer is full then wait for the number of remaining items less than BUFFER_SIZE
      if len(pri_buffer) >= pri_buffer.maxlen:
        while len(pri_buffer) > BUFFER_SIZE:
          # print('Waiting for data unloaded...')
          breaker.might_to_exit()
          time.sleep(SLEEP_TIME)

      # Reset batch
      _X1 = []
      _X2 = []
      _Y1 = []
      # _Y2 = []
      _count_batch = 0

      return (_X1, _X2, _Y1, _count_batch)

    # not thing more than accumulate batch count
    return (_X1, _X2, _Y1, _count_batch)


  while True:
    if len(pri_buffer) < pri_buffer.maxlen:
      X1_batch = []
      X2_batch = []
      Y1_batch = []
      # Y2_batch = []
      count_batch = 0

      for s in dataset:
        lgth = len(s)
        if min_length > lgth or lgth > max_length:
          continue

        encoded_1 = charrnn_encode_sequence(s, vocab, max_length, masking_ratio=masking_ratio)[0]
        encoded_2 = charrnn_encode_sequence(s, vocab, max_length, masking_ratio=masking_ratio)[0]

        X1_batch.append(encoded_1)
        X2_batch.append(encoded_2)
        Y1_batch.append(close_distance_scale)
        X1_batch, X2_batch, Y1_batch, count_batch = _queue_or_count(X1_batch, X2_batch, Y1_batch, count_batch)

        # Reverse the oder to force distance symmetric
        X1_batch.append(encoded_2)
        X2_batch.append(encoded_1)
        Y1_batch.append(close_distance_scale)
        X1_batch, X2_batch, Y1_batch, count_batch = _queue_or_count(X1_batch, X2_batch, Y1_batch, count_batch)

        # if not lean_dataset or is_training_split():
        if not lean_dataset:
          prev = get_prev_text(s)

          encoded_1 = charrnn_encode_sequence(s, vocab, max_length, masking_ratio=masking_ratio)[0]
          encoded_2 = charrnn_encode_sequence(prev, vocab, max_length, masking_ratio=masking_ratio)[0]

          X1_batch.append(encoded_1)
          X2_batch.append(encoded_2)
          Y1_batch.append(neutral_distance_scale)
          X1_batch, X2_batch, Y1_batch, count_batch = _queue_or_count(X1_batch, X2_batch, Y1_batch, count_batch)

          # Because of the approximation of all far distance is at low accuracy the masking is applied back and forth loosely (differently)
          # Reverse the oder to force distance symmetric
          encoded_2 = charrnn_encode_sequence(s, vocab, max_length, masking_ratio=masking_ratio)[0]
          encoded_1 = charrnn_encode_sequence(prev, vocab, max_length, masking_ratio=masking_ratio)[0]

          X1_batch.append(encoded_1)
          X2_batch.append(encoded_2)
          Y1_batch.append(neutral_distance_scale)
          X1_batch, X2_batch, Y1_batch, count_batch = _queue_or_count(X1_batch, X2_batch, Y1_batch, count_batch)

        else:

          # Invalidating prediction just count for the real truth values
          # So just include the neutral_distance_scale cases for training
          if is_training_split():
            prev = get_prev_text(s)

            # Try to make two comparators balanced in lean_dataset regime
            if  will_happen_with_respect_to(probability=0.5):
              encoded_1 = charrnn_encode_sequence(s, vocab, max_length, masking_ratio=masking_ratio)[0]
              encoded_2 = charrnn_encode_sequence(prev, vocab, max_length, masking_ratio=masking_ratio)[0]
            else:
              encoded_2 = charrnn_encode_sequence(s, vocab, max_length, masking_ratio=masking_ratio)[0]
              encoded_1 = charrnn_encode_sequence(prev, vocab, max_length, masking_ratio=masking_ratio)[0]

            X1_batch.append(encoded_1)
            X2_batch.append(encoded_2)
            Y1_batch.append(neutral_distance_scale)
            X1_batch, X2_batch, Y1_batch, count_batch = _queue_or_count(X1_batch, X2_batch, Y1_batch, count_batch)

        # ATTENTION: Big trouble for you if you put this one outside of the loop
        # REMEMBER: If you put this one outside of the loop everything is perfect except the performance, THAT'S IT!!!
        if will_happen_with_respect_to(probability=0.5):
          # print('append to prev_texts_queue, length: ', len(prev_texts_queue))
          prev_texts_queue.append(s)


def create_generator(buffer_size, pribuf_looping=False, steps_per_epoch=None):
  if pribuf_looping:
    assert steps_per_epoch is not None, "steps_per_epoch is required when pribuf_looping is set to True"
    pribuf_size = steps_per_epoch + 1
  else:
    pribuf_size = buffer_size * 2

  subbuf_size = buffer_size
  print('[INFO] Primary deque maxlen: ', pribuf_size)
  print('[INFO] Secondary deque maxlen: ', subbuf_size)

  sub_buffer = deque(maxlen=subbuf_size)
  pri_buffer = deque(maxlen=pribuf_size)

  def is_primary_buffer_ready():
    if pribuf_looping:
      return len(pri_buffer) >= pribuf_size
    else:
      return len(pri_buffer) >= subbuf_size

  def is_secondary_buffer_ready():
    return len(sub_buffer) > 0

  def generator():
    while True:
      if is_secondary_buffer_ready():
        yield sub_buffer.pop()

      elif is_primary_buffer_ready():
        for i in range(subbuf_size):
          batch = pri_buffer.pop()
          sub_buffer.appendleft(batch)
          if pribuf_looping:
            pri_buffer.appendleft(batch)
            # print('loop the curren batch, pri_buffer len: ', len(pri_buffer))

      else:
        # print('Waiting for data...')
        time.sleep(SLEEP_TIME)

  return pri_buffer, generator()

def start_buffer_data_thread(_cfg, _dataset, _pri_buffer, _split_type, _model):
  breaker = ThreadBreaker()
  _thread.start_new_thread(buffer_data, (), {'cfg': cfg, 'dataset': _dataset, 'pri_buffer': _pri_buffer, 'split_type': _split_type, 'breaker': breaker, 'model': _model})
  return breaker

def calculate_training_steps(_cfg, _dataset):
  sample_quantity = len(_dataset) * 3 if _cfg['lean_dataset'] else len(_dataset) * 4
  steps_per_epoch = math.ceil(sample_quantity/_cfg['batch_size'])

  return steps_per_epoch, sample_quantity

def calculate_validating_steps(_cfg, _dataset):
  sample_quantity = len(_dataset) * 2 if _cfg['lean_dataset'] else len(_dataset) * 4
  steps_per_epoch = math.ceil(sample_quantity/_cfg['batch_size'])

  return steps_per_epoch, sample_quantity

BATCH_SIZE = cfg['batch_size']
BUFFER_SIZE = cfg['buffer_size']
_pribuf_looping = cfg['pribuf_looping']
_lean_dataset = cfg['lean_dataset']

steps_per_epoch, training_sample_quantity = calculate_training_steps(_cfg=cfg, _dataset=training_dataset)
print('[INFO] training_sample_quantity', training_sample_quantity)
print('[INFO] training steps_per_epoch: ', steps_per_epoch)

validation_steps, validating_sample_quantity = calculate_validating_steps(_cfg=cfg, _dataset=validating_dataset)
print('[INFO] validating_sample_quantity', validating_sample_quantity)
print('[INFO] validation_steps: ', validation_steps)

```
```python
max_length = cfg['max_length']

class Tester:
  def __init__(self, model, dataset):

    self.size = 501
    self.model = model
    self.dataset = dataset
    self.arguments = deque(maxlen=self.size)
    self.result_cache = {
      'preds1': None,
      'preds2': None,
    }

    for i in range(self.size):
      self.arguments.appendleft(random.choice(self.dataset))

  def will_happen_with_respect_to(self, probability):
    to_happen = random.random() <= probability
    return to_happen

  def prepare_data(self):
    _data1 = {
      'input_1': [],
      'input_2': [],
    }

    _data2 = {
      'input_1': [],
      'input_2': [],
    }

    _argument = '<p>This implementation of RMSprop uses plain momentum, not Nesterov momentum.</p>'
    for item in self.dataset:

      _data1['input_1'].append(charrnn_encode_sequence(item, vocab, max_length)[0])
      _data1['input_2'].append(charrnn_encode_sequence(_argument, vocab, max_length)[0])

      # Reverse the order of item and argument
      _data2['input_1'].append(charrnn_encode_sequence(_argument, vocab, max_length)[0])
      _data2['input_2'].append(charrnn_encode_sequence(item, vocab, max_length)[0])

      # in case of argument pool is not full yet, always use previous one as argument and stack item into pool
      if self.size != len(self.arguments):
        _argument = item
        self.arguments.appendleft(item)

      # in case of argument pool is ready, pick from and stack item into pool randomly
      elif self.will_happen_with_respect_to(probability=0.5):
        _argument = random.choice(self.arguments)
        self.arguments.appendleft(item)

    _data1['input_1'] = np.array(_data1['input_1'])
    _data1['input_2'] = np.array(_data1['input_2'])

    _data2['input_1'] = np.array(_data2['input_1'])
    _data2['input_2'] = np.array(_data2['input_2'])

    return _data1, _data2

  def get_prediction(self):

    if self.result_cache.get('preds1') is None or self.result_cache.get('preds2') is None:
      data1, data2 = self.prepare_data()
      preds1 = self.model.predict_on_batch(x=data1)
      preds2 = self.model.predict_on_batch(x=data2)
      # print(preds1[0].tolist())
      # print(preds2[0].tolist())

      # cache prediction
      self.result_cache['preds1'] = preds1
      self.result_cache['preds2'] = preds2

    return self.result_cache['preds1'], self.result_cache['preds2']

  def test_symmetric_distance(self):
    preds1, preds2 = self.get_prediction()

    losses = []
    for pred1, pred2 in zip(preds1, preds2):
      loss = np.absolute(np.subtract(pred1, pred2)).mean()
      losses.append(loss)

    return losses

  def test_distance_mean(self):
    """
    Result of this test is the distances against 1 (ideally almost pairs of inputs are different)
    and a big number is a signal of both potentially good embedding structure and pattern detectors (comparators)
    """
    preds1, preds2 = self.get_prediction()

    pred1_distance_means = []
    pred2_distance_means = []
    for pred1, pred2 in zip(preds1, preds2):
      distance1 = np.subtract(1, pred1).mean()
      pred1_distance_means.append(distance1)
      distance2 = np.subtract(1, pred2).mean()
      pred2_distance_means.append(distance2)

    return pred1_distance_means, pred2_distance_means

```
```python
def do_training(_cfg, _steps_per_epoch, _validation_steps, _debug_generator=False):
  EPOCHS = _cfg['epochs']
  BATCH_SIZE = _cfg['batch_size']
  BUFFER_SIZE = _cfg['buffer_size']
  pribuf_looping = cfg['pribuf_looping']

  pretrained_model = None
  if _lean_dataset:
    pretrained_trainer_vers = cfg['pretrained_trainer_vers']
    pretrained_model_vers = cfg['pretrained_model_vers']

    version = pretrained_model_vers if pretrained_model_vers else pretrained_trainer_vers
    print(f'[INFO] Model {version} will be used to generate truth labels of dataset.')

    pretrained_model = design_model(cfg)
    pretrained_model.load_weights(f"parser/pretrained_embedding/trainers/{version}.h5")
    pretrained_model.trainable = False
  else:
    print(f'[INFO] Training model by data with hypothetic truth labels.')


  # Training generator
  split_type = 'training'
  training_queue, training_generator = create_generator(buffer_size=BUFFER_SIZE, pribuf_looping=pribuf_looping, steps_per_epoch=_steps_per_epoch)
  training_queue_breaker = start_buffer_data_thread(_cfg=_cfg, _dataset=training_dataset, _pri_buffer=training_queue, _split_type=split_type, _model=pretrained_model)

  # Validating generator
  split_type = 'validating'
  validating_queue, validating_generator = create_generator(buffer_size=BUFFER_SIZE, pribuf_looping=pribuf_looping, steps_per_epoch=_validation_steps)
  validating_queue_breaker = start_buffer_data_thread(_cfg=_cfg, _dataset=validating_dataset, _pri_buffer=validating_queue, _split_type=split_type, _model=pretrained_model)

  if _debug_generator:
    return training_generator, validating_generator

  model = create_model(_cfg)
  model.fit(
      training_generator,
      batch_size=BATCH_SIZE,
      steps_per_epoch=_steps_per_epoch,
      epochs=EPOCHS,
      validation_data=validating_generator,
      validation_batch_size=BATCH_SIZE,
      validation_steps=_validation_steps,
      shuffle='batch',
    )

  training_queue_breaker.exit()
  validating_queue_breaker.exit()

  return model

def do_tictoc_training(_cfg):
  EPOCHS = _cfg['epochs']
  BATCH_SIZE = _cfg['batch_size']
  BUFFER_SIZE = _cfg['buffer_size']
  pribuf_looping = cfg['pribuf_looping']

  pretrained_trainer_vers = cfg['pretrained_trainer_vers']
  pretrained_model_vers = cfg['pretrained_model_vers']
  version = pretrained_model_vers if pretrained_model_vers else pretrained_trainer_vers
  print(f'[INFO] Model {version} will be used to generate truth labels of dataset.')

  pretrained_model = design_model(cfg)
  pretrained_model.load_weights(f"parser/pretrained_embedding/trainers/{version}.h5")
  pretrained_model.trainable = False

  _validating_dataset = validating_dataset

  tmp_weight_filepath = "parser/tmp/model_weights.h5"
  pretrained_model.save_weights(tmp_weight_filepath, overwrite=True)

  for i in range(EPOCHS):
    print('[TRAINING] Grand Epoch: ', i)

    if i % 2 == 0:
      _training_dataset = training1_dataset
      _cfg['emb_trainable'] = True
      _epochs = 1
    else:
      _training_dataset = training2_dataset
      _cfg['emb_trainable'] = False
      _epochs = 3

    _steps_per_epoch, training_sample_quantity = calculate_training_steps(_cfg=_cfg, _dataset=_training_dataset)
    print('[INFO] training_sample_quantity', training_sample_quantity)
    print('[INFO] training steps_per_epoch: ', _steps_per_epoch)

    _validation_steps, validating_sample_quantity = calculate_validating_steps(_cfg=_cfg, _dataset=_validating_dataset)
    validation_steps, validating_sample_quantity = calculate_validating_steps(_cfg=cfg, _dataset=validating_dataset)
    print('[INFO] validating_sample_quantity', validating_sample_quantity)
    print('[INFO] validation_steps: ', _validation_steps)

    split_type = 'training'
    training_queue, training_generator = create_generator(buffer_size=BUFFER_SIZE, pribuf_looping=pribuf_looping, steps_per_epoch=_steps_per_epoch)
    training_queue_breaker = start_buffer_data_thread(_cfg=_cfg, _dataset=_training_dataset, _pri_buffer=training_queue, _split_type=split_type, _model=pretrained_model)

    split_type = 'validating'
    validating_queue, validating_generator = create_generator(buffer_size=BUFFER_SIZE, pribuf_looping=pribuf_looping, steps_per_epoch=_validation_steps)
    validating_queue_breaker = start_buffer_data_thread(_cfg=_cfg, _dataset=_validating_dataset, _pri_buffer=validating_queue, _split_type=split_type, _model=pretrained_model)

    model = design_model(_cfg)
    model.load_weights(tmp_weight_filepath)

    model.fit(
        training_generator,
        batch_size=BATCH_SIZE,
        steps_per_epoch=_steps_per_epoch,
        epochs=_epochs,
        validation_data=validating_generator,
        validation_batch_size=BATCH_SIZE,
        validation_steps=_validation_steps,
        shuffle='batch',
      )

    del pretrained_model
    pretrained_model = model
    pretrained_model.trainable = False
    pretrained_model.save_weights(tmp_weight_filepath, overwrite=True)

    training_queue_breaker.exit()
    validating_queue_breaker.exit()

    tester = Tester(dataset=_validating_dataset, model=pretrained_model)
    losses = tester.test_symmetric_distance()
    print('Divergence of 2 distances of pair of samples: ', losses)
    pred1_distance_means, pred2_distance_means = tester.test_distance_mean()
    print('(pred1: input_1 vs input_2) Distance of 2 different samples: ', pred1_distance_means)
    print('(pred2: input_2 vs input_1) Distance of 2 different samples: ', pred2_distance_means)

  return pretrained_model

```
```python
if not cfg['tictoc']:
  char_model = do_training(
      _cfg=cfg,
      _steps_per_epoch=steps_per_epoch,
      _validation_steps=validation_steps,
    )
else:
  char_model = do_tictoc_training(
      _cfg=cfg,
    )

```
```python
trainable = cfg['emb_trainable']
new_emb_vers = cfg['new_emb_vers']
if trainable and new_emb_vers:
  char_embedding_layer_weights = char_model.get_layer('char_embedding').get_weights()
  with open(f'parser/pretrained_embedding/{new_emb_vers}.pickle', 'wb') as f:
      pickle.dump(char_embedding_layer_weights, f, pickle.HIGHEST_PROTOCOL)
```
```python
new_trainer_version = cfg['new_trainer_version']
char_model.save_weights(f"parser/pretrained_embedding/trainers/{new_trainer_version}.h5", overwrite=True)
```
```python
def inspect_data(batch1, index):
  texts = batch1[0]
  labels = batch1[1]
  texts_1 = texts['input_1'][index]
  print('text 1 input shape: ', texts['input_1'].shape)
  print('text 1 input: ', texts_1.tolist())
  texts_2 = texts['input_2'][index]
  print('text 2 input shape: ', texts['input_2'].shape)
  print('text 2 input: ', texts_2.tolist())
  label_1 = labels['distance_1'][index]
  print('label 1 shape: ', labels['distance_1'].shape)
  print('label 1: ', label_1)

# training_generator, validating_generator = do_training( _cfg=cfg, _steps_per_epoch=steps_per_epoch, _validation_steps=validation_steps, _debug_generator=True)

# Test looping
# for b in validating_generator:
  # time.sleep(1)
  # pass

```
```python
# batch1 = next(training_generator)
# index = 0
```
```python
# raise Exception('WIP.')
# inspect_data(batch1, index)
# index += 1
```
```python
max_length = cfg['max_length']
raw_data = [
    {
    'input_1': '<p>The exhibition will follow several high-profile fashion exhibitions for the VA, including <a>Balenciaga: \
                Shaping Fashion</a>, <a>Mary Quant</a> and the record-breaking <a>Christian Dior: Designer of Dreams</a>.</p>',
    'input_2': '<p>The exhibition will follow several high-profile fashion exhibitions for the VA, including <a>Balenciaga: \
                Shaping Fashion</a>, <a>Mary Quant</a> and the record-breaking <a>Christian Dior: Designer of Dreams</a>.</p>',
    },
    {
      'input_1': '<p>Geometric Deep Learning is an attempt for geometric unification of a broad class of ML problems from the \
                perspectives of symmetry and invariance. </p>',
      'input_2': '<p>Geometric Deep Learning is an attempt for geometric unification of a broad class of ML problems from the \
                perspectives of symmetry and invariance. </p>',
    },
    {
    'input_1': '<p>Geometric Deep Learning is an attempt for geometric unification of a broad class of ML problems from the perspectives \
                of symmetry and invariance. </p>',
    'input_2': '<p>The exhibition will follow several high-profile fashion exhibitions for the VA, including <a>Balenciaga: Shaping Fashion</a>, \
                <a>Mary Quant</a> and the record-breaking <a>Christian Dior: Designer of Dreams</a>.</p>',
    },
    {
    'input_1': '<p>The exhibition will follow several high-profile fashion exhibitions for the VA, including <a>Balenciaga: Shaping Fashion</a>, \
                <a>Mary Quant</a> and the record-breaking <a>Christian Dior: Designer of Dreams</a>.</p>',
    'input_2': '<p>Geometric Deep Learning is an attempt for geometric unification of a broad class of ML problems from the perspectives of symmetry and invariance. </p>',
    },
    {
      'input_1': '<div>Advertisement</div>',
      'input_2': '<div>Advertisement</div>',
    },
    {
      'input_1': '<div>Advertisement</div>',
      'input_2': '<p>Geometric Deep Learning is an attempt for geometric unification of a broad class of ML problems from the perspectives of symmetry and invariance. <p>',
    },
    {
      'input_1': '<p>Geometric Deep Learning is an attempt for geometric unification of a broad class of ML problems from the perspectives of symmetry \
                and invariance. <p>',
      'input_2': '<div>Advertisement</div>',
    },
]

def transform_data(raw):
  _data = {
    'input_1': [],
    'input_2': [],
  }

  for row in raw:
    _data['input_1'].append(charrnn_encode_sequence(row['input_1'], vocab, max_length)[0])
    _data['input_2'].append(charrnn_encode_sequence(row['input_2'], vocab, max_length)[0])

  _data['input_1'] = np.array(_data['input_1'])
  _data['input_2'] = np.array(_data['input_2'])

  return _data

samples = transform_data(raw_data)
preds = char_model.predict_on_batch(x=samples)

ind = 0
output_1 = preds[0]
output_2 = preds[1]
output_3 = preds[2]
for row in output_1:
    print(output_1[ind],"\t", output_2[ind],"\t", output_3[ind])
    ind += 1
```
```python
tester = Tester(dataset=validating_dataset, model=char_model)

losses = tester.test_symmetric_distance()
print('Divergence of 2 distances of pair of samples: ', losses)

pred1_distance_means, pred2_distance_means = tester.test_distance_mean()
print('(pred1: input_1 vs input_2) Distance of 2 different samples: ', pred1_distance_means)
print('(pred2: input_2 vs input_1) Distance of 2 different samples: ', pred2_distance_means)
```
```md
### Training result

- v01x02u02: (continue of v01x02u01) learning_rate = 1e-3; Nadam; loss = 0.0704
- v01x02u03: (continue of v01x02u01) learning_rate = 5e-3; RMSprop; loss = 0.0035
- ([CAVEAT!] The perforance seems very bad. This is the obvious example of overfitting, despite the loss is tiny, the model is useless on predicting unseen data)
- v01x02u04: (retrain v01x02u00) epoch = 79; learning_rate = 5e-3; RMSprop; loss = 0.0036; Bad performance on test data
- v01x02u05: (retrain v01x02u00) epoch = 29; learning_rate = 5e-3; RMSprop; loss = 0.0082; Bad performance on test data
- v01x02u06: (retrain v01x02u00) epoch = 7; learning_rate = 5e-3; RMSprop; loss = 0.0812; Performance: [0.00486887] [0.9875436] [0.01160717] [1.0660381] [0.01404977]

- v01x03u00: (Shink network size, merge conv layers); epoch = 131; learning_rate = 1e-3; RMSprop; batch_size = 1536; val_loss: 0.5923 - val_tf_op_layer_distance_1_loss: 0.2487 - val_tf_op_layer_distance_2_loss: 0.3240

> The best one because it's a result of fixing many mistakes and misundertanding as well as many improvements
- Fix bad dataset
- Use BatchNormalization instead of LayerNormalization (inappropriate)
- Balance information flow between 2 encoders
- Preserve sequnce length (in order to balance encoders) by using only stripe of 1 in conv block, using return_sequences in rnn block
- In dataset generator, reverse the oder to force distance being symmetric
- Using more thang 1 projections to compute distances (dense of size 3 and dense of size 7)
- Implement validating dataset

- v01x04u00: epoch = 101; parameters = 111,369; learning_rate = 1e-3; RMSprop; loss: 0.1180 - val_loss: 0.0994.
> Add third projection for distance measurement. Using Maximum layer so that there is one output.
> Embedding layer has input of size 2 and output of size 2

- trainer_v01x04u00_re01_freezeemb: epoch = 49; learning_rate = 2e-4, RMSprop; loss ~ 0.09 - val_loss ~ 0.07

- v01x04u01 (<- trainer_v01x04u00_re01_freezeemb): epoch = 101; learning_rate = 2e-4 | 1e-4 | 1e-3; No improvement

```
```md
- trainer_v41117x06u00_re00_full -> v41117x06u00 (Use lean dataset): epochs = 7; learning_rate = 1e-3;
> loss: 0.4049 - distance_1_loss: 0.1128 - distance_2_loss: 0.1335 - distance_3_loss: 0.1237 - val_loss: 0.4071 - val_distance_1_loss: 0.1106 - val_distance_2_loss: 0.1383 - val_distance_3_loss: 0.1231
> epochs = 11; learning_rate = 1e-4; Nadam
> loss: 0.3475 - distance_1_loss: 0.0963 - distance_2_loss: 0.1085 - distance_3_loss: 0.1151 - val_loss: 0.3532 - val_distance_1_loss: 0.0959 - val_distance_2_loss: 0.1132 - val_distance_3_loss: 0.1165

- trainer_v41117x06u00_re01_freezeemb (<- trainer_v41117x06u00_re00_full): epochs = 51; learning_rate = 1e-4; Nadam;
> loss: 0.0697 - distance_1_loss: 0.0202 - distance_2_loss: 0.0189 - distance_3_loss: 0.0098 - val_loss: 0.0463 - val_distance_1_loss: 0.0092 - val_distance_2_loss: 0.0086 - val_distance_3_loss: 0.0079

- trainer_v41117x06u01_re02_full (<- trainer_v41117x06u00_re01_freezeemb) <-> v41117x06u01 (Use lean dataset): epochs = 11; learning_rate = 1e-4;
> loss: 0.0663 - distance_1_loss: 0.0198 - distance_2_loss: 0.0184 - distance_3_loss: 0.0092 - val_loss: 0.0426 - val_distance_1_loss: 0.0080 - val_distance_2_loss: 0.0081 - val_distance_3_loss: 0.0076

- trainer_v41117x06u02_re03_full_newtr (<- v41117x06u01) <-> v41117x06u02: epochs = 21; learning_rate = 1e-3; RMSprop
> loss: 0.1086 - distance_1_loss: 0.0269 - distance_2_loss: 0.0388 - distance_3_loss: 0.0182 - val_loss: 0.0891 - val_distance_1_loss: 0.0174 - val_distance_2_loss: 0.0274 - val_distance_3_loss: 0.0200
> epochs = 7; learning_rate = 1e-4; Nadam
> loss: 0.0721 - distance_1_loss: 0.0216 - distance_2_loss: 0.0200 - distance_3_loss: 0.0133 - val_loss: 0.0553 - val_distance_1_loss: 0.0126 - val_distance_2_loss: 0.0111 - val_distance_3_loss: 0.0145

- trainer_v41117x06u02_re04_freezeemb (<- trainer_v41117x06u02_re03_full_newtr): epochs = 51; learning_rate = 1e-4; Nadam;
> loss: 0.0533 - distance_1_loss: 0.0174 - distance_2_loss: 0.0168 - distance_3_loss: 0.0087 - val_loss: 0.0347 - val_distance_1_loss: 0.0086 - val_distance_2_loss: 0.0085 - val_distance_3_loss: 0.0072

- trainer_v41117x06u03_re05_full (<- trainer_v41117x06u02_re04_freezeemb) <-> v41117x06u03 (Use lean dataset): epochs = 7; learning_rate = 1e-4;
> loss: 0.0503 - distance_1_loss: 0.0169 - distance_2_loss: 0.0157 - distance_3_loss: 0.0078 - val_loss: 0.0308 - val_distance_1_loss: 0.0071 - val_distance_2_loss: 0.0071 - val_distance_3_loss: 0.0068
```
```md
- trainer_v4117x07u00_re00_full -> v4117x07u00
> Move neutral_distance_scale further to opposite pole; Validating just counts for close_distance_scale cases; Exclude 1 neutral_distance_scale case type from dataset.
> Embedding output size of 7; Training process switch regime quicker (fewer epoch number).
> epochs = 3; learning_rate = 1e-3; RMSprop
> loss: 1.4397 - distance_1_loss: 0.4579 - distance_2_loss: 0.4575 - distance_3_loss: 0.4576 - val_loss: 0.0927 - val_distance_1_loss: 0.0201 - val_distance_2_loss: 0.0111 - val_distance_3_loss: 0.0130

- trainer_v4117x07u00_re01_freezeemb (<- trainer_v4117x07u00_re00_full): epochs = 3; learning_rate = 1e-3; RMSprop

- trainer_v4117x07u01_re02_full (<- trainer_v4117x07u00_re01_freezeemb) ~ v4117x07u01: epochs = 3; learning_rate = 1e-4; RMSprop
> loss: 0.0691 - distance_1_loss: 0.0164 - distance_2_loss: 0.0147 - distance_3_loss: 0.0108 - val_loss: 0.0531 - val_distance_1_loss: 0.0092 - val_distance_2_loss: 0.0088 - val_distance_3_loss: 0.0095

- trainer_v4117x07u02_re03_full_scratch (<< trainer_v4117x07u01_re02_full) ~ v4117x07u02: epochs = 1; learning_rate = 1e-3; RMSprop
> loss: 1.1945 - distance_1_loss: 0.2059 - distance_2_loss: 0.1866 - distance_3_loss: 0.1661 - val_loss: 0.4223 - val_distance_1_loss: 0.0254 - val_distance_2_loss: 0.0313 - val_distance_3_loss: 0.0342

- trainer_v4117x07u02_re04_freezeemb (<- trainer_v4117x07u02_re03_full_scratch): epochs = 2; learning_rate = 1e-3; RMSprop
> loss: 0.2883 - distance_1_loss: 0.0274 - distance_2_loss: 0.0448 - distance_3_loss: 0.0301 - val_loss: 0.2039 - val_distance_1_loss: 0.0111 - val_distance_2_loss: 0.0141 - val_distance_3_loss: 0.0201

- trainer_v4117x07u03_re05_full (<- trainer_v4117x07u02_re04_freezeemb) ~ v4117x07u03: epochs = 1; learning_rate = 1e-4; RMSprop
> loss: 0.2008 - distance_1_loss: 0.0170 - distance_2_loss: 0.0269 - distance_3_loss: 0.0171 - val_loss: 0.1523 - val_distance_1_loss: 0.0077 - val_distance_2_loss: 0.0097 - val_distance_3_loss: 0.0117
> Difference of 2 distances of pair of samples:  [0.034085553, 0.042949438, 0.037845995]

*****(terminated)*****
- trainer_v4117x07u04_re05_preemb_full_scratch (<< trainer_v4117x07u03_re05_full, <- v4117x07u03) ~ v4117x07u04: epochs = 1; learning_rate = 1e-3; RMSprop
> loss: 1.1408 - distance_1_loss: 0.1541 - distance_2_loss: 0.1889 - distance_3_loss: 0.1512 - val_loss: 0.4422 - val_distance_1_loss: 0.0339 - val_distance_2_loss: 0.0338 - val_distance_3_loss: 0.0365
> Difference of 2 distances of pair of samples:  [0.06925536, 0.064115696, 0.07904126]

- trainer_v4117x07u04_re06_freezeemb (<- trainer_v4117x07u04_re05_preemb_full_scratch): epochs = 2; learning_rate = 1e-3; RMSprop
> loss: 0.2639 - distance_1_loss: 0.0247 - distance_2_loss: 0.0314 - distance_3_loss: 0.0250 - val_loss: 0.2090 - val_distance_1_loss: 0.0151 - val_distance_2_loss: 0.0159 - val_distance_3_loss: 0.0211
> Difference of 2 distances of pair of samples:  [0.038376033, 0.0489659, 0.059671916]

- trainer_v4117x07u05_re07_full (<- trainer_v4117x07u04_re06_freezeemb) ~ v4117x07u05: epochs = 1; learning_rate = 1e-4; RMSprop
> loss: 0.1843 - distance_1_loss: 0.0159 - distance_2_loss: 0.0178 - distance_3_loss: 0.0133 - val_loss: 0.1620 - val_distance_1_loss: 0.0139 - val_distance_2_loss: 0.0117 - val_distance_3_loss: 0.0164
> Difference of 2 distances of pair of samples:  [0.036764663, 0.03671882, 0.04485482]
***** * *****

- trainer_v4117x07u03_re06_freezeemb (<- trainer_v4117x07u03_re05_full): epochs = 2; learning_rate = 1e-4; RMSprop
> loss: 0.1585 - distance_1_loss: 0.0147 - distance_2_loss: 0.0249 - distance_3_loss: 0.0142 - val_loss: 0.1219 - val_distance_1_loss: 0.0063 - val_distance_2_loss: 0.0078 - val_distance_3_loss: 0.0085
> Difference of 2 distances of pair of samples:  [0.027268814, 0.03409237, 0.03261827]

- trainer_v4117x07u04_re07_full (<- trainer_v4117x07u03_re06_freezeemb) ~ v4117x07u04: epochs = 1; learning_rate = 1e-4; RMSprop
> loss: 0.1349 - distance_1_loss: 0.0132 - distance_2_loss: 0.0201 - distance_3_loss: 0.0129 - val_loss: 0.1004 - val_distance_1_loss: 0.0062 - val_distance_2_loss: 0.0072 - val_distance_3_loss: 0.0074
> Difference of 2 distances of pair of samples:  [0.026287805, 0.030200014, 0.030710159]
```
```md
- trainer_v4117x08u00_re00_full_scratch -> v4117x08u00: Reduce comparators size; Reduce decoder dense layers into 7, 5, 3 respectively.
> epochs = 1; learning_rate = 1e-3; RMSprop;
> loss: 1.6991 - distance_1_loss: 0.4846 - distance_2_loss: 0.4812 - distance_3_loss: 0.4645 - val_loss: 0.1624 - val_distance_1_loss: 0.0133 - val_distance_2_loss: 0.0159 - val_distance_3_loss: 0.0255
> Difference of 2 distances of pair of samples:  [0.049636662, 0.038192183, 0.09868133]

- trainer_v4117x08u00_re01_freezeemb (<- trainer_v4117x08u00_re00_full):
> epochs = 2; learning_rate = 1e-3; RMSprop
> loss: 0.1818 - distance_1_loss: 0.0307 - distance_2_loss: 0.0290 - distance_3_loss: 0.0362 - val_loss: 0.1212 - val_distance_1_loss: 0.0121 - val_distance_2_loss: 0.0082 - val_distance_3_loss: 0.0213
> Difference of 2 distances of pair of samples:  [0.061726622, 0.046443712, 0.091107436]

- trainer_v4117x08u01_re02_full_scratch (v4117x08u00) -> v4117x08u01:
> epochs = 1; learning_rate = 1e-3; RMSprop;
> loss: 1.6805 - distance_1_loss: 0.4756 - distance_2_loss: 0.4731 - distance_3_loss: 0.4614 - val_loss: 0.1552 - val_distance_1_loss: 0.0236 - val_distance_2_loss: 0.0215 - val_distance_3_loss: 0.0138
> Difference of 2 distances of pair of samples:  [0.075441495, 0.07104477, 0.054643705]

- trainer_v4117x08u01_re03_freezeemb (<- trainer_v4117x08u01_re02_full_scratch):
> epochs = 2; learning_rate = 1e-3; RMSprop
> loss: 0.1566 - distance_1_loss: 0.0301 - distance_2_loss: 0.0289 - distance_3_loss: 0.0234 - val_loss: 0.1318 - val_distance_1_loss: 0.0197 - val_distance_2_loss: 0.0223 - val_distance_3_loss: 0.0223
> Difference of 2 distances of pair of samples:  [0.08107499, 0.0743386, 0.08680491]

- trainer_v4117x08u02_re04_full (trainer_v4117x08u01_re03_freezeemb) -> v4117x08u02:
> epochs = 1; learning_rate = 1e-4; RMSprop;
> loss: 0.1209 - distance_1_loss: 0.0197 - distance_2_loss: 0.0219 - distance_3_loss: 0.0212 - val_loss: 0.0801 - val_distance_1_loss: 0.0092 - val_distance_2_loss: 0.0095 - val_distance_3_loss: 0.0112
> Difference of 2 distances of pair of samples:  [0.050537713, 0.048360188, 0.0647546]

- trainer_v4117x08u02_re04_freezeemb (<- trainer_v4117x08u02_re04_full):
> epochs = 3; learning_rate = 1e-4; RMSprop
> loss: 0.0832 - distance_1_loss: 0.0150 - distance_2_loss: 0.0152 - distance_3_loss: 0.0128 - val_loss: 0.0628 - val_distance_1_loss: 0.0077 - val_distance_2_loss: 0.0077 - val_distance_3_loss: 0.0088
> Difference of 2 distances of pair of samples:  [0.040630516, 0.03862795, 0.052181438]

- trainer_v4117x08u03_re05_full (<- trainer_v4117x08u02_re04_freezeemb) -> v4117x08u03:
> epochs = 1; learning_rate = 1e-4; RMSprop;
> loss: 0.0746 - distance_1_loss: 0.0135 - distance_2_loss: 0.0138 - distance_3_loss: 0.0121 - val_loss: 0.0532 - val_distance_1_loss: 0.0066 - val_distance_2_loss: 0.0069 - val_distance_3_loss: 0.0076
> Difference of 2 distances of pair of samples:  [0.031054916, 0.033531073, 0.044989407]

- trainer_v4117x08u03_re06_freezeemb (<- trainer_v4117x08u03_re05_full):
> epochs = 3; learning_rate = 1e-4; RMSprop
> loss: 0.0622 - distance_1_loss: 0.0126 - distance_2_loss: 0.0125 - distance_3_loss: 0.0100 - val_loss: 0.0463 - val_distance_1_loss: 0.0064 - val_distance_2_loss: 0.0066 - val_distance_3_loss: 0.0071
> Difference of 2 distances of pair of samples:  [0.029187953, 0.030912718, 0.044271752]

- trainer_v4117x08u04_re07_full (<- trainer_v4117x08u03_re06_freezeemb) -> v4117x08u04:
> epochs = 3; learning_rate = 1e-4; RMSprop;
> loss: 0.0525 - distance_1_loss: 0.0117 - distance_2_loss: 0.0115 - distance_3_loss: 0.0096 - val_loss: 0.0377 - val_distance_1_loss: 0.0061 - val_distance_2_loss: 0.0064 - val_distance_3_loss: 0.0064
> Difference of 2 distances of pair of samples:  [0.025891265, 0.027960178, 0.03889767]

- trainer_v4117x08u04_re08_freezeemb_scratch_newtr (<- v4117x08u04):
> epochs = 3; learning_rate = 1e-3; RMSprop;
> loss: 1.4653 - distance_1_loss: 0.4443 - distance_2_loss: 0.4436 - distance_3_loss: 0.4386 - val_loss: 0.1518 - val_distance_1_loss: 0.0146 - val_distance_2_loss: 0.0157 - val_distance_3_loss: 0.0148
> Difference of 2 distances of pair of samples:  [0.026055524, 0.050250757, 0.040580038]

- trainer_v4117x08u05_re09_full (<- trainer_v4117x08u04_re08_freezeemb_scratch_newtr) -> v4117x08u05:
> epochs = 1; learning_rate = 1e-3; RMSprop;
> loss: 0.1710 - distance_1_loss: 0.0296 - distance_2_loss: 0.0330 - distance_3_loss: 0.0259 - val_loss: 0.1021 - val_distance_1_loss: 0.0126 - val_distance_2_loss: 0.0113 - val_distance_3_loss: 0.0140
> Difference of 2 distances of pair of samples:  [0.024734234, 0.03689484, 0.0431055]

- trainer_v4117x08u05_re10_freezeemb (<- trainer_v4117x08u05_re09_full):
> epochs = 3; learning_rate = 1e-4; RMSprop;
> loss: 0.0969 - distance_1_loss: 0.0152 - distance_2_loss: 0.0183 - distance_3_loss: 0.0129 - val_loss: 0.0766 - val_distance_1_loss: 0.0091 - val_distance_2_loss: 0.0088 - val_distance_3_loss: 0.0107
> Difference of 2 distances of pair of samples:  [0.0193223, 0.026459107, 0.032876227]

- trainer_v4117x08u06_re11_full (<- trainer_v4117x08u05_re10_freezeemb) -> v4117x08u06:
> epochs = 3; learning_rate = 1e-4; RMSprop;
> loss: 0.0629 - distance_1_loss: 0.0125 - distance_2_loss: 0.0128 - distance_3_loss: 0.0096 - val_loss: 0.0457 - val_distance_1_loss: 0.0069 - val_distance_2_loss: 0.0068 - val_distance_3_loss: 0.0063
> Difference of 2 distances of pair of samples:  [0.018687766, 0.021196494, 0.024526516]

- trainer_v4117x08u06_re12_freezeemb_scratch_newtr (<- v4117x08u06):
> epochs = 5; learning_rate = 1e-3; RMSprop;
> loss: 1.3780 - distance_1_loss: 0.4409 - distance_2_loss: 0.4373 - distance_3_loss: 0.4362 - val_loss: 0.0938 - val_distance_1_loss: 0.0123 - val_distance_2_loss: 0.0154 - val_distance_3_loss: 0.0126
> Difference of 2 distances of pair of samples:  [0.044141825, 0.061190974, 0.059907958]

- trainer_v4117x08u07_re13_full (<- trainer_v4117x08u06_re12_freezeemb_scratch_newtr) -> v4117x08u07:
> epochs = 1; learning_rate = 1e-3; RMSprop;
> loss: 0.1272 - distance_1_loss: 0.0301 - distance_2_loss: 0.0279 - distance_3_loss: 0.0230 - val_loss: 0.0753 - val_distance_1_loss: 0.0128 - val_distance_2_loss: 0.0102 - val_distance_3_loss: 0.0124
> Difference of 2 distances of pair of samples:  [0.04246903, 0.035525884, 0.0653189]

- trainer_v4117x08u07_re14_freezeemb (<- trainer_v4117x08u07_re13_full):
> epochs = 3; learning_rate = 1e-4; RMSprop;
> loss: 0.0759 - distance_1_loss: 0.0168 - distance_2_loss: 0.0135 - distance_3_loss: 0.0130 - val_loss: 0.0528 - val_distance_1_loss: 0.0069 - val_distance_2_loss: 0.0068 - val_distance_3_loss: 0.0078
> Difference of 2 distances of pair of samples:  [0.028895056, 0.02595199, 0.04808144]

- trainer_v4117x08u08_re15_full (<- trainer_v4117x08u07_re14_freezeemb) -> v4117x08u08:
> epochs = 3; learning_rate = 1e-4; RMSprop;
> loss: 0.0548 - distance_1_loss: 0.0131 - distance_2_loss: 0.0120 - distance_3_loss: 0.0091 - val_loss: 0.0373 - val_distance_1_loss: 0.0058 - val_distance_2_loss: 0.0061 - val_distance_3_loss: 0.0060
> Difference of 2 distances of pair of samples:  [0.017690765, 0.016607057, 0.03614808]

- trainer_v4117x08u08_re16_freezeemb (<- trainer_v4117x08u08_re15_full):
> epochs = 5; learning_rate = 1e-4; RMSprop;
> loss: 0.0480 - distance_1_loss: 0.0128 - distance_2_loss: 0.0109 - distance_3_loss: 0.0081 - val_loss: 0.0326 - val_distance_1_loss: 0.0055 - val_distance_2_loss: 0.0060 - val_distance_3_loss: 0.0052
> Difference of 2 distances of pair of samples:  [0.014509545, 0.017023578, 0.023978848]

- trainer_v4117x08u09_re17_full (<- trainer_v4117x08u08_re16_freezeemb) -> v4117x08u09:
> epochs = 3; learning_rate = 1e-4; RMSprop;
> val_loss: 0.0288 - val_distance_1_loss: 0.0054 - val_distance_2_loss: 0.0058 - val_distance_3_loss: 0.0049
> Difference of 2 distances of pair of samples:  [0.016934907, 0.018039314, 0.018901136]
```
```md
- trainer_v4117x09u00_re00_full_scratch (new structure of convs and rnns) -> v4117x09u00:
> epochs = 3; learning_rate = 1e-3; RMSprop;
> epochs = 7; learning_rate = 1e-4; RMSprop;
> loss: 1.5611 - distance_1_loss: 0.5059 - distance_2_loss: 0.5059 - distance_3_loss: 0.5056 - val_loss: 1.6297 - val_distance_1_loss: 0.5290 - val_distance_2_loss: 0.5294 - val_distance_3_loss: 0.5286
> Difference of 2 distances of pair of samples:  [0.021286268, 0.02132288, 0.022075232]

- trainer_v4117x09u01_re01_full_scratch (<- trainer_v4117x09u00_re00_full_scratch) -> v4117x09u01:
> grand epochs = 31; learning_rate = 1e-4; RMSprop;
> loss: 0.0473 - distance_1_loss: 0.0047 - distance_2_loss: 0.0045 - distance_3_loss: 0.0039 - val_loss: 0.0414 - val_distance_1_loss: 0.0029 - val_distance_2_loss: 0.0025 - val_distance_3_loss: 0.0020
> Difference of 2 distances of pair of samples:  [0.06160318, 0.047427166, 0.055081192]

- trainer_v4117x09u01_re02_freezeemb_scratch_newtr (<- v4117x09u01):
> epochs = 3; learning_rate = 1e-3; RMSprop;
> loss: 1.7542 - distance_1_loss: 0.5319 - distance_2_loss: 0.5316 - distance_3_loss: 0.5376 - val_loss: 1.6887 - val_distance_1_loss: 0.5161 - val_distance_2_loss: 0.5131 - val_distance_3_loss: 0.5143
> Difference of 2 distances of pair of samples:  [0.067706525, 0.051629826, 0.09510795]

- trainer_v4117x09u02_re03_full (<- trainer_v4117x09u01_re02_freezeemb_scratch_newtr) -> v4117x09u02:
> epochs = 3; learning_rate = 1e-3; RMSprop;
> loss: 0.2062 - distance_1_loss: 0.0354 - distance_2_loss: 0.0368 - distance_3_loss: 0.0344 - val_loss: 0.1748 - val_distance_1_loss: 0.0267 - val_distance_2_loss: 0.0270 - val_distance_3_loss: 0.0281
> Difference of 2 distances of pair of samples:  [0.14084293, 0.1417937, 0.14526837]

- trainer_v4117x09u03_re04_tictoc (<- trainer_v4117x09u02_re03_full) -> v4117x09u03:
> grand epochs = 11; learning_rate = 1e-4; RMSprop;
> loss: 0.0894 - distance_1_loss: 0.0050 - distance_2_loss: 0.0059 - distance_3_loss: 0.0054 - val_loss: 0.0906 - val_distance_1_loss: 0.0056 - val_distance_2_loss: 0.0062 - val_distance_3_loss: 0.0066
> Difference of 2 distances of pair of samples:  [0.057340298, 0.060244925, 0.061446954]
```
```md
- trainer_v411x10u00_re00_full_scratch_newtr  -> v411x10u00:
> epochs = 23; learning_rate = 1e-3; RMSprop;
> loss: 0.8648 - distance_1_loss: 0.2874 - distance_2_loss: 0.2832 - distance_3_loss: 0.2821 - val_loss: 0.9089 - val_distance_1_loss: 0.2983 - val_distance_2_loss: 0.3018 - val_distance_3_loss: 0.2970
> Divergence of 2 distances of pair of samples:  [0.07944203, 0.0715782, 0.06529673]
> (pred1: input_1 vs input_2) Distance of 2 different samples:  [0.96494114, 0.951791, 0.9579267]
> (pred2: input_2 vs input_1) Distance of 2 different samples:  [0.96239793, 0.95012164, 0.9563638]

- trainer_v411x10u00_re01_freezeemb_scratch_newtr (<- v411x10u00):
> epochs = 31; learning_rate = 1e-3; RMSprop;
> loss: 0.8968 - distance_1_loss: 0.2872 - distance_2_loss: 0.2878 - distance_3_loss: 0.2849 - val_loss: 0.9144 - val_distance_1_loss: 0.2921 - val_distance_2_loss: 0.2947 - val_distance_3_loss: 0.2915
> Divergence of 2 distances of pair of samples:  [0.054365903, 0.061435238, 0.05195756]
> (pred1: input_1 vs input_2) Distance of 2 different samples:  [0.9682776, 0.9652071, 0.9668705]
> (pred2: input_2 vs input_1) Distance of 2 different samples:  [0.96731555, 0.96434784, 0.9662962]

- trainer_v411x10u01_re02_tictoc (<- trainer_v411x10u00_re01_freezeemb_scratch_newtr) -> v411x10u01:
> epochs = 11; learning_rate = 1e-4; RMSprop;
> loss: 0.0427 - distance_1_loss: 0.0033 - distance_2_loss: 0.0035 - distance_3_loss: 0.0030 - val_loss: 0.0379 - val_distance_1_loss: 0.0013 - val_distance_2_loss: 0.0017 - val_distance_3_loss: 0.0021
> Divergence of 2 distances of pair of samples:  [0.021087246, 0.025890617, 0.026460841]
> (pred1: input_1 vs input_2) Distance of 2 different samples:  [0.9642659, 0.96205664, 0.95877695]
> (pred2: input_2 vs input_1) Distance of 2 different samples:  [0.9642279, 0.9617481, 0.95837444]

- trainer_v411x10u01_re03_freezeemb_scratch_newtr (<- v411x10u01):
> epochs = 33; learning_rate = 1e-3; RMSprop;
> loss: 0.8948 - distance_1_loss: 0.2887 - distance_2_loss: 0.2863 - distance_3_loss: 0.2845 - val_loss: 0.9063 - val_distance_1_loss: 0.2914 - val_distance_2_loss: 0.2891 - val_distance_3_loss: 0.2913
> Divergence of 2 distances of pair of samples:  [0.036684286, 0.03464032, 0.031166082]
> (pred1: input_1 vs input_2) Distance of 2 different samples:  [0.97010344, 0.97128147, 0.9670318]
> (pred2: input_2 vs input_1) Distance of 2 different samples:  [0.9697173, 0.971231, 0.96695256]

- trainer_v411x10u02_re04_tictoc (<- trainer_v411x10u01_re03_freezeemb_scratch_newtr) -> v411x10u02:
> epochs = 7; learning_rate = 1e-4; RMSprop;
> loss: 0.0419 - distance_1_loss: 0.0033 - distance_2_loss: 0.0037 - distance_3_loss: 0.0029 - val_loss: 0.0381 - val_distance_1_loss: 0.0018 - val_distance_2_loss: 0.0024 - val_distance_3_loss: 0.0021
> Divergence of 2 distances of pair of samples:  [0.027256358, 0.029980812, 0.027633287]
> (pred1: input_1 vs input_2) Distance of 2 different samples:  [0.9704229, 0.9704831, 0.9670166]
> (pred2: input_2 vs input_1) Distance of 2 different samples:  [0.9707592, 0.97048724, 0.96691155]

- trainer_v411x10u02_re05_freezeemb_scratch_newtr (<- v411x10u02):
> epochs = 17; learning_rate = 1e-3; RMSprop;
> loss: 0.9521 - distance_1_loss: 0.2890 - distance_2_loss: 0.2897 - distance_3_loss: 0.2878 - val_loss: 0.9869 - val_distance_1_loss: 0.3032 - val_distance_2_loss: 0.3019 - val_distance_3_loss: 0.2996
> Divergence of 2 distances of pair of samples:  [0.08110849, 0.07405919, 0.07383506]
> (pred1: input_1 vs input_2) Distance of 2 different samples:  [0.96304977, 0.96587664, 0.9675462]
> (pred2: input_2 vs input_1) Distance of 2 different samples:  [0.96230036, 0.96467155, 0.96672916]

- trainer_v411x10u03_re06_tictoc (<- trainer_v411x10u02_re05_freezeemb_scratch_newtr) -> v411x10u03:
> epochs = 11; learning_rate = 1e-4; RMSprop;
> loss: 0.0831 - distance_1_loss: 0.0038 - distance_2_loss: 0.0039 - distance_3_loss: 0.0040 - val_loss: 0.0814 - val_distance_1_loss: 0.0031 - val_distance_2_loss: 0.0032 - val_distance_3_loss: 0.0042o
> Divergence of 2 distances of pair of samples:  [0.038884707, 0.03850267, 0.044099044]
> (pred1: input_1 vs input_2) Distance of 2 different samples:  [0.9535243, 0.9563325, 0.9538221]
> (pred2: input_2 vs input_1) Distance of 2 different samples:  [0.9537119, 0.9563284, 0.9538446]
```
```md
- trainer_v5x10u00_re00_full_scratch_newtr  -> v5x10u00:
> epochs = 13; learning_rate = 1e-3; RMSprop;
> loss: 0.8725 - distance_1_loss: 0.2889 - distance_2_loss: 0.2834 - distance_3_loss: 0.2828 - val_loss: 0.9347 - val_distance_1_loss: 0.3027 - val_distance_2_loss: 0.3127 - val_distance_3_loss: 0.3028
> Divergence of 2 distances of pair of samples:  [0.06955675, 0.069537304, 0.065454386]
> (pred1: input_1 vs input_2) Distance of 2 different samples:  [0.95179355, 0.9245245, 0.94187266]
> (pred2: input_2 vs input_1) Distance of 2 different samples:  [0.9502538, 0.92379713, 0.9407532]

- trainer_v5x10u00_re01_freezeemb_scratch_newtr (<- v5x10u00):
> epochs = 17; learning_rate = 1e-3; RMSprop;
> loss: 0.9727 - distance_1_loss: 0.2981 - distance_2_loss: 0.2981 - distance_3_loss: 0.2988 - val_loss: 0.9679 - val_distance_1_loss: 0.2960 - val_distance_2_loss: 0.2978 - val_distance_3_loss: 0.2993
> Divergence of 2 distances of pair of samples:  [0.052788354, 0.05439594, 0.05105939]
> (pred1: input_1 vs input_2) Distance of 2 different samples:  [0.9587563, 0.9511156, 0.9499051]
> (pred2: input_2 vs input_1) Distance of 2 different samples:  [0.9591355, 0.9512607, 0.95055836]

- trainer_v5x10u01_re02_tictoc (<- trainer_v5x10u00_re01_freezeemb_scratch_newtr) -> v5x10u01:
> epochs = 7; learning_rate = 1e-3; RMSprop;
> loss: 0.0823 - distance_1_loss: 0.0042 - distance_2_loss: 0.0044 - distance_3_loss: 0.0056 - val_loss: 0.0772 - val_distance_1_loss: 0.0025 - val_distance_2_loss: 0.0026 - val_distance_3_loss: 0.0045
> Divergence of 2 distances of pair of samples:  [0.03146939, 0.03593482, 0.036283962]
> (pred1: input_1 vs input_2) Distance of 2 different samples:  [0.9630693, 0.95593, 0.9531112]
> (pred2: input_2 vs input_1) Distance of 2 different samples:  [0.9631203, 0.9560219, 0.9525938]

- trainer_v5x10u01_re02_freezeemb_scratch_newtr (<- v5x10u01):
> epochs = 7; learning_rate = 1e-3; RMSprop;
> loss: 1.2672 - distance_1_loss: 0.3479 - distance_2_loss: 0.3680 - distance_3_loss: 0.3697 - val_loss: 1.1989 - val_distance_1_loss: 0.3297 - val_distance_2_loss: 0.3464 - val_distance_3_loss: 0.3507
> Divergence of 2 distances of pair of samples:  [0.1353924, 0.15888196, 0.12628779]
> (pred1: input_1 vs input_2) Distance of 2 different samples:  [0.94780463, 0.9214602, 0.9055155]
> (pred2: input_2 vs input_1) Distance of 2 different samples:  [0.9483188, 0.92170423, 0.9056003]

- trainer_v5x10u02_re03_full (<- trainer_v5x10u01_re02_freezeemb_scratch_newtr) -> v5x10u02:
> epochs = 5; learning_rate = 1e-3; RMSprop;
> loss: 1.0051 - distance_1_loss: 0.3025 - distance_2_loss: 0.3022 - distance_3_loss: 0.3019 - val_loss: 0.9648 - val_distance_1_loss: 0.2910 - val_distance_2_loss: 0.2957 - val_distance_3_loss: 0.2905
> Divergence of 2 distances of pair of samples: [0.092219986, 0.09734352, 0.083635576]
> (pred1: input_1 vs input_2) Distance of 2 different samples:  [0.9757183, 0.96721876, 0.9754189]
> (pred2: input_2 vs input_1) Distance of 2 different samples:  [0.97595894, 0.9679688, 0.9758508]

- trainer_v5x10u03_re04_tictoc (<- trainer_v5x10u02_re03_full) -> v5x10u03:
> epochs = 97; learning_rate = 1e-4; RMSprop;
> loss: 0.0479 - distance_1_loss: 0.0055 - distance_2_loss: 0.0053 - distance_3_loss: 0.0039 - val_loss: 0.0420 - val_distance_1_loss: 0.0033 - val_distance_2_loss: 0.0033 - val_distance_3_loss: 0.0023
> Divergence of 2 distances of pair of samples:  [0.080105916, 0.07581319, 0.05385025]
> (pred1: input_1 vs input_2) Distance of 2 different samples:  [0.97792566, 0.96241385, 0.9666872]
> (pred2: input_2 vs input_1) Distance of 2 different samples:  [0.9787123, 0.9633786, 0.96718156]
```
