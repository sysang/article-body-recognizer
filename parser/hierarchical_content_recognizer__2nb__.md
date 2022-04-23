---
jupyter:
  jupytext:
    cell_metadata_filter: -all
    formats: md,ipynb
    main_language: python
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.7.1
---

```python
import os
os.chdir('/workspace/HtmlSecReg/')
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
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adagrad, Nadam
from tensorflow.keras.optimizers.schedules import InverseTimeDecay
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM, Bidirectional, Dropout
from tensorflow.keras.layers import concatenate, Reshape, SpatialDropout1D, Conv1D, Flatten, AveragePooling1D, MaxPool1D, Average, Maximum, Multiply, Add
from tensorflow.keras.models import Model, Sequential

from tensorflow import config as config
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import TensorBoard, Callback

from scraping.utils import base64StrDecode
from scraping import models
from scraping import vers_models
from scraping.system_specs import char_emb_training_specs
from scraping.transformer import LxmlTree
from scraping.transformer import transform_top_level_nodes_to_sequence
```
```python
# from urllib import parse

# p = re.compile('(https?\:\/\/)(www\.)?((?:[0-9a-z\-\_]+\.)+)(\w+\/?).*$')
# repl = lambda matchObj: matchObj.groups()[3] +  matchObj.groups()[2]

# SPLIT_NAME = 'validating'
# # datasource = 'parser/hierarchical_content_training_dataset.jl'
# datasource = 'parser/hierarchical_content_val_dataset.jl'

# with open(datasource, 'r') as reader:
#   lines = reader.readlines()

#   for line in lines:
#     obj = json.loads(line)
#     url = obj['url']
#     url = parse.unquote(obj['url'])
#     new_fpath = p.sub(repl, url) + 'jl'
#     new_fpath = new_fpath.replace('/', '.')
#     new_fpath = f'parser/htmlsecreg-dataset/{SPLIT_NAME}/{new_fpath}'
#     print(new_fpath)

#     opmode = 'a' if os.path.exists(new_fpath) else 'x'

#     with open(new_fpath, opmode) as new_file:
#       new_file.write(line)
```

```python
PRESET = [
  {
    'SCHEME': 0,
    'dropout_fine_tuning': 0,
    'batch_size': 57,
    'epochs': 109,
    'optimizer': RMSprop,
    'learning_rate': 5e-4,
  },
  {
    'SCHEME': 1,
    'dropout_fine_tuning': 0.03,
    'batch_size': 43,
    'epochs': 109,
    'optimizer': Nadam,
    'learning_rate': 5e-4,
  },
]

SCHEME = 1

cfg = {
    'pretrained_emb_vers': 'v5x10u03',
    'pretrained_version': 'v3x24x00x00r71',
    'new_version': 'v3x24x00x00r72',
    'dropout_fine_tuning': PRESET[SCHEME]['dropout_fine_tuning'],
    'max_length': 75000,
    'sequence_clip_ratio': 0.07,
    'num_classes': char_emb_training_specs['NUM_CLASSES'],
    'num_categories': 501, #  to validate dataset does not have item's value exceed this
    'optimizer': PRESET[SCHEME]['optimizer'],
    'learning_rate': PRESET[SCHEME]['learning_rate'],
    'batch_size': PRESET[SCHEME]['batch_size'],
    'epochs': PRESET[SCHEME]['epochs'],
    'buffer_size': 17,
    'pribuf_looping': True, # If is True then buffer_size makes no affect and is set to steps_per_epoch
    # 'decay_steps': 10,
    # 'decay_rate': 0.2479,  # Formular: (2e-4 / 5e-5 - 1) / floor(121 / 10)
    # 'model': vers_models.HierarchyV3_100100,
    'emb_trainable': False,
    'decoder_trainable': True,
    'model': models.Hierarchy,
  }

print(cfg)
SLEEP_TIME = 0.15
```
```python
from scraping.char_dict import vocabularies as vocab
print('Vocabuary index: ', vocab)

def charrnn_encode_sequence(text, vocab, maxlen):
    '''
    Encodes a text into the corresponding encoding for prediction with
    the model.
    '''

    oov = vocab['oov']
    encoded = np.array([vocab.get(x, oov) for x in text])
    return sequence.pad_sequences([encoded], padding='post', maxlen=maxlen)

```
```python
def is_jsonline_file(fname):
  is_jlext_p = re.compile('\.jl$')

  return is_jlext_p.search(fname) is not None

def count_dataset_size(source_path):
  counter = 0

  path_obj = Path(source_path)
  for file in path_obj.iterdir():
    if not is_jsonline_file(file.name):
      continue

    with open(file.absolute()) as reader:
      counter += len(reader.readlines())

  return counter

def load_dataset(source_path, split=None):
  path_obj = Path(source_path)
  is_jlext_p = re.compile('\.jl$')
  dataset = []

  for file in path_obj.iterdir():
    if not is_jsonline_file(file.name):
      continue

    with jsonlines.open(file.absolute()) as reader:
      for obj in reader:
        html_text = base64StrDecode(obj['text'])
        tree = LxmlTree(html_text)

        title_node_ind = int(obj['title'])
        article_node_ind = int(obj['article'])

        try:
          article_distributed_probability = tree.calc_distributed_probability_of_truth(target_number=article_node_ind)
          title_distributed_probability = tree.calc_distributed_probability_of_truth(target_number=title_node_ind, percentage_of_itinerary=0.5)
        except:
          raise Exception(obj)

        dataset.append({
            'hierarchy': transform_top_level_nodes_to_sequence(lxmltree=tree),
            'article_distributed_probability': article_distributed_probability,
            'title_distributed_probability': title_distributed_probability,
            'url': obj['url']
          })

  return dataset

validating_dataset_file = 'parser/htmlsecreg-dataset/tmp/validating_dataset.pickle'
training_dataset_file = 'parser/htmlsecreg-dataset/tmp/training_dataset.pickle'

validating_dataset = load_dataset('parser/htmlsecreg-dataset/validating', cfg)
random.shuffle(validating_dataset)
# with open(validating_dataset_file, 'wb') as f:
#     pickle.dump(validating_dataset, f, pickle.HIGHEST_PROTOCOL)

training_dataset = load_dataset('parser/htmlsecreg-dataset/training', cfg)
random.shuffle(training_dataset)
# with open(training_dataset_file, 'wb') as f:
#     pickle.dump(training_dataset, f, pickle.HIGHEST_PROTOCOL)

# with open(validating_dataset_file, 'rb') as f:
#   validating_dataset = pickle.load(f)

# with open(training_dataset_file, 'rb') as f:
#   training_dataset = pickle.load(f)
```
```python
from urllib import parse

# max_title_node = 0
# max_article_node = 0
# for item in (training_dataset + validating_dataset):
#   title_node = int(item['title_node'])
#   article_node = int(item['article_node'])
#   if title_node > max_title_node:
#     print('title_node:', title_node)
#     print(' --- ', parse.unquote(item['url']), '\n')
#     max_title_node = title_node

#   if article_node > max_article_node:
#     print('article_node: ', article_node)
#     print(' --- ', parse.unquote(item['url']), '\n')
#     max_article_node = article_node

# print('max_title_node: ', max_title_node)
# print('max_article_node: ', max_article_node)
```
```python
def buffer_data(cfg, dataset, pri_buffer, pribuf_looping=False):
  BATCH_SIZE = cfg['batch_size']
  num_categories = cfg['num_categories']
  max_length = cfg['max_length']
  sequence_clip_ratio = cfg['sequence_clip_ratio']

  if pribuf_looping:
    BUFFER_SIZE = pri_buffer.maxlen - 1
  else:
    BUFFER_SIZE = pri_buffer.maxlen / 2

  def check_if_append_will_happen(probability=0.5):
    to_happen = random.random() <= probability
    return to_happen

  def _queue_or_count(_X_batch, _Y_article, _Y_title, _count_batch):

    # Firstly, count batch's volume
    _count_batch += 1

    if _count_batch % BATCH_SIZE == 0:
      _X_batch = np.array(_X_batch)
      _Y_article = np.array(_Y_article)
      _Y_title = np.array(_Y_title)

      _Y_batch = {
        'abstract_content_output': _Y_article,
        'detail_content_output': _Y_article,
        'detail_title_output': _Y_title,
      }

      batch = (_X_batch, _Y_batch)

      pri_buffer.appendleft(batch)
      # print('pri_buffer.appendleft(): ', len(pri_buffer))

      # if the primary buffer is full then wait for the number of remaining items less than BUFFER_SIZE
      if len(pri_buffer) >= pri_buffer.maxlen:
        while len(pri_buffer) > BUFFER_SIZE:
          time.sleep(SLEEP_TIME)

      _X_batch = []
      _Y_article = []
      _Y_title = []
      _count_batch = 0

    return (_X_batch, _Y_article, _Y_title, _count_batch)

  def to_distributed_categorial(_distributed):
    _arr = [0.0] * num_categories

    for ind, prob in _distributed:
      _arr[ind] = prob

    return _arr

  while True:
    if len(pri_buffer) < pri_buffer.maxlen:
      X_batch = []
      Y_batch = []
      Y_title = []
      Y_article = []
      count_batch = 0

      for item in dataset:

        hierarchy = item['hierarchy']
        article_distributed_probability = item['article_distributed_probability']
        title_distributed_probability = item['title_distributed_probability']

        max_hierarchy_index = len(hierarchy)
        if max_hierarchy_index > (max_length * (1 + sequence_clip_ratio)):
          print("[WARNING] Data item exceeds maximum length(%s > %s), verify dataset to fix this issue. Debugging info: %s"
            % (max_hierarchy_index, max_length, item['url']))

        X_batch.append(charrnn_encode_sequence(hierarchy, vocab, max_length)[0])

        Y_article.append(to_distributed_categorial(article_distributed_probability))
        Y_title.append(to_distributed_categorial(title_distributed_probability))

        X_batch, Y_title, Y_article, count_batch = _queue_or_count(X_batch, Y_article, Y_title, count_batch)


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


training_sample_quantity = len(training_dataset)
print('training_sample_quantity: ', training_sample_quantity)
validating_sample_quantity = len(validating_dataset)
print('validating_sample_quantity: ', validating_sample_quantity)

BATCH_SIZE = cfg['batch_size']
steps_per_epoch = math.ceil(training_sample_quantity/BATCH_SIZE)
validation_steps = math.ceil(validating_sample_quantity/BATCH_SIZE)

BUFFER_SIZE = cfg['buffer_size']
_pribuf_looping = cfg['pribuf_looping']

# Training generator
training_queue, training_generator = create_generator(buffer_size=BUFFER_SIZE, pribuf_looping=_pribuf_looping, steps_per_epoch=steps_per_epoch)
_thread.start_new_thread(buffer_data, (cfg, training_dataset, training_queue, _pribuf_looping))

# Validating generator
validating_queue, validating_generator = create_generator(buffer_size=BUFFER_SIZE, pribuf_looping=_pribuf_looping, steps_per_epoch=validation_steps)
_thread.start_new_thread(buffer_data, (cfg, validating_dataset, validating_queue, _pribuf_looping))

```
```python
# Test datasets
# for b in training_generator:
#   print('test')
#   time.sleep(0.5)

# index = 0
# batch = next(training_generator)
```


```python
# input = batch[0]
# print('input shape: ', input.shape)
# print('input item length []: ', input[index].shape)
# print('input []: ', input[index].tolist())
# labels = batch[1]
# print('labels keys: ', labels.keys())
# print('detail_content_output shape: ', labels['detail_content_output'].shape)
# print('detail_content_output []: ', labels['detail_content_output'][index])
# print('detail_title_output shape: ', labels['detail_title_output'].shape)
# print('detail_title_output []: ', labels['detail_title_output'][index])
# print('abstract_content_output shape: ', labels['abstract_content_output'].shape)
# print('abstract_content_output [1]: ', labels['abstract_content_output'][index])
# index += 1
```
```python
def create_hierarchy_model(cfg):
  model = cfg['model'](cfg)
  return model

hierarchy_model = create_hierarchy_model(cfg)

pretrained_emb_vers = cfg['pretrained_emb_vers']
if pretrained_emb_vers:
  print(f'Load embedding layer weights from {pretrained_emb_vers}....')
  objectRep = open(f"parser/pretrained_embedding/{pretrained_emb_vers}.pickle", "rb")
  char_embedding_layer_weights = pickle.load(objectRep)
  hierarchy_model.get_layer('char_embedding').set_weights(char_embedding_layer_weights)
  objectRep.close()

pretrained_version = cfg['pretrained_version']
if pretrained_version:
  print(f'Load weights from {pretrained_version}....')

  hierarchy_model.load_weights(f"parser/models/{pretrained_version}.h5")

hierarchy_model.summary()
```
```python
EPOCHS = cfg['epochs']
BATCH_SIZE = cfg['batch_size']

class TrainOverValidationLossRatioCallback(Callback):
    def on_epoch_end(self, epoch, logs):
        lr = self.model.optimizer.lr
        loss = logs['loss']
        val_loss = logs['val_loss']
        # compensate for model that has loss decrease forcely
        # beause the validating loss tends to change inversely against training loss
        compensated = math.sqrt(loss)
        data = (val_loss - loss) / loss / compensated
        tf.summary.scalar('validation_over_training_loss', data=data, step=epoch)

logdir = "logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir + "/metrics")
file_writer.set_as_default()

hierarchy_model.fit(
     training_generator,
     batch_size=BATCH_SIZE,
     steps_per_epoch=steps_per_epoch,
     epochs=EPOCHS,
     validation_data=validating_generator,
     validation_batch_size=BATCH_SIZE,
     validation_steps=validation_steps,
     shuffle='batch',
     callbacks=[TensorBoard(log_dir=logdir, histogram_freq=1), TrainOverValidationLossRatioCallback()]
   )

```
```python
new_version = cfg['new_version']
if new_version:
  hierarchy_model.save_weights(f'parser/models/{new_version}.h5')
```
