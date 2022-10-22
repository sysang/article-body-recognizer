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

```javascript
var nb = IPython.notebook;
var kernel = IPython.notebook.kernel;
var command = "NOTEBOOK_FULL_PATH = '" + nb.base_url + nb.notebook_path + "'";
kernel.execute(command);
```
```python
# The working notebook file path is stored in NOTEBOOK_FULL_PATH
# Change working directory to project root which is parent of NOTEBOOK_FULL_PATH
# Typing Shift + Enter on obove cell to help proper javascript execution

import os
import time
from pathlib import Path
from jupyter_core.paths import jupyter_config_dir
from traitlets.config import Config

def get_config():
  return Config()

try:
  config_file = os.path.join(jupyter_config_dir(), 'jupyter_notebook_config.py')
  config_content = open(config_file).read()
  exec(config_content)
  root_dir = c['ContentsManager']['root_dir']
  project_root = Path(root_dir + NOTEBOOK_FULL_PATH).parent
  os.chdir(project_root)
  print(f'Working directory has been changed to {project_root}')
except:
  raise Exception('Could not automatically change working directory.')

# os.chdir('path/to/project/root/')
```
```python
import math
import pickle

import numpy as np

from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adagrad, Nadam

from tensorflow.keras.callbacks import TensorBoard, Callback

from article_body_recognizer.system_specs import char_emb_training_specs
from article_body_recognizer.char_dict import vocabularies as vocab
from article_body_recognizer.ANNs import charemb_comparator
from article_body_recognizer.ANNs import charemb_network
from article_body_recognizer import charemb_dataset
from article_body_recognizer.training_utils import charrnn_encode_sequence
from article_body_recognizer.charemb_training import Tester

import importlib
importlib.reload(charemb_dataset)

create_generator = charemb_dataset.create_generator
start_buffer_data_thread = charemb_dataset.start_buffer_data_thread


```
```python
cfg = {
  'emb_trainable': True,
  'pretrained_emb_vers': None, # str or None
  'new_emb_vers': 'v5x10u03',
  'pretrained_trainer_vers': 'trainer_v5x10u03_re04_tictoc', # str or None
  'new_trainer_version' : 'trainer_v5x10u03_re04_tictoc',
  'lean_dataset': True,
  'pretrained_model_vers': None,  # if set this will get higher priority than pretrained_trainer_vers
  'embedding_model_class': charemb_network.CharEmbeddingV5,
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
```
```python


def calculate_training_steps(_cfg, _dataset):
  sample_quantity = len(_dataset) * 3 if _cfg['lean_dataset'] else len(_dataset) * 4
  steps_per_epoch = math.ceil(sample_quantity/_cfg['batch_size'])

  return steps_per_epoch, sample_quantity

def calculate_validating_steps(_cfg, _dataset):
  sample_quantity = len(_dataset) * 2 if _cfg['lean_dataset'] else len(_dataset) * 4
  steps_per_epoch = math.ceil(sample_quantity/_cfg['batch_size'])

  return steps_per_epoch, sample_quantity

validating_dataset = charemb_dataset.load_data('article_body_recognizer/charemb-dataset/validating', cfg)
training1_dataset = charemb_dataset.load_data('article_body_recognizer/charemb-dataset/training1', cfg)
training2_dataset = charemb_dataset.load_data('article_body_recognizer/charemb-dataset/training2', cfg)

BATCH_SIZE = cfg['batch_size']
BUFFER_SIZE = cfg['buffer_size']
_pribuf_looping = cfg['pribuf_looping']
_lean_dataset = cfg['lean_dataset']

steps_per_epoch, training_sample_quantity = calculate_training_steps(_cfg=cfg, _dataset=training1_dataset)
print('[INFO] training_sample_quantity', training_sample_quantity)
print('[INFO] training steps_per_epoch: ', steps_per_epoch)

validation_steps, validating_sample_quantity = calculate_validating_steps(_cfg=cfg, _dataset=validating_dataset)
print('[INFO] validating_sample_quantity', validating_sample_quantity)
print('[INFO] validation_steps: ', validation_steps)

```
```python
def do_training(_cfg):
  EPOCHS = _cfg['epochs']
  BATCH_SIZE = _cfg['batch_size']
  BUFFER_SIZE = _cfg['buffer_size']
  pribuf_looping = cfg['pribuf_looping']

  pretrained_trainer_vers = cfg['pretrained_trainer_vers']
  pretrained_model_vers = cfg['pretrained_model_vers']
  version = pretrained_model_vers if pretrained_model_vers else pretrained_trainer_vers
  print(f'[INFO] Model {version} will be used to generate truth labels of dataset.')

  pretrained_model = charemb_comparator.CharembComparatorV1(cfg)
  pretrained_model.load_weights(f"article_body_recognizer/pretrained_embedding/trainers/{version}.h5")
  pretrained_model.trainable = False

  _validating_dataset = validating_dataset

  tmp_weight_filepath = "article_body_recognizer/tmp/model_weights.h5"
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

    model = charemb_comparator.CharembComparatorV1(_cfg)
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
char_model = do_training(_cfg=cfg)

```
```python
trainable = cfg['emb_trainable']
new_emb_vers = cfg['new_emb_vers']
if trainable and new_emb_vers:
  char_embedding_layer_weights = char_model.get_layer('char_embedding').get_weights()
  with open(f'article_body_recognizer/pretrained_embedding/{new_emb_vers}.pickle', 'wb') as f:
      pickle.dump(char_embedding_layer_weights, f, pickle.HIGHEST_PROTOCOL)
```
```python
new_trainer_version = cfg['new_trainer_version']
char_model.save_weights(f"article_body_recognizer/pretrained_embedding/trainers/{new_trainer_version}.h5", overwrite=True)
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
