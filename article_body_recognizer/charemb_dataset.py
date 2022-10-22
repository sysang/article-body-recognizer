import random
import time
import _thread
from collections import deque
from pathlib import Path

import numpy as np

from article_body_recognizer.char_dict import vocabularies as vocab
from article_body_recognizer.training_utils import charrnn_encode_sequence


SLEEP_TIME = 0.10

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


def load_data(source_path, split=None, loop=True):
  path_obj = Path(source_path)
  dataset = []

  for file in path_obj.iterdir():

    with open(file.absolute()) as reader:
      lines = reader.readlines()
      for line in lines:
        dataset.append(line)

  return dataset


def load_data_stream(source_path, split=None, loop=True):
  path_obj = Path(source_path)

  while True:
    for file in path_obj.iterdir():

      with open(file.absolute()) as reader:
        lines = reader.readlines()
        for line in lines:
          yield line

    if not loop:
      return None


def count_dataset_size(data_source, cfg):
  max_length = cfg['max_length']
  min_length = cfg['min_length']

  dataset = load_data(data_source, loop=False)

  counter = 0
  for s in dataset:
    lgth = len(s)
    if min_length > lgth or lgth > max_length:
      continue

    counter += 1

  return counter


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

    if _count_batch % batch_size == 0:
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
  _thread.start_new_thread(buffer_data, (), {'cfg': _cfg, 'dataset': _dataset, 'pri_buffer': _pri_buffer, 'split_type': _split_type, 'breaker': breaker, 'model': _model})
  return breaker

