import random
from collections import deque

import numpy as np

from article_body_recognizer.char_dict import vocabularies as vocab
from article_body_recognizer.training_utils import charrnn_encode_sequence

class Tester:
  def __init__(self, model, dataset, cfg):
    self.size = 501
    self.model = model
    self.dataset = dataset
    self.arguments = deque(maxlen=self.size)
    self.result_cache = {
      'preds1': None,
      'preds2': None,
    }
    self.input_max_length = cfg['max_length']

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

      _data1['input_1'].append(charrnn_encode_sequence(item, vocab, self.input_max_length)[0])
      _data1['input_2'].append(charrnn_encode_sequence(_argument, vocab, self.input_max_length)[0])

      # Reverse the order of item and argument
      _data2['input_1'].append(charrnn_encode_sequence(_argument, vocab, self.input_max_length)[0])
      _data2['input_2'].append(charrnn_encode_sequence(item, vocab, self.input_max_length)[0])

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

