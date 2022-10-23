import random
import time
import _thread
from collections import deque
from pathlib import Path

import numpy as np

from article_body_recognizer.char_dict import vocabularies as vocab
from article_body_recognizer.training_utils import charrnn_encode_sequence


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
