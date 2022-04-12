---
jupyter:
  jupytext:
    cell_metadata_filter: -all
    formats: md,ipynb
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.7.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import os
os.chdir('/workspace/upwork/martien_brouver/mylapi/scraping')
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import math
import html
import base64
import pickle
import jsonlines
import json
from lxml import etree

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Input, Embedding, Dense, LSTM, Bidirectional, Dropout
from tensorflow.keras.layers import concatenate, Reshape, SpatialDropout1D
from tensorflow.keras.models import Model, Sequential

from tensorflow import config as config
from tensorflow.keras.utils import to_categorical

from textgenrnn import textgenrnn
from textgenrnn.AttentionWeightedAverage import AttentionWeightedAverage
from textgenrnn.utils import (
    generate_after_epoch,
    save_model_weights,
    textgenrnn_encode_sequence,
    textgenrnn_generate,
    textgenrnn_texts_from_file,
    textgenrnn_texts_from_file_context,
)
```
```python
loaded_model = tf.keras.models.load_model('parser/models/v6')
```
```python
from importlib import reload
from scraping import utils, transformer, configurations

reload(utils)
reload(transformer)
reload(configurations)

cfg = configurations.cfg
vocab= configurations.vocab

filter_html = utils.filter_html
transform_top_level_nodes_to_sequence = transformer.transform_top_level_nodes_to_sequence

num_nodes = cfg['num_encoded_nodes']
max_length = cfg['max_length']

class Extractor():
  def __init__(self, model):
    self.model = model

  def getContent(self, html_text):
    _text = filter_html(html_text)
    root = etree.XML(_text)
    hierarchy = transform_top_level_nodes_to_sequence(_text)

    x = np.zeros((num_nodes, max_length))
    x_index = 0
    for node in hierarchy:
      encoded_text = textgenrnn_encode_sequence(_text[-max_length:], vocab, max_length)
      x[x_index] = encoded_text
      x_index += 1
      if x_index >= num_nodes:
        break

    content_preds, _ = self.model(np.array([x]))
    content_preds = content_preds.numpy()

    content_node_number = content_preds.argmax()
    print('content_node_number: ', content_node_number)

    index = [el for el in root.iter()]
    if content_node_number > len(index):
      raise Exception('The predicted index was out of range. Can not extract content from html text.')

    content_node = index[content_node_number]
    content = str(etree.tostring(content_node), 'utf-8') # Need to filter out html tag

    return content

F_PATH = "scraping/etc/tmp.html"

with open(F_PATH, 'r') as f:
  html_text = f.read()

extractor = Extractor(loaded_model)
content = extractor.getContent(html_text)
```
