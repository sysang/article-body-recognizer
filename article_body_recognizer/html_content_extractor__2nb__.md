---
jupyter:
  jupytext:
    cell_metadata_filter: -all
    formats: md,ipynb
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import os
os.chdir('/workspace/HtmlSecReg')
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
from tensorflow.keras.optimizers import Adam, RMSprop

from tensorflow.keras.layers import Input, Embedding, Dense, LSTM, Bidirectional, Dropout
from tensorflow.keras.layers import concatenate, Reshape, SpatialDropout1D
from tensorflow.keras.models import Model, Sequential

from tensorflow import config as config
from tensorflow.keras.utils import to_categorical

from requests_html import HTMLSession

from article_body_recognizer.ANNs.v3x24x00 import HierarchyV3x24x00
from article_body_recognizer.system_specs import char_emb_training_specs
from article_body_recognizer.training_utils import charrnn_encode_sequence
```
```python

```
```python
from importlib import reload
from article_body_recognizer.char_dict import vocabularies as vocab
from article_body_recognizer import utils, transformer
from article_body_recognizer.transformer import LxmlTree

reload(utils)
reload(transformer)

filter_html = utils.filter_html
transform_top_level_nodes_to_sequence = transformer.transform_top_level_nodes_to_sequence


class Recognizer():
  TEXT_LENGTH_POS = 5

  def __init__(self):
    self.cfg = {
      'num_categories': 501,
      'num_classes': char_emb_training_specs['NUM_CLASSES'],
      'max_length': 75000,
      'learning_rate': 5e-4,
      'emb_trainable': False,
      'decoder_trainable': False,
      'optimizer': RMSprop,
    }
    self.model = HierarchyV3x24x00(self.cfg)
    self.model.load_weights(f"article_body_recognizer/models/v3x24x00x00r44.h5")

  def parse_html_to_plain(self, node):
    texts = []
    for txt in node.itertext():
      txt = txt.strip().replace('\t', '').replace('\n', '')
      # check fo empty string
      if txt:
        texts.append(txt)

    return ' '.join(texts)


  def pred_to_content(self, content_node_index, pred):
    pred_np = pred.numpy()
    pred_np = pred_np.squeeze()
    top_k_num = 5
    # likelihook is ascending from left to right
    # higher level nodes should be left-oriented
    top_k = pred_np.argpartition(-top_k_num)[-top_k_num:]
    contents = []

    for num in top_k.tolist():
      if num >= len(content_node_index):
        print('[INFO] The predicted index was out of range. Can not extract content from html text.')
        continue

      content_node = content_node_index[num]
      content = self.parse_html_to_plain(content_node)
      prob = float(pred_np[num])
      children = list(content_node.iterchildren()) if getattr(content_node, 'iterchildren', None) else []
      contents.append((num, content_node, prob, children, content, len(content)))

    # sort by content length
    # content length is descending from left to right
    # higher level nodes should be left-oriented
    contents.sort(key=lambda r: r[self.TEXT_LENGTH_POS], reverse=True)

    if not len(contents):
      return None

    result = contents[0]
    for ctn in contents[1:]:
      # check if successor is child of and successor's probability is higher
      if ctn[1] in result[3] and ctn[2] >= result[2]:
        result = ctn

    # print('target: ', result)

    return result


  def getContent(self, html_text):
    simplified_html = filter_html(html_text)
    lxmltree = LxmlTree(simplified_html)
    content_node_index = [el for el in lxmltree.root.iter()]

    original = self.parse_html_to_plain(lxmltree.root)
    print('[Original (length of %s): ]' % len(original), original)

    hierarchy = transform_top_level_nodes_to_sequence(lxmltree)
    encoded_text = charrnn_encode_sequence(hierarchy, vocab, self.cfg['max_length'])

    preds= self.model(encoded_text)
    abstract_pred = preds[0]
    detail_pred = preds[1]

    candidate_1 = self.pred_to_content(content_node_index, abstract_pred)
    candidate_2 = self.pred_to_content(content_node_index, detail_pred)

    # priority longer content
    candidate = candidate_1 if candidate_1[self.TEXT_LENGTH_POS] > candidate_2[self.TEXT_LENGTH_POS] else candidate_2

    return candidate

```
```python
# F_PATH = "article_body_recognizer/tmp.html"
# with open(F_PATH, 'r') as f:
#   html_text = f.read()

html_text = ''
url = 'https://vitalik.ca/general/2022/09/17/layer_3.html'
with HTMLSession() as session:
  r = session.get(url)
  html_text = r.text

recognizer = Recognizer()
candidate = recognizer.getContent(html_text)
print('[Result (length of %s): ]' % len(candidate[4]), candidate[4])

with open('article_body_recognizer/result.txt', 'w') as f:
  html_text = f.write(candidate[4])
```
```python
# html_text
```
