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
os.chdir('/workspace/upwork/martien_brouver/mylapi/scraping/')

import re
import jsonlines
import json
import base64
# from stop_words import get_stop_words

from scraping.utils import etree_parse_xml

def base64StrDecode(encoded_str: str, scheme='utf-8') -> str:
  # >>> type(decoded_bytes)
  # <class 'bytes'>
  decoded_bytes = base64.b64decode(encoded_str)
  decoded_str = str(decoded_bytes, scheme)

  return decoded_str

```
```python
DATA_INDEX = 2
with jsonlines.open('parser/hierarchical_content_training_dataset.jl') as reader:
  dataset = list(reader)

  test = base64StrDecode(dataset[DATA_INDEX]['text'])
  print(test)

```
```python
from lxml import etree
tree = etree_parse_xml(test)

print('[Original]')
for el in tree.iter():
  print(str(etree.tostring(el), encoding='utf-8'))
```
```python
# stop_words = get_stop_words('en')
stop_words_path = 'parser/nlp_stop_words.json'
with open(stop_words_path, 'r', encoding='utf8', errors='ignore') as json_file:
    stop_words = json.load(json_file)
    stop_words = stop_words['stop_words']
    print('stop words: ', stop_words)
```
```python
tree = etree_parse_xml(test)
start_punctuation = re.compile('^(\.|\,|\:|\;|\'|\‘|\"|\“|\(|\!|\?|\|(\'s)|(\’s)|(\'s)|\*|\[|\-|\—|\–|\#)+')
stop_punctuation_p = re.compile('(\.|\,|\:|\;|\'|\’|\"|\”|\)|\!|\?|\|(\'s)|(\’s)|(\'s)|\*|\]|\-|\—|\–|\#)+$')

for el in tree.iter():
  _text = el.text
  if _text is None:
    continue
  splited = _text.split()

  filtered = []
  for w in splited:
    _w = w.lower()
    _w = stop_punctuation_p.sub('', _w)
    _w = start_punctuation.sub('', _w)
    if _w not in stop_words and str() != _w:
      print(w, ' ->' ,_w)
      filtered.append(_w)

  el.text = ' '.join(filtered)

```
```python
print('[Stop words removed]')
for el in tree.iter():
  print(el.text if el.text is not None else '')
```
