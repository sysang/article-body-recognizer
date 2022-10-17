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

os.chdir('/workspace/upwork/martien_brouver/mylapi/scraping')
WORKING_DIR = '/workspace/upwork/martien_brouver/mylapi/scraping'
```
```python
import datetime
import jsonlines
import html
import math

from pathlib import Path
from os.path import exists
from importlib import reload

from lxml import etree

from scraping.utils import load_text
from scraping.transformer import get_nodes_indexed
```
```python
from scraping import utils
from scraping import transformer

reload(utils)
filter_html = utils.filter_html
base64StrEncode = utils.base64StrEncode
base64StrDecode = utils.base64StrDecode

reload(transformer)
LxmlTree = transformer.LxmlTree

def store_text_to_file(f_path, body_text, builder_body_text, url):

  js_code = "<script>" + load_text('parser/dataset_builder.js')+ "</script> "
  css_code = "<style tyle='text/css'>" + load_text('parser/dataset_builder.css')+ "</style> "

  try:
    f = open(f_path, 'w')
    orig_url = f"<script> var url = encodeURIComponent('{url}');</script>\n"
    orig_body_text = f"<script> var body_html = '{body_text}'</script>\n"
    html_page = f"""<!DOCTYPE html>\n<html>\n<head>\n
      <meta http-equiv="Content-Type" content="text/html; charset=utf-8">\n
        {orig_url}
        {orig_body_text}
      </head>
        {builder_body_text}
        {css_code}
        {js_code}
      </html>"""
    f.write(html_page)
  except:
    pass

def jsonline_to_html_file(relative_file_path, containing_dir):
    f_path = '/workspace/upwork/martien_brouver/mylapi/scraping/' + relative_file_path
    with jsonlines.open(f_path) as reader:
        for obj in reader:
            f_name = obj['url'].split('//')[-1]
            f_name = f_name.replace('/', '-')
            f_name = "{}/parser/tmp/{}/{}.html".format(WORKING_DIR, containing_dir, f_name)

            if os.path.isfile(f_name):
              continue

            try:
              _text = filter_html(obj['original'])
              if _text is None:
                continue
              encoded_text = base64StrEncode(_text)
              builder_text = get_nodes_indexed(_text)
              store_text_to_file(f_name, body_text=encoded_text, builder_body_text=builder_text, url=obj['url'])
            except Exception:
              print('[DEBUG] Error while processing: {}'.format(obj['url']))
              # raise Exception()


def build_target_uri(start_number, file_quanity):
  _sources = {}
  groups_number = math.ceil(file_quanity / 10)
  file_number = 0
  for folder_number in range(1, groups_number + 1):
    folder_name = 'topm_s{}_x50_n{}'.format(start_number, folder_number * 10)
    _sources[folder_name] = []
    for j in range(10):
      file_number += 1
      if file_number > file_quanity:
        break
      file_name = 'raw_html_topm_sites_s{}_x50_{:03d}.jl'.format(start_number, file_number)
      _sources[folder_name].append(file_name)

  return _sources

# Show sources for verify
def verify_sources(_sources):
  for folder_name, files in _sources.items():
    print(folder_name, ' :')
    for file in files:
      print('   ', file)

def build_dataset(_sources):
  for group, files  in _sources.items():
    for f in files:
      tracking = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
      print(f'\n[INFO] {tracking} - Working on {f}\n')
      f_path = 'warehouse/' + f
      jsonline_to_html_file(f_path, group)

targets = build_target_uri(950, 31)

verify_sources(targets)
build_dataset(targets)

```
```python
# TEST INDEXING HIERARCHICAL NODES
#    <div id="1">
#      <div id="2"> <!-- level 2 -->
#        <div id="3"> <!-- level 3 -->
#          Text3
#        </div>
#        <div id="4"> <!-- level 3 -->
#          <div id="5">Text5</div> <!-- level 4 -->
#          <div id="6">Text6</div> <!-- level 4 -->
#        </div>
#      </div>
#      <div id="7"> <!-- level 2 -->
#        <div id="8"> <!--level 3 -->
#          Text8
#        </div>
#        <div id="9"> <!-- level 3 -->
#          <div id="10"> <!-- level 4 -->
#            <div id="11">Text11</div> <!-- level 5 -->
#          </div> #        </div>
#      </div>
#      <div id="12"> <!-- level2 -->
#        Text1
#      </div>
#    </div>

test_text = """
    <div id="1"><div id="2"><div id="3"> Text3 </div><div id="4"><div id="5">Text5</div><div id="6">Text6</div></div></div><div id="7"><div id="8"> Text8 </div><div id="9"><div id="10"><div id="11">Text11</div></div></div></div><div id="12"> Text12 </div></div>
  """

result = get_nodes_indexed(test_text)
# print(result)
```
```python
F_PATH = 'scraping/etc/output.html'
html_text = load_text(F_PATH)

# tree = etree_parse_xml(html_text)
```
```python

def load_text(f_path):
  with open(f_path, 'r') as f:
    text = f.read()

  return text

F_PATH = "warehouse/tmp.html"

# Test base65encode/base64decode
# html_text = load_text(F_PATH)
# en = base65StrEncode(html_text)
# print(len(en))
# print(en)

# de = base65StrDecode(en)
# print(len(de))
# print(de)
```





