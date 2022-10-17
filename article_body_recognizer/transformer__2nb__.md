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
F_PATH = "/workspace/upwork/martien_brouver/mylapi/scraping/parser/tmp.html"
STORING_PATH = '/workspace/upwork/martien_brouver/mylapi/scraping/parser/output.html'
```
```python
import os
os.chdir('/workspace/upwork/martien_brouver/mylapi/scraping/')
```
```python
import re
import html
import math
from lxml import etree
import base64


from scraping.transformer import LxmlTree
from scraping.transformer import transform_top_level_nodes_to_sequence
from scraping.utils import base64StrDecode
from scraping.utils import load_text

```
```python
data = {"url":"https%3A%2F%2Fwww.abc.net.au%2Fabckids%2Fabc-kids-listen-app%2F11131286","text":"PGJvZHkgPiA8YSA+U2tpcCB0byBtYWluIGNvbnRlbnQ8L2E+IDxkaXYgPiA8ZGl2ID4gPGRpdiA+IDxkaXYgPiA8aGVhZGVyID4gPHNlY3Rpb24gPiA8ZGl2ID4gPGRpdiA+IDxkaXYgPiA8aDE+IDxhID4gQUJDIEtpZHMgPC9hPiA8L2gxPiA8L2Rpdj4gPC9kaXY+IDwvZGl2PiA8L3NlY3Rpb24+IDwvaGVhZGVyPjxkaXYgPiA8YSA+IE1lbnUgPC9hPiA8L2Rpdj4gPGRpdiA+ICA8L2Rpdj4gPC9kaXY+IDwvZGl2PiA8ZGl2ID4gPGFydGljbGUgPiA8ZGl2ID4gPGRpdiA+IDxkaXYgPiA8aDEgPiBBQkMgS2lkcyBsaXN0ZW4gYXBwIDwvaDE+IDwvZGl2PiA8L2Rpdj4gPGRpdiA+IDxkaXYgPiA8ZGl2ID4gPGRpdiA+IDxkaXYgPiA8ZGl2ID4gPGRpdiA+IDxkaXYgPiA8ZGl2ID4gPGRpdiA+IDxkaXYgPkltYWdlOiA8L2Rpdj4gPC9kaXY+IDwvZGl2PiA8L2Rpdj4gPC9kaXY+IDwvZGl2PiA8L2Rpdj4gPC9kaXY+IDwvZGl2PiA8L2Rpdj4gPC9kaXY+IDxkaXYgPiA8ZGl2ID4gPGRpdiA+IDxwPkFCQyBLaWRzIGxpc3RlbiBpcyBhIGZyZWUgZGVkaWNhdGVkIGtpZHMgYXBwIGFuZCBkaWdpdGFsIHJhZGlvIHN0YXRpb24gcHJvdmlkaW5nIHByZXNjaG9vbC1hZ2VkIGNoaWxkcmVuIGFuZCB0aGVpciBmYW1pbGllcyB3aXRoIGEgc2FmZSBzcGFjZSB0byBoZWFyIHRydXN0ZWQsIGVkdWNhdGlvbmFsIGFuZCBlbnRlcnRhaW5pbmcgYXVkaW8gcHJvZ3JhbXMgdGhhdCBmZWF0dXJlIG11c2ljIGFuZCBzdG9yaWVzIGZyb20gdGhlaXIgZmF2b3VyaXRlIEFCQyBLaWRzIGZyaWVuZHMuIEF2YWlsYWJsZSBvbiBpUGhvbmUsIGlQYWQgYW5kIEFuZHJvaWTihKIgcGhvbmVzIGFuZCB0YWJsZXRzLiA8L3A+PGRpdiA+IDxhID4gRG93bmxvYWQgZnJlZSBmcm9tIHRoZSBBcHAgU3RvcmUgPC9hPiA8L2Rpdj48ZGl2ID4gPGEgPiBEb3dubG9hZCBmcmVlIG9uIEdvb2dsZSBQbGF5IDwvYT4gPC9kaXY+PGgzPldoYXQncyBzbyBncmVhdCBhYm91dCB0aGUgQUJDIEtpZHMgbGlzdGVuIGFwcD88L2gzPjx1bD48bGk+UHJvdmlkZXMgY2hpbGRyZW4gYWdlZCAwLTUgYW5kIHRoZWlyIGZhbWlsaWVzIHdpdGggYSB3YXkgdG8gYWNjZXNzIHRoZSBtdXNpYyBhbmQgc3RvcmllcyBmcm9tIHRoZSBBQkMgdGhhdCB0aGV5IGxvdmUgaW4gYSB0cnVzdGVkIG9ubGluZSBlbnZpcm9ubWVudC4gQUJDIEtpZHMgbGlzdGVuIGNhcmVzIGFib3V0IHByb3ZpZGluZyBBdXN0cmFsaWFuIGZhbWlsaWVzIHdpdGggYSBzYWZlIHNwYWNlIGZvciB0aGVpciBjaGlsZHJlbiB0byBhY2Nlc3MgZWR1Y2F0aW9uYWwgYW5kIGVudGVydGFpbmluZyBhdWRpbyBjb250ZW50LjwvbGk+PGxpPllvdSBhbmQgeW91ciBjaGlsZCBjYW4gbGlzdGVuIHRvIG91ciByYWRpbyBzdGF0aW9uLCB3aGljaCBoYXMgYmVlbiBkZXNpZ25lZCB0byBzdWl0IHRoZSBjaGFuZ2luZyBuZWVkcyBvZiBhIGNoaWxkJ3MgZGF5IGFuZCBuaWdodCwgb3Igc2ltcGx5IGNob29zZSBhIHByb2dyYW0gb24gZGVtYW5kIHRoYXQgYmVzdCBzdWl0cyB5b3VyIG5lZWRzIGF0IHRoYXQgbW9tZW50LjwvbGk+PGxpPk91ciBwcm9ncmFtcyBhcmUgaW5zcGlyZWQgYnkgdGhlIEVhcmx5IFllYXJzIExlYXJuaW5nIEZyYW1ld29yayBvZiBBdXN0cmFsaWEsIHdoaWNoIGVuY291cmFnZXMgY2hpbGRyZW4gdG8gbGVhcm4gdGhyb3VnaCBwbGF5LiBUaGUgaGlnaCBxdWFsaXR5IGF1ZGlvIHByb2dyYW1zIGdpdmUgcHJlc2Nob29sZXJzIGEgc3BhY2Ugd2hlcmUgdGhleSBjYW4gZ2V0IHRoZWlyIGJvZGllcyBtb3ZpbmcgYW5kIGJyYWlucyB3b3JraW5nLiBFeHBsb3JlLCBsZWFybiwgYW5kIHBsYXksIGFuZCBsYXRlciB3aW5kIGRvd24sIHJlc3QsIGFuZCBzbGVlcC48L2xpPjxsaT5FcGlzb2RlcyBjYW4gYmUgZG93bmxvYWRlZCBhbmQgd2lsbCB0aGVuIGJlIGF2YWlsYWJsZSBmb3Igc2V2ZW4gZGF5cyBpbiB0aGUgYXBwIGJlZm9yZSBleHBpcmluZywgYWxsb3dpbmcgZm9yIGxpc3RlbmluZyB3aXRob3V0IGFuIGludGVybmV0IGNvbm5lY3Rpb24uPC9saT48bGk+SXQncyBmcmVlIGFuZCBjb21tZXJjaWFsIGZyZWUuPC9saT48L3VsPjxoMz5PdGhlciB3YXlzIHRvIGxpc3RlbiB0byBBQkMgS2lkcyBsaXN0ZW48L2gzPjxwPkluIGFkZGl0aW9uIHRvIHRoZSBBQkMgS2lkcyBsaXN0ZW4gYXBwLCB5b3UgY2FuIGxpc3RlbiB0byBBQkMgS2lkcyBsaXN0ZW4gb24gYXZhaWxhYmxlIERBQisgcmFkaW9zIChyZXNjYW4gcmVjZWl2ZXJzIGZvciBBQkMgS2lkcyBsaXN0ZW4pIG9yIHN0cmVhbSBpdCBhdCBhYmMubmV0LmF1L2tpZHNsaXN0ZW4uPC9wPjxoMz5TdXBwb3J0PC9oMz48cD5Gb3IgYW55IHN1cHBvcnQgcXVlcmllcyBhYm91dCB0aGUgQUJDIEtpZHMgbGlzdGVuIGFwcCBwbGVhc2UgdXNlIHRoZSA8YSA+QUJDIEtpZHMgbGlzdGVuIENvbnRhY3QgVXMgZm9ybTwvYT4uPC9wPjxwPkJlZm9yZSBjb21wbGV0aW5nIHRoZSBjb250YWN0IGZvcm0gcGxlYXNlIGVuc3VyZSB0aGF0IHlvdXIgcXVlcnkgaXMgbm90IGFscmVhZHkgYWRkcmVzc2VkIGluIHRoZSBGQVFzIGluIHRoZSBhcHAgaXRzZWxmLjwvcD48cD5UaGlzIGFwcCBpcyBkZXNpZ25lZCBmb3IgcGhvbmVzIHJ1bm5pbmcgaU9TIDkuMCBhbmQgQW5kcm9pZCA1LjAgYW5kIGFib3ZlLjwvcD48ZGl2ID4gPGEgPiBEb3dubG9hZCBmcmVlIGZyb20gdGhlIEFwcCBTdG9yZSA8L2E+IDwvZGl2PjxkaXYgPiA8YSA+IERvd25sb2FkIGZyZWUgb24gR29vZ2xlIFBsYXkgPC9hPiA8L2Rpdj48cD5BbmRyb2lkIGlzIGEgdHJhZGVtYXJrIG9mIEdvb2dsZSBJbmMuPC9wPiA8L2Rpdj4gPGRpdiA+IDxoMj5SZWxhdGVkPC9oMj4gPHVsPiA8bGk+PGRpdiA+IDxkaXYgPiA8ZGl2ID4gPGgzPiA8YSA+IEFCQyBLaWRzIGFwcCA8L2E+PC9oMz4gPGRpdiA+IDxwPlRoZSBmcmVlIEFCQyBLaWRzIGFwcCBpcyBkZXNpZ25lZCBmb3Igb3VyIHByZXNjaG9vbCBhbmQgeW91bmdlciBzY2hvb2wtYWdlZCB2aWV3ZXJzLiBXYXRjaCBhbGwgeW91ciBBQkMgS2lkcyBmYXZvdXJpdGVzIGFueXRpbWUgYW5kIGFueXdoZXJlLjwvcD4gPC9kaXY+IDwvZGl2PiA8L2Rpdj4gPC9kaXY+PC9saT4gPGxpPjxkaXYgPiA8ZGl2ID4gPGRpdiA+IDxoMz4gPGEgPiBBQkMgS2lkcyBQbGF5IDwvYT48L2gzPiA8ZGl2ID4gPHA+QUJDIEtpZHMgUGxheSBmZWF0dXJlcyBnYW1lcyBmcm9tIEFCQyBLaWRzIGZhdm91cml0ZXMgR2lnZ2xlIEhvb3QsIEJhbmFuYXMgaW4gUHlqYW1hcyBhbmQgUGxheSBTY2hvb2whPC9wPiA8L2Rpdj4gPC9kaXY+IDwvZGl2PiA8L2Rpdj48L2xpPiA8bGk+PGRpdiA+IDxkaXYgPiA8ZGl2ID4gPGgzPiA8YSA+IEFCQyBNRSBBcHAgPC9hPjwvaDM+IDxkaXYgPiA8cD5JcyB5b3VyIHByaW1hcnkgc2Nob29sIGFnZSBjaGlsZCByZWFkeSB0byBtb3ZlIGZyb20gQUJDIEtpZHMgdG8gQUJDIE1FPyBXaHkgbm90IGRvd25sb2FkIHRoZSBBQkMgTUUgYXBwITwvcD4gPC9kaXY+IDwvZGl2PiA8L2Rpdj4gPC9kaXY+PC9saT4gPGxpPjxkaXYgPiA8ZGl2ID4gPGRpdiA+IDxoMz4gPGEgPiBQbGF5IFNjaG9vbCBQbGF5IFRpbWUgPC9hPjwvaDM+IDxkaXYgPiA8cD5QbGF5IFRpbWUgZW5jb3VyYWdlcyBpbWFnaW5hdGlvbiBhbmQgY3JlYXRpdml0eSB0aHJvdWdoIG9wZW4tZW5kZWQgcGxheSwgZXhwbG9yYXRvcnkgdGFza3MgYW5kIGVuZ2FnZW1lbnQgd2l0aCBQbGF5IFNjaG9vbCBjaGFyYWN0ZXIncyB3b3JsZC48L3A+IDwvZGl2PiA8L2Rpdj4gPC9kaXY+IDwvZGl2PjwvbGk+IDxsaT48ZGl2ID4gPGRpdiA+IDxkaXYgPiA8aDM+IDxhID4gUGxheSBTY2hvb2wgQXJ0IE1ha2VyIDwvYT48L2gzPiA8ZGl2ID4gPHA+R2V0IGNyZWF0aXZlIHdpdGggSHVtcHR5LCBKZW1pbWEsIEJpZyBUZWQgYW5kIExpdHRsZSBUZWQhIE1ha2UgcGljdHVyZXMgYW5kIGFuaW1hdGVkIG1vdmllcyB1c2luZyBQbGF5IFNjaG9vbCB0b3lzIGFuZCBjcmFmdCBpdGVtcy48L3A+IDwvZGl2PiA8L2Rpdj4gPC9kaXY+IDwvZGl2PjwvbGk+IDxsaT48ZGl2ID4gPGRpdiA+IDxkaXYgPiA8aDM+IDxhID4gQUJDIFJlYWRpbmcgRWdncyA8L2E+PC9oMz4gPGRpdiA+IDxwPkFCQyBSZWFkaW5nIEVnZ3MgbWFrZXMgbGVhcm5pbmcgdG8gcmVhZCBmdW4gd2l0aCBzZWxmLXBhY2VkIGxlc3NvbnMsIGludGVyYWN0aXZlIGdhbWVzLCBjb2xvdXJmdWwgYW5pbWF0aW9ucyBhbmQgZXhjaXRpbmcgcmV3YXJkcy48L3A+IDwvZGl2PiA8L2Rpdj4gPC9kaXY+IDwvZGl2PjwvbGk+IDwvdWw+IDwvZGl2PiA8L2Rpdj4gPC9kaXY+IDwvZGl2PiA8L2FydGljbGU+PC9kaXY+IDwvZGl2PiA8L2Rpdj4gIDwvYm9keT4=","title":"22","article":"36"}

def to_distributed_categorial(_distributed):
  _arr = [0.0] * 501

  for ind, prob in _distributed:
    _arr[ind] = prob

  return _arr

def sum_distributed_probability(_distributed):
  sum = 0
  for ind, p in _distributed:
    sum = sum + p

  return sum

def verify_node_annotation(_data):
  title_node_ind = int(_data['title'])
  article_node_ind = int(_data['article'])

  _html_text = base64StrDecode(_data['text'])
  _tree = LxmlTree(_html_text)

  title_node = _tree.get_node_by_number(title_node_ind)
  title = _tree.node2string(title_node)
  print('[TITLE] ', title)

  article_node = _tree.get_node_by_number(article_node_ind)
  article = _tree.node2string(article_node)
  print('[CONTENT] ', article)

verify_node_annotation(data)

```
```python
title_node_ind = int(data['title'])
article_node_ind = int(data['article'])

html_text = base64StrDecode(data['text'])
tree = LxmlTree(html_text)

article_distributed_probability = tree.calc_distributed_probability_of_truth(target_number=article_node_ind)
print('distributed_probability(article): ', article_distributed_probability)
print('total(article): ', sum_distributed_probability(article_distributed_probability))

title_distributed_probability = tree.calc_distributed_probability_of_truth(target_number=title_node_ind, percentage_of_itinerary=0.5)
print('distributed_probability(title): ', title_distributed_probability)
print('total(title): ', sum_distributed_probability(title_distributed_probability))

```
```python
article_label = to_distributed_categorial(article_distributed_probability)
print(article_label)
```
```python
title_label = to_distributed_categorial(title_distributed_probability)
print(title_label)
```
```python
# TEST CASES
# 1. Check if nodes are assigned number consistently
# 2. Check if data does not comply the threshold rule: a) There exists a node has level higher than
# threshold. b) There exists node has child and has level smaller than threshold
# 3. Check if there exist paths of two nodes that are overlaping (one is child of other)
# 4. Check if represented data convey comprehensively html structure.

#    <div id="1">
#      <div id="2"> <!-- level 2 -->
#        <div id="3"> <!-- level 3 -->
#          Text3
#        </div>
#        <div id="4"> <!-- level 3 -->
#          <div id="5">Text5</div>
#          <div id="6">Text6</div>
#        </div>
#      </div>
#      <div id="7"> <!-- level 2 -->
#        <div id="8"> <!--level 3 -->
#          Text8
#        </div>
#        <div id="9"> <!-- level 3 -->
#          <div id="10">
#            <div id="11">Text11</div>
#          </div>
#        </div>
#      </div>
#      <div id="12"> <!-- level2, this node is very special coz we going to test if thresholed is 3, #2 is not going to be taken (overlaping) but #12 will be -->
#        Text12
#      </div>
#    </div>

test_text = """
    <div id="1"><div id="2"><div id="3"> Text3 </div><div id="4"><div id="5">Text5</div><div id="6">Text6</div></div></div><div id="7"><div id="8"> Text8 </div><div id="9"><div id="10"><div id="11">Text11</div></div></div></div><div id="12"> Text12 </div></div>
  """

sequential_code = transform_top_level_nodes_to_sequence(html_text=test_text)
nodes = sequential_code.split('<#>')
print('Number of Nodes', len(nodes))
for item in nodes:
  print('Text length: ', len(item))
  print(item)
  print('')
```
```python
# html_text = load_text(F_PATH)
html_text = data['text']
html_text = base64StrDecode(html_text)

sequential_code = transform_top_level_nodes_to_sequence(html_text=html_text)
nodes = sequential_code.split('<#>')
print('Number of Nodes', len(nodes))
for item in nodes:
  print('Text length: ', len(item))
  print(item)
  print('')
```

```python
data = """
    <div id="1">
      <div id="2">
        <div id="3">Text3</div>
        <div id="4">Text4</div>
        <div id="5">Text5</div>
      </div>
      <div id="6">
        <div id="7">Text7</div>
        <div id="8">
          <div id="9">Text9</div>
          <div id="10">Text10</div>
        </div>
      </div>
    </div>
  """

tree = etree.XML(data)

counter = 1
for el in tree.iter():
  el.set('title', str(counter))
  counter += 1

# print(str(etree.tostring(tree)))
```

