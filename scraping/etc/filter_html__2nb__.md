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
import pprint
import re
import os
os.chdir('/workspace/upwork/martien_brouver/mylapi/scraping')

```
```python
detect_nested_a_in_span = re.compile('(\<span[^\>]*\>.*?)\<a[^\>]*\>([^\<]+)\<\/a\>(.*?\<\/span\>)')
repl = lambda matchObj: matchObj.groups()[0] + matchObj.groups()[1] + matchObj.groups()[2]

_str = '<span ><a >Contribute</a></span>'
output = detect_nested_a_in_span.sub(repl, _str)
print('01:1 -> ', output == '<span >Contribute</span>', ' -> ', output)

_str = '<span ><a ><span ><a >Contribute</a></span></a></span>'
output = detect_nested_a_in_span.sub(repl, _str)
print('02:1 -> ', output == '<span ><a ><span >Contribute</span></a></span>', ' -> ', output)

_str = '<span > <a >Contribute</a> <a > links </a> </span>'
output = detect_nested_a_in_span.sub(repl, _str)
print('03:1 -> ', output == '<span > Contribute <a > links </a> </span>', ' -> ', output)

_str = output
output = detect_nested_a_in_span.sub(repl, _str)
print('03:2 -> ', output == '<span > Contribute  links  </span>', ' -> ', output)

_str = '<span > <a ><span >Contribute </span></a></span>'
output = detect_nested_a_in_span.sub(repl, _str)
print('04:1 -> ', output == '<span > <a ><span >Contribute </span></a></span>', ' -> ', output) #  remains the same
```

```python
detect_nested_span_in_a = re.compile('(\<a[^\>]*\>.*?)\<span[^\>]*\>([^\<]+)\<\/span\>(.*?\<\/a\>)')
repl = lambda matchObj: matchObj.groups()[0] + matchObj.groups()[1] + matchObj.groups()[2]

_str = '<a > <span > Contribute </span> </a>'
output = detect_nested_span_in_a.sub(repl, _str)
print('01:1 -> ', output == '<a >  Contribute  </a>', ' -> ', output)

_str = '<a > <span >Contribute</span> <span > links </span> </a>'
output = detect_nested_span_in_a.sub(repl, _str)
print('02:1 -> ', output == '<a > Contribute <span > links </span> </a>', ' -> ', output)

_str = output
output = detect_nested_span_in_a.sub(repl, _str)
print('02:2 -> ', output == '<a > Contribute  links  </a>', ' -> ', output)

```

```python
detect_nested_span_in_span = re.compile('(\<span[^\>]*\>.*?)\<span[^\>]*\>([^\<]+)\<\/span\>(.*?\<\/span\>)')
repl = lambda matchObj: matchObj.groups()[0] + matchObj.groups()[1] + matchObj.groups()[2]

_str = '<span > <span >Contribute</span> </span>'
output = detect_nested_span_in_span.sub(repl, _str)
print('01:1 -> ', output == '<span > Contribute </span>', ' -> ', output)

_str = '<span > <span >Contribute</span> <span > links </span> </span>'
output = detect_nested_span_in_span.sub(repl, _str)
print('02:1 -> ', output == '<span > Contribute <span > links </span> </span>', ' -> ', output)

_str = output
output = detect_nested_span_in_span.sub(repl, _str)
print('02:2 -> ', output == '<span > Contribute  links  </span>', ' -> ', output)

```

```python
detect_nested_nav_in_nav = re.compile('(\<nav[^\>]*\>.*?)\<nav[^\>]*\>(.*?)\<\/nav\>(.*?\<\/nav\>)')
repl = lambda matchObj: matchObj.groups()[0] + matchObj.groups()[1] + matchObj.groups()[2]

_str = '<nav><h2>Main navigation</h2><nav><ul><li>link 1.1</li></ul></nav><nav><ul><li>link 2.1</li></ul></nav></nav>'
output = detect_nested_nav_in_nav.sub(repl, _str)
print('01:1 -> ', output == '<nav><h2>Main navigation</h2><ul><li>link 1.1</li></ul><nav><ul><li>link 2.1</li></ul></nav></nav>', ' -> ', output)

_str = output
output = detect_nested_nav_in_nav.sub(repl, _str)
print('01:2 -> ', output == '<nav><h2>Main navigation</h2><ul><li>link 1.1</li></ul><ul><li>link 2.1</li></ul></nav>', ' -> ', output)

_str = output
output = detect_nested_nav_in_nav.sub(repl, _str)
print('01:3 -> ', output == '<nav><h2>Main navigation</h2><ul><li>link 1.1</li></ul><ul><li>link 2.1</li></ul></nav>', ' -> ', output)
```

```python
from importlib import reload
from scraping import utils

reload(utils)

filter_html = utils.filter_html

def load_text(f_path):
  with open(f_path, 'r') as f:
    text = f.read()

  return text

def store_text_to_file(f_path, text):
  with open(f_path, 'w') as f:
    text = f.write(text)

def show_result(html):
  printer = pprint.PrettyPrinter()
  printer.pprint(html)

F_PATH = "/workspace/upwork/martien_brouver/mylapi/scraping/scraping/etc/tmp.html"
STORING_PATH = '/workspace/upwork/martien_brouver/mylapi/scraping/scraping/etc/output.html'

html_text = load_text(F_PATH)

output_text = filter_html(html_text)
show_result(output_text)

store_text_to_file(STORING_PATH, output_text)
```
