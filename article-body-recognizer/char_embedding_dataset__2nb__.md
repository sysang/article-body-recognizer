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
import re
import os
os.chdir('/workspace/upwork/martien_brouver/mylapi/scraping/')
```
```python
import re
import json
import jsonlines
import html

from pathlib import Path
from urllib import parse
from lxml import etree
from collections import deque

from scraping.transformer import LxmlTree
from scraping.system_specs import char_emb_training_specs
```
```python
MAX_LENGTH = char_emb_training_specs['MAX_LENGTH']
MIN_LENGTH = char_emb_training_specs['MIN_LENGTH']


def is_jsonline_file(fname):
  is_jlext_p = re.compile('\.jl$')

  return is_jlext_p.search(fname) is not None


def build_source(file_name_template, quantity):
  _sources = []
  for i in range(quantity):
    ind = i + 1
    _sources.append(file_name_template.format(ind))

  return _sources


def harvest_node_text_from_xml_tree(html_text):

  html_text = html_text
  tree = LxmlTree(html_text)
  node_list = []

  def traverse_down(node):
    _text = tree.node2string(node)
    children = node.getchildren()

    if not len(children) or len(_text) <= MAX_LENGTH:
      node_list.append(_text)
      return True

    for child in children:
      traverse_down(child)

  traverse_down(tree.root)

  return node_list

def extract_domain_url(url):
  p = re.compile('(https?\:\/\/)(www\.)?((?:[0-9a-z\-\_]+\.)+)(\w+\/?).*$')
  repl = lambda matchObj: matchObj.groups()[2] +  matchObj.groups()[3]

  return p.sub(repl, url)

def collect_websites(soure_path):
  path_obj = Path(soure_path)

  for file in path_obj.iterdir():
    if not is_jsonline_file(file.name):
      continue

    with open(file.absolute(), 'r') as reader:
      lines = reader.readlines()
      line = lines[0]
      obj = json.loads(line)
      url = obj['url']
      url = parse.unquote(obj['url'])
      domain_url = extract_domain_url(url)
      websites[domain_url] = True

# training_source_path = 'parser/dataset/training'
# validating_source_path = 'parser/dataset/validating'
# collect_websites(training_source_path)
# collect_websites(validating_source_path)
# print(websites)
```
```python

def collect_data(_sources, _websites, invalid_patterns, debug=True, _batch_size=500):

  def check_if_invalid_text(_text, invalid_patterns):
    common = [
      # REPEATING LIST OF LINKS
      re.compile(r'(:?(\s*<a>.+?<\/a>\s*){3,20})|(:?(\s*<li>\s*<a>.+?<\/a>\s*<\/li>\s*){2,20})'),

      # many wrapping div tags; zero-width-space code
      re.compile(r'(:?(<div>\s*){5,20}.*(<\/div>\s*){5,20})|(:?\u200b)'),

      # POTENTIAL REPEATING
      re.compile(r'(:?^<li>\s*<a>.+<\/a>\s*<\/li>$)|(:?^<div>\s*<a>.+<\/a>\s*<\/div>$)'),
      # <ul> <li>Perspective</li> </ul>
      re.compile(r'^(:?<ul>\s*<li>\s*\w+(\s\w+)?\s*<\/li>\s*<\/ul>)$'),
      #  <div><div><h5><div>Related</div></h5></div></div>
      re.compile(r'^(:?<div>\s*(<div>\s*)*<h5>\s*<div>\s*Related\s*</div>\s*</h5>(\s*</div>)*\s*</div>)$'),
      # <div><div><h2>Most Recent</h2></div></div>
      re.compile(r'^(:?<div>\s*<div>\s*<h\d>\s*Most\sRecent<\/h\d>\s*<\/div>\s*<\/div>)$'),
      # <div>January 5, 2022 | 8:13 PM GMT</div> --- <div>January 7, 2021 at 4:23 a.m. EST</div>
      re.compile('^(:?<div>.*\d+\:\d+\s(PM|AM|p\.?m\.?|a\.?m\.?)\s(GMT|EST)<\/div>)$'),
      # <p>Milwaukee; Saturday, 7:30 p.m. EST</p>
      re.compile('^(:?<p>.*\d+\:\d+\s(PM|AM|p\.?m\.?|a\.?m\.?)\s(GMT|EST)<\/p>)$'),
      # <p>Glendale, Arizona; Thursday, 9 p.m. EST</p>
      re.compile('^(:?<p>.*\d+\s(PM|AM|p\.?m\.?|a\.?m\.?)\s(GMT|EST)<\/p>)$'),
      # <i>—Michael Ordoña</i>
      re.compile(r'^<i>.*<\/i>$'),
      re.compile(r'^<p>\s*<i>.*<\/i>\s*<\/p>$'),
      #  <header><h2>Latest</h2></header>
      re.compile(r'^<header>\s*<h\d>\s*Latest\s*<\/h\d>\s*</header>$'),

      # SOCIAL NETWORKS
      re.compile(r'(:?<a>(\s*.acebook\s*)|(:?\s*.witter\s*)|(:?\s*.inkedIn\s*)</a>){2,20}|(:?on\s(.acebook|.witter|.nstagram))'),
      # <ul> <li> Facebook </li> <li> Twitter </li> <li> Show more sharing options </li> </ul>
      # <ul> <li> <a> Facebook </a> </li> <li> Twitter </li> <li> Show more sharing options </li> </ul>
      re.compile(r'^(:?<ul>\s*(<li>\s*(<a>\s*)?(Facebook|Twitter)(\s*<\/a>)?\s*<\/li>\s*)(<li>.*<\/li>\s*)?<\/ul>)$'),
      # Connect with me on LinkedIn --- Follow me on Twitter --- Friend me on Faceook
      re.compile(r'(:?Friend\sme\son\sFaceook)|(:?Connect\swith\sme\son\sLinkedIn)|(:?Follow\sme\son\sTwitter)'),

      # FREQUENT WORDS
      re.compile(r'(:?(S|s)ign\s(U|u)p)|(:?(S|s)igning\s(U|u)p)|(:?SPONSORED\:)|(:?<h\d>(A|a)ll\s(N|n)ews<\/h\d>)|(:?<div>(A|a)ll\s(N|n)ews<\/div>)|(:?RSS\s(F|f)eed)|(:?(A|a)ll\s(R|r)ights\s(R|r)eserved)|(:?(N|n)ewsletters)'),
      re.compile(r'(:?<a>(R|r)egister<\/a>)|(:?<a>(S|s)ubscribe<\/a>)'),
      re.compile(r'(:?Copyright)|(:?Newsletter)'),
      re.compile(r'(:?Please\stry\sagain\slater\.)'),
      re.compile(r'(:?(S|s)ign\s(I|i)n)|(:?(S|s)earch\sjobs)|(:?Most\sviewed)|(:?Choose\syour\splan)|(:?Read\sthe\sfull\sstory)'),
      re.compile(r'(:?(ADVERTISEMENT)|(SUBSCRIPTIONS?)|(REGISTRATION)|(RECOMMENDED)|(Subscribe))|(:?(R|r)ead\s(M|m)ore)|(:?(F|f)ollow\s(U|u)s)|(:?(T|t)erms\sof\s(S|s)ervice)|(:?Privacy\sPolicy)'),
      re.compile(r'(:?<h\d>Advertisement<\/h\d>)|(:?<div>Advertisement<\/div>)|(:?<a>\s*(L|l)oad\s(M|m)ore\s*<\/a>)'),
      re.compile(r'(:?<small>Advertisement<\/small>)'),
      re.compile(r'(:?<p>(T|t)el\:\s+\+\d\d.+<\/p>)|(:?<div>(T|t)el\:\s+\+\d\d.+<\/div>)|(:?<i>(T|t)el\:\s+\+\d\d.+<\/i>)'),
      re.compile(r'(:?<p>(E|e)mail\:.*?(\w+\.)+\w+<\/p>)|(:?<div>(E|e)mail\:.*?(\w+\.)+\w+<\/div>)|(:?<i>(E|e)mail\:.*?(\w+\.)+\w+<\/i>)'),
      re.compile(r'(:?adblocking)|(:?blocking\sad)|(:?Download)|(:?Podcast)|(:?(C|c)ontact\s(U|u)s)|(:?(A|a)d\s(C|c)hoices)'),
      # Click here to download  --- Try 1 month for --- You’re all set --- Thanks for reading!
      re.compile(r'(:?Click\shere\sto\sdownload)|(:?Try\s\d+\smonths?\sfor)|(:?You\’re\sall\sset)|(:?Thanks\sfor\sreading\!?)'),


      # POTENTIALLY MEANINGLESS
      re.compile(r'(:?<a>(https?\:\/\/)(\w+\.)+\w+\/.*<\/a>)|(:?<p>(https?\:\/\/)(\w+\.)+\w+\/.*<\/p>)'), # to match http link inside tag a, p
      re.compile(r'(:?<a>(www\.)?(\w+\.)+\w+\/?.*<\/a>)'), # to match http link inside tag a
      re.compile(r'(:?(<div>\s*){2,20}\s*(<\/div>\s*){2,20})'),
      re.compile(r'(:?(<a>\@.+<\/a>))'),
      re.compile(r'<template>.+?<\/template>'),

      # CHINESE CHARACTERS
      re.compile(r'[\u4e00-\u9fff]+') # (Many thanks) Credit to @xecgr, source: https://stackoverflow.com/questions/34587346/python-check-if-a-string-contains-chinese-character/34587468#34587468
    ]

    for p in (common + invalid_patterns):
      matched = p.search(_text)
      if matched:
        return matched

  def check_if_duplicated(_text):
    for item in historica_queue:
      if item == _text:
        return True

  def save_batch(_batch, _count_batch, _batch_number, _file_name, _is_forced=False):
    if _count_batch % BATCH_SIZE == 0 or _is_forced:
      texts = '\n'.join(_batch)
      with open(_file_name, 'w') as new_file:
        new_file.write(texts)
        print('[INFO] Write to file: ', _file_name)

        _batch_number += 1
        _count_batch = 0
        _batch = []

    return _batch, _count_batch, _batch_number


  BATCH_SIZE = 500

  if _batch_size:
    BATCH_SIZE = _batch_size
    print('[INFO] BATCH_SIZE: ', BATCH_SIZE)

  for src in _sources:
    print(src)

  corpus_source_path = 'warehouse'
  path_obj = Path(corpus_source_path)

  dataset_path = 'parser/charembdataset/{}_{:03d}.txt'
  count_batch = 0
  count_invalid = 0
  batch = []
  batch_number = 1
  prev_text_length = 0
  prev = ''
  historica_queue = deque(maxlen=33)

  for file in path_obj.iterdir():
    if not is_jsonline_file(file.name):
      continue

    if file.name not in _sources:
      continue

    with jsonlines.open(file.absolute()) as reader:
      line_number = 1
      for obj in reader:
        url = obj['url']
        url = parse.unquote(obj['url'])
        domain_url = extract_domain_url(url)
        line_number += 1

        if _websites.get(domain_url, None):
          html_text = obj['text']

          if not html_text:
            continue

          try:
            str_elements = harvest_node_text_from_xml_tree(html_text)
          except:
            print('[INFO] Error when working on: ', file.name)
            raise Exception(f"url: {url} - line number: {line_number}")

          # str_elements = harvest_node_text_from_xml_tree(html_text)

          tag_name = domain_url.replace('.', '').replace('/', '')
          file_name = dataset_path.format(tag_name, batch_number)

          for _text in str_elements:
            lgth = len(_text)
            _text = _text.strip()
            _text = html.unescape(_text)

            if MIN_LENGTH > lgth or lgth > MAX_LENGTH:
              continue

            if check_if_duplicated(_text):
              # if debug:
                # print('[DUPLICATED] ', _text)
              continue

            matched = check_if_invalid_text(_text, invalid_patterns)
            if matched:
              count_invalid += 1
              if debug:
                print(f'[{count_invalid} - INVALID] ', _text)
                # print('[MATCHED] ', matched.re.pattern)
              continue

            batch.append(_text)
            count_batch += 1
            prev_text_length = lgth
            prev = _text
            historica_queue.append(_text)

            batch, count_batch, batch_number = save_batch(_batch=batch, _count_batch=count_batch, _batch_number=batch_number, _file_name=file_name)

  if len(batch) > 0:
      batch, count_batch, batch_number = save_batch(_batch=batch, _count_batch=count_batch, _batch_number=batch_number, _file_name=file_name, _is_forced=True)
```
```python
def process_theguardiancom():
  websites = {
    'theguardian.com/': True,
  }

  sources = build_source('raw_html_the_guardian_{:03d}.jl', 34)

  invalid_patterns = [
    re.compile(r'(:?</h2>Commentin.*?isabled)|(:?<i>A.letter.to.*?<\/i>)|(:?family\@theguardian\.com)|(:?(<li>.+edition<\/li>)+)'),
    re.compile(r'^(<header>\s*)+|(<div>\s*)+<time>.+?\s\d\d\d\d\s*<\/time>(\s*<\/header>)+|(\s<\/div>)+$'),
    re.compile(r'^(<header>\s*)+|(<div>\s*)+<time>.*\d+\.\d+(am)|(pm)\s*(BST)|(GMT)\s*<\/time>(<\s*\/header>)+|(<\s*\/div>)+$'),
    re.compile(r'^<p>\s*<a>\s*<time>\d+\.\d+(am|pm)\s(GMT|BST)<\/time>\s*\d+:\d+\s*<\/a>\s*<\/p>$'),
    re.compile(r'^<p>Updated\s*<time>at\s\d+\.\d+(am|pm)\s*(GMT|BST)<\/time>\s*<\/p>$'),
    re.compile(r'(^(<div>\s*)+<time>\s*Published\:\s.+?<\/time>(\s*<\/div>)+$)'),
    re.compile(r'(^(<div>\s*)+<time>\s*Published\:.+?\s\d\d\:\d\d\sA|PM\s*<\/time>(\s*<\/div>)+$)'),
    re.compile(r'(:?<a>A|a\sletter\sto\s\.\.\.<\/a>)|(:?<p>.hotographer\:.+<a>.+<\/a><\/p>)|(:?<div>\s*Men\'s\stops\s*<div>.*<\/div>\s*\</div>)'),
    re.compile(r'(:?<a>The\sGuardian\s\-\sBack\sto\shome<\/a>)|(:?(his)|(her)\sTikTok\saccount)|(:?.+\.co\.uk)|(:?How\sto\slisten\sto\spodcasts\:)|(:?Comfort\sEating\swith)'),
    re.compile(r'<main><div><article><div\/><\/article><\/div><\/main>'),
    re.compile(r'(:?<div>\s*Blind\sdate\s*<div>)|(:?<h\d>Most\sviewed\sin\sUK\snews<\/h\d>)|(:?all\sprices\sinclude\staxes)|(:?More\slive\sreviews)'),
    re.compile(r'(:?<li>International\sedition<\/li>)|(:?<a>Reuse\sthis\scontent<\/a>)'),
  ]

  collect_data(sources, websites, invalid_patterns=invalid_patterns)

# process_theguardiancom()
```
```python
def process_cna():
  hierarchical_encoded_depth = 14
  websites = {
    'channelnewsasia.com/': True,
    'cnalifestyle.channelnewsasia.com/': True,
  }

  sources = build_source('raw_html_cna_asia_{:03d}.jl', 16)

  invalid_patterns = [
    re.compile(r'(:?<p>.+hassle\sto\sswitch\sbrowsers.+<\/p>)|(:?<a>\sBookmark\s<\/a>)|(:?BOOKMARK\sTHIS\:)|(:?mediacorp\.com\.sg)'),
    re.compile(r'(:?Special\sReports)|(:?Recent\sSearches)|(:?Main\snavigation)|(:?This\sbrowser\sis\sno\slonger\ssupported)|(?:upgrade\sto\sa\ssupported\sbrowser)'),
    re.compile(r'(:?Expand\sto\sread\sthe\sfull\sstory)|(:?Source\:\s)|(:?Related\sTopics)|(:?Also\sworth\sreading)|(:?RELATED\:)|(:?Related\sarticles\:)'),
    re.compile(r'(:?\(Updated\:\s\d\d\s\w\w\w\s20\d\d\s\d\d\:\d\dA|PM\))|(:?(<div>\s*)<article>\s*<\/article>(\s*<\/div>)+)'), # to match (Updated: 04 Jul 2020 07:26AM)
    re.compile(r'(:?<span>\s*<b>commentary<\/b>\s*<\/span>)|(:?Load\smore\sepisodes)|(:?Video\:\sYouTube)|(:?EXPLORE\sTHEIR\sSTORIES)|(:?NTUC\sIncome)'),
    re.compile(r'(:?Talking\sPoint)|(:?.nsure\syou\sset\syour\saccount\sto)|(:?Audio\sissues\sinherent\sfrom\ssource)|(:?You\sMay\sAlso\sLike)|(:?Content\sis\sloading\.\.\.)'),
    re.compile(r'(:?Other\slatest\scommentaries)'),
  ]

  collect_data(sources, websites, invalid_patterns=invalid_patterns)

# process_cna()
```
```python
def process_foxnews():
  websites = {
    'foxbusiness.com/': True,
    'foxnews.com/': True,
  }

  sources = build_source('raw_html_foxnews_{:03d}.jl', 10)

  invalid_patterns = [
    re.compile(r'(:?be\sreached\sat\s.+\@foxnews\.com)|(:?stories\syou\sneed\-to\-know)|(:?\w+\@foxnews\.com)|(:?\@pricegroup\.com)'),
    re.compile(r'(:?(provided|Sponsored)\sby\sCredible)|(:?20\d\d\sFOX\sNews\sNetwork\,\sLLC)|(:?Follow\s(him|her)\s(on|at))'),
    re.compile(r'(:?Hot\sTopics)|(:?<div><div><a>\sV|video<\/a><\/div><\/div>)|(:?<div><div><div>\s*V|video<\/div><\/div><\/div>)|(:?foxnews\.com)'),
    re.compile(r'(:?Watch\sTV)|(:?<h1>Markets<\/h1>)|(:?Closed\sMarket)|(:?Trade\sNow)|(:?Volume\:\s\d+)|(:?Close\sChg\sChg\s\%)'),
    re.compile(r'(:?<label>\s*next\s*<\/label>)|(:?<label>\s*prev\s*<\/label>)'),
    re.compile(r'(:?<p>\s*Published\s.*?<\/p>)|(:?CLICK\sHERE\sTO)'),
  ]

  collect_data(sources, websites, invalid_patterns=invalid_patterns)

# process_foxnews()
```
```python
def process_newsweek():
  websites = {
    'newsweek.com/': True,
  }

  sources = build_source('raw_html_newsweek_{:03d}.jl', 16)

  invalid_patterns = [
    re.compile(r'(:?\wewsweek\.com)|(:?<a>Newsweek<\/a>)|(:?<div>\s*<header>\s*<h1>.*?<\/h1>\s*<\/header>\s*<\/div>)'),
    re.compile(r'(:?<div>\s*By.+?P|AM\sEDT<\/div>)|(:?<div>\s*By.+?P|AM\sEST<\/div>)'),
    re.compile(r'(:?<div>\s*On.+?P|AM\sEDT<\/div>)|(:?<div>\s*On.+?P|AM\sEST<\/div>)'),
    re.compile(r'(:?Request\sReprint\sLicensing)|(:?Choose\syour\ssubscription)|(:?The\svideo\scan\sbe\sviewed)'),
  ]

  collect_data(sources, websites, invalid_patterns=invalid_patterns)

# process_newsweek()
```
```python
def process_politico():
  websites = {
    'politico.com/': True,
  }

  sources = build_source('raw_html_politico_{:03d}.jl', 7)

  invalid_patterns = [
    re.compile(r'(:?White\sHouse\sCouncil)|(:?NaN\sdistricts\sleft)|(:?www\.politico\.com)'),
    re.compile(r'(:?<header>\s*<h2>.+?<\/h2>\s*<\/header>)|(:?\[email.protected\])'),
    re.compile(r'(:?<div><div><div>)|(:?<li>\s*<article>\s*<div>\s*<header>.*?<\/header>\s*<\/div>\s*<\/article>\s*<\/li>)'),
    re.compile(r'^<div>Last\supdated\s.+\s(a\.m\.|p\.m\.)\sEST</div>$'),
    re.compile(r'(:?<a>Take\sa\slook\sinside<\s/a>)'),
  ]

  collect_data(sources, websites, invalid_patterns=invalid_patterns)

# process_politico()
```
```python
def process_thehill():
  websites = {
    'thehill.com/': True,
  }

  sources = build_source('raw_html_topm_sites_s700_x50_{:03d}.jl', 34)

  invalid_patterns = [
    re.compile(r'(:?<a>Skip\sto\smain\scontent<\/a>)|(:?<a>Coronavirus\sReport</a>)|(:?Follow\sthe\sHill)|(:?\@?thehill\.com)|(:?HILL\.TV)'),
    re.compile(r'(:?<a>Sunday\sTalk\sShows<\/a>)|(:?<a>Submit\sa\sjob</a>)|(:?<a>Become\sa\sContributor</a>)|(:?Photos\sof\sthe Week)'),
    re.compile(r'(:?<li>Page\s*\d+<\/li>)'),
  ]

  collect_data(sources, websites, invalid_patterns=invalid_patterns)

# process_thehill()
```
```python
def process_telegraph():
  websites = {
    'telegraph.co.uk/': True,
  }

  sources = build_source('raw_html_telegraph_{:03d}.jl', 21)

  invalid_patterns = [
    # #8220; --- #8221; Fallback processing for special characters when html.unescape can not process properly
    re.compile(r'(:?\#\d+\;)'),

    # <header> <h2> Opinion </h2> </header> --- <div> <div> <a> <h2>News</h2> </a> </div> </div>
    re.compile(r'^(:?<header>\s*<h2>\s*.*\s*<\/h2>\s*<\/header>)$|^(:?<div> <div>\s*<a>\s*<h2\s*.*\s*</h2>\s*<\/a>\s*<\/div>\s*<\/div>)$'),

    # <div> <div> <h2>Life Style</h2> </div> </div>
    re.compile(r'^(:?<div>\s*<div>\s*<h2>\s*.*\s*</h2>\s*</div>\s*</div>)$'),

    re.compile(r'(:?We\'ve\snoticed\syou\'re\sadblocking)|(:?award\-winning\sjournalism)|(:?urge\syou\sto\sturn\soff\syour\sad\sblocker)'),
    re.compile(r'(:?<a>adblocking\sinstructions<\/a>)|(:?Thank\syou\sfor\syour\ssupport)|(:?You\sare\susing\san\soutdated\sbrowser\.)'),

    # <div> <div> By Jake Goodwill <time>5 Jan 2022, 7:00am</time> </div> </div>
    re.compile(r'^(:?<div>\s*<div>.*<time>.*\d+\:\d+(am|pm)<\/time>\s*<\/div>\s*<\/div>)$'),

    # <div> <div> <span><b>10</b>%</span> Offer </div> </div> --- <div> 41047 Used recently </div> ---  More articles
    re.compile(r'(:?<div>\s*<div>\s*<span><b>\d+<\/b>\%<\/span>\s*Offer\s*<\/div>\s*<\/div>)|(:?\d+\sUsed\srecently)|(:?More\sarticles)'),
    # Have you considered... --- ££-££££ --- Answers at bottom of page . . . --- This week's question...
    re.compile(r'(:?Have you considered\.\.\.)|(:?£+\-£+)|(:?Answers\sat\sbottom\sof\spage\s*.\s*\.\s*\.)|(:?This\sweek\'s\squestion\.\.\.)'),
    # <li> <div> 28 discounts </div> </li>
    re.compile(r'(:?<li>\s*<div>\s*\d+\sdiscounts\s*</div>\s*</li>)'),

    #  <ul><li>Iceland</li><li>Austria</li><li>Bulgaria</li><li>Croatia</li><li>Finland</li><li>Norway</li><li>Slovenia</li><li>Turkey</li></ul>
    re.compile(r'^<ul>(\s*<li>\w+<\/li>\s*){2,20}<\/ul>$'),

    # <div> <div> 20% Coupon </div> </div> --- coupon code: 20%
    re.compile(r'(:?<div>\s*<div>\s*\d+%\s*.+<\/div>\s*<\/div>)|(:?coupon\scode\:\s\d+\%)'),

    # Further details Code -- Download the free app and get more Groupon deals --- Read our review
    re.compile(r'(:?Further\sdetails\sCode)|(:?Download\sthe\sfree\sapp\sand\sget\smore\sGroupon\sdeals)|(:?Read\sour\sreview)'),

    #  <div> Prev <a> Next </a>  </div>
    re.compile(r'(:?<div>\s*(Prev|Next)\s*<a>\s*(Prev|Next)\s*<\/a>\s*<\/div>)'),

    # <h2>27. Sam Simmonds (Exeter Chiefs / England / British and Irish Lions) +2</h2>
    re.compile(r'(:?<h2>\d+\.\s*[\w\s]+\([\/\s\w]+\).*<\/h2>)'),

    # <ul> <li><a>Bucharest</a></li> </ul> ---
    re.compile(r'^(:?<ul>\s*<li>\s*<a>\s*\w+\s*<\/a>\s*<\/li>\s*<\/ul>)$|(:?(C|c)heck\s(A|a)vailability)'),

    # <div><div>1970 - 1970</div><div>Coupé</div></div>
    # <div><div>96</div><div><div>Lancia Lambda</div><div><div>1922 - 1931</div><div>Convertible</div><div>Italy</div></div></div></div>
    re.compile(r'(:?<div>\s*<div>\s*\d+\s\-\s\d+\s*<\/div><div>.+<\/div>\s*<\/div>)'),

    # <div><div>Car type</div><div><div>Convertible</div><div>Hatchback</div><div>Saloon</div><div>SUV</div><div>Coupé</div><div>MPV</div><div>Estate</div></div></div>
    re.compile(r'(:?<div><div>[\w\s]+<\/div><div>(<div>[\w\s]+<\/div>)+<\/div><\/div>)'),

    # <p>Price: from £39,990</p> --- <p>Energy consumption: Up to 3.0mpkWh</p> --- <p><b>Price:+</b>from £60,970</p> --- <p><b>Range: </b>Up to 285 miles</p>
    re.compile(r'(:?<p>\s*(<b>)?\s*(Range|Price|Energy\sconsumption)\:\s*(<\/b>)?.*<\/p>)'),

    # <li> <div> <div> <h3> Sydney </h3> </div> </div> </li>
    re.compile(r'^(:?(<li>)?(\s*<div>)+\s*<h\d>\s*\w+(\s\w+)?\s*<\/h\d>\s*(<\/div>\s*)+(<\/li>)?)$'),
    # <li> <article> <div> <h2> <a> USA </a> </h2> </div> </article> </li> --- <li> <article> <div> <h2> <a> Resort guides </a> </h2> </div> </article> </li>
    re.compile(r'^(:?<li>\s*<article>\s*<div>\s*<h2>\s*<a>\s*\w+(\s\w+)?\s*<\/a>\s*<\/h2>\s*<\/div>\s*<\/article>\s*<\/li>)$'),

    # <section> <div> <article> </article> </div> </section> --- <choose> <when> </when> </choose> --- <div> <div><opta/></div> </div>
    re.compile(r'^(:?<section>\s*<div>\s*<article>\s*<\/article>\s*<\/div>\s*<\/section>)$|(:?<choose>\s*<when>\s*<\/when>\s*<\/choose>)|(:?<div>\s*<div><opta\/><\/div>\s*<\/div>)'),
    # <section> <div> <article> </article> <article> </article> </div> </section>
    re.compile(r'(:?<section>\s*<div>\s*(<article>\s*<\/article>\s*)+\s*<\/div>\s*<\/section>)'),

  ]

  collect_data(sources, websites, invalid_patterns=invalid_patterns)

# process_telegraph()
```
```python
def process_wapo():
  DEBUG = False
  websites = {
    'washingtonpost.com/': True,
  }

  sources = build_source('raw_html_wapo_{:03d}.jl', 31)

  invalid_patterns = [
    # <wp-ad/> --- <option>January 05, 2022</option>
    re.compile(r'(:?<wp\-ad\/>)|^(:?<option>.*<\/option>)$'),
    # <p> <scrollyvidend>- - -</scrollyvidend> </p> --- <div> <header> </header>  </div>
    re.compile(r'(:?<scrollyvidend>)|^(:?<div>\s*<header>\s*<\/header>\s*<\/div>)$'),
    # <div> <div> <div>Video</div> </div> </div>
    re.compile(r'(:?<div>\s*<div>\s*<div>Video<\/div>\s*<\/div>\s*<\/div>)'),

    # Desktop notifications are on | Turn off --- <ul> <li> Edward Munn </li> <li>·</li> </ul>
    re.compile(r'(:?Desktop\snotifications\sare\son\s\|\sTurn\soff)|(:?<ul>\s*<li>\s*\w+(\s\w+){0,2}\s*<\/li>\s*<li>·<\/li>\s*<\/ul>)'),
    # <div> <div> Most Read Travel </div> </div> --- @washpost.com
    re.compile(r'^(:?<div>\s*<div>\s*Most\s*Read(\s\w+){1,3}\s*<\/div>\s*<\/div>)$|(:?\@washpost\.com)'),
    # The Washington Post --- You might also like: --- Success! Check your inbox for details.
    re.compile(r'(:?The\sWashington\sPost)|(:?You\smight\salso\slike\:\s)|(:?Success\!\sCheck\syour\sinbox\sfor\sdetails\.)'),
    # <ul><li><a>Books</a></li><li>Review</li></ul> --- <b>Published by:</b> --- <b>Available on:</b> --- <b>Developed by:</b>
    re.compile('^(:?<ul>\s*<li>\s*<a>\s*.*</a>\s*</li>\s*<li>\s*Reviews?</li>\s*</ul>)$|(:?<b>Developed\sby\:<\/b>)|(:?<b>Available\son\:<\/b>)|(:?<b>Published\sby\:<\/b>)'),
    # By WashPostPR --- <a>Southwest employee hospitalized</a> | --- Democracy Dies in Darkness
    re.compile('(:?By\sWashPostPR)|^(:?<a>.+?<\/a>\s\|)$|(:?Democracy\sDies\sin\sDarkness)'),
    # <div> <div>Load More</div> </div> --- Follow all of our --- Coronavirus: What you need to read --- Illustrations by iStock
    re.compile('^(:?<div>\s*<div>Load\sMore<\/div>\s*<\/div>)$|(:?Follow\sall\sof\sour)|(:?Coronavirus\:\sWhat\syou\sneed\sto\sread)|(:?Illustrations\sby\siStock)'),
    # <div>Alabama has fully vaccinated 2,349,085 people,</div> --- <div>and 47.9% of the state’s entire population.</div> --- <div>covering 50.9% of the eligible population, 5 and older...</div>
    # <div>668,394 people have received a booster shot,</div>
    re.compile('(:?<div>\w+\shas\sfully\svaccinated.*<\/div>)|(:?<div>(and)?\s\d+\.\d+\%\sof\sthe\sstate\’s\sentire.*<\/div>)'),
    re.compile('(:?<div>covering\s\d+\.\d+\%\sof\sthe\seligible.*<\/div>)|(:?<div>[\d\,]+\speople\shave\sreceived\sa\sbooster\sshot.*<\/div>)'),
    # National park alternatives --- Where to leaf peep this fall --- A case for making a travel journal --- How to find ‘greener’ --- How to get your passport
    re.compile('(:?National\spark\salternatives)|(:?Where\sto\sleaf\speep\sthis\sfall)|(:?A\scase\sfor\smaking\sa\stravel\sjournal)|(:?How\sto\sfind\s\‘greener\’)|(:?How\sto\sget\syour\spassport)'),
    # <h6><i>— Donna St. George</i></h6> --- <div><div><p>Illinois Gov. J.B. Pritzker (D):</p></div></div> ---- <div>January 5, 2022 | 8:13 PM GMT</div> --- <div>January 7, 2021 at 4:23 a.m. EST</div>
    re.compile('(:?<h6>\s*<i>\—\s.+<\/i>\s*<\/h6>)|^(:?<div>\s*<div>\s*<p>.+\(\w\)\:<\/p>\s*<\/div>\s*<\/div>)$|^(:?<div>.*\d+\:\d+\s(PM|AM|p\.?m\.?|a\.?m\.?)\s(GMT|EST)<\/div>)$'),
    # <b>Twitter: </b><a><b>@capow14</b> --- <div><div><p>954-488-2955</p></div></div>
    re.compile(r'(:?<b>\w+\:\s*<\/b>\s*<a>\s*<b>\@\w+<\/b>)|^(:?<div>(\s*<div>)*<p>\d+\-\d+\-\d+<\/p>(<\/div>\s*)*<\/div>)$'),
    # <div><div>Retropolis</div></div>
    re.compile(r'^(:?<div>\s*<div>\s*\w+\s*<\/div>\s*<\/div>)$'),
    # <div><div>102 days to go</div></div> ---  <div><div>38 hours to go</div></div> --- <div><div>30 minutes in</div></div>
    re.compile(r'^(<div>\s*<div>\s*\w+\s(hours?|days?)(\sto\sgo|\safter)\s*<\/div>\s*<\/div>)$'),
    # <div><span><div>Reported by Hannah Allam</div>,</span></div>
    re.compile(r'^(:?<div><span><div>[\w\s]+<\/div>\,.?<\/span><\/div>)$'),
    # <div>At Senate</div> --- Consumer Product Safety Commission --- <div>Termed position</div>
    re.compile(r'(:?<div>At\sSenate<\/div>)|(:?Consumer\sProduct\sSafety\sCommission)|(:?<div>Termed\sposition<\/div>)'),
    # <div><div>Rebecca F. Dye</div><div>June 29, 2016</div><div>Confirmed</div></div>
    re.compile(r'(:?<div><div>.*<\/div>\s*<div>\w+\.?\s\d\d\,\s\d\d\d\d\s*</div>\s*<div>\s*\w+\s*<\/div>\s*<\/div>)'),
    # <ul><li>By Karla Adam</li><li>Jun 12, 2019</li></ul>
    re.compile(r'^(:?<ul>\s*<li>[\w\s]+<\/li>\s*<li>[\w\s\,]+\d\d\d\d\s*<\/li><\/ul>)$'),
  ]

  collect_data(sources, websites, invalid_patterns=invalid_patterns, debug=DEBUG)

# process_wapo()
```
```python
def process_apnews():
  DEBUG = False
  websites = {
    'apnews.com/': True,
  }

  sources = build_source('raw_html_apnews_{:03}.jl', 9)

  invalid_patterns = [
    # <p>______________________________</p>
    re.compile(r'<p>_+<\/p>'),
    # https://apnews.com/article/ashli-babbitt-capitol-siege-a15c7e52a04d932972b7a284c7a8f7df
    re.compile(r'(:?(https\:\/\/)?apnews\.com.*)'),
    #  <div>Full Coverage:+Capitol siege</div> --- <div>Capitol siege+One Year Later</div> ---  <div>Capitol Riots+One Year Later</div>
    re.compile(r'(:?<div>Full\sCoverage\:.?Capitol\ssiege<\/div>)|(:?<div>Capitol\ssiege..?One\sYear\sLater<\/div>)|(:?<div>Capitol\sRiots..?One\sYear\sLater<\/div>)'),
    # <p>Anoka 5, Armstrong/Cooper 1</p>  ---  <p>Duluth Marshall 2, Cloquet/Esko/Carlton 1</p> --- <p>Heritage Academy - Laveen 23, Basis Charter Phoenix 19</p>
    re.compile(r'^(:?<p>([\w\s\.\-]+\,)*[\w\s\/\.\-]+\d+<\/p>)$'),
    # <p>Shonto vs. Fredonia, ccd.</p>
    re.compile(r'^(:?<p>.+ccd\.<\/p>)$'),
    # <p>7. Ultimate Custom Night, Clickteam, LLC</p>
    re.compile(r'^(:?<p>.+LLC<\/p>)$'),
  ]

  collect_data(sources, websites, invalid_patterns=invalid_patterns, debug=DEBUG)

# process_apnews()
```
```python
def process_latimes():
  DEBUG = True
  websites = {
    'latimes.com/': True,
  }

  sources = build_source('raw_html_topm_sites_s150_x50_{:03}.jl', 29)

  invalid_patterns = [
    # LATIMES.COM
    re.compile(r'(:?LATIMES\.COM)|(:?\@?latimes\.com)'),
    # <div> <ps-promo> <div><div><div><p> <a>2020</a> </p></div></div></div> </ps-promo> </div>
    # <ps-youtubeplayer></ps-youtubeplayer> --- <ps-nativo-module> </ps-nativo-module> --- <ps-connatix-module> </ps-connatix-module>
    re.compile(r'(:?<\/?ps\-[\w\-]+>)'),
    #  <a> <span> <?xml version="1.0" encoding="utf-8"?>
    re.compile(r'\?xml\sversion\='),
    # <time>Dec. 3, 2021 11:38 AM PT </time>
    re.compile(r'^(:?<time>.*<\/time>)$'),
    # You may occasionally receive promotional content from the Los Angeles Times. --- More From the Los Angeles Times
    # Read All Read Less
    re.compile(r'(:?Los\sAngeles\sTimes)|(:?Read\sAll\sRead\sLess)'),
    # <div><div> 1/22 </div><div>
    re.compile(r'(:?<div>\s*<div>\s*\d+\/\d+\s*<\/div>\s*<div>)'),
    # <div> website </div> <div> instagram </div>
    # <div> <div> website </div> <div> <a>instagram</a> </div> </div>
    # <div> <div> 800-966-6490 </div> <div> website </div> </div>
    re.compile(r'(:?<div>\s*(<a>\s*)?website(\s*<\/a>)?\s*<\/div>\s*<div>\s*(<a>\s*)?instagram(\s*<\/a>)?\s*<\/div>)'),
    # <div> <div> website </div> </div>
    re.compile(r'^<div>\s*<div>\swebsite\s<\/div>\s*<\/div>$'),
  ]

  collect_data(sources, websites, invalid_patterns=invalid_patterns, debug=DEBUG)

# process_latimes()
```
```python
def process_nypost():
  DEBUG = True
  websites = {
    'nypost.com/': True,
  }

  sources = build_source('raw_html_nypost_{:03}.jl', 9)

  invalid_patterns = [
    # New York Post --- Buy Now --- Shop some of our other favorites:
    re.compile(r'(:?New\sYork\sPost)|(:?Buy\s(N|n)ow)|(:?Shop\ssome\sof\sour\sother\sfavorites\:)'),
    # <span>December 8, 2021 | 4:58pm</span>
    re.compile(r'^<span>.+\d+\:\d+(pm|am)</span>$'),
    #  <div> <div> <div> <header> </header> </div> </div> </div>
    re.compile(r'<div>\s*<div>\s*<div>\s*<header>\s*<\/header>\s*<\/div>\s*<\/div>\s*<\/div>'),
    # <a>#LiveInFrontOfAStudioAudience</a>
    re.compile(r'<a>\#\w+<\/a>'),
    # <h2>More Videos</h2>
    re.compile(r'<h\d>More\sVideos<\/h\d>'),
    # <div> <div> <a> walmart, $100</a> </div> </div>
    re.compile(r'<div>\s*<div>\s*<a>\s*[\w\s]+\,\s*\$\d+<\/a>\s*<\/div>\s*<\/div>'),
  ]

  collect_data(sources, websites, invalid_patterns=invalid_patterns, debug=DEBUG)

# process_nypost()
```
```python
def process_smh():
  DEBUG = True
  websites = {
    'smh.com.au/': True,
  }

  sources = build_source('raw_html_smh_{:03}.jl', 10)

  invalid_patterns = [
    # Sydney Morning Herald -- Public School --- The best photos from around the world --- Australia votes
    re.compile(r'(:?Sydney\sMorning\sHerald)|(:?Public\sSchool)|(:?The\sbest\sphotos\sfrom\saround\sthe\sworld)|(:?Australia\svotes)'),
    # register or subscribe --- MeToo movement
    re.compile(r'(:?register\sor\ssubscribe)|(:?MeToo\smovement)'),
    # <p>Twitter: @JacquelineMaley</p>
    re.compile(r'(:?<p>(T|t)witter\:\s*\@\w+<\/p>)|(:?Most\sViewed)'),
    # <div><h5>★★★½</h5><h5><a>Review</a></h5></div> --- Compiled by DS
    re.compile(r'(:?<h\d>Review<\/h\d>)|(:?Compiled\sby\s\w+)'),
    # <ul><li><time>December 8, 2021</time></li></ul>
    # <ul><li><time>November 21, 2021</time></li><li>by Jacqueline Maley</li></ul>
    re.compile(r'^<ul>\s*(<li>\s*<time>.*<\/time>\s*<\/li>\s*)(<li>.*<\/li>\s*)?\s*<\/ul>$'),
  ]

  collect_data(sources, websites, invalid_patterns=invalid_patterns, debug=DEBUG)

# process_smh()
```
```python
def process_psychologytoday():
  DEBUG = True
  websites = {
    'psychologytoday.com/': True,
  }

  sources = build_source('raw_html_psychologytoday_{:03}.jl', 38)

  invalid_patterns = [
    # Find a Therapist --- Get Help --- View Help Index --- See More Experts
    re.compile(r'(:?Find\sa\sTherapist)|(:?Get\sHelp)|(:?View\sHelp\sIndex)|(:?See\sMore\sExperts)'),
    # <div> advertisement </div> --- A philosopher looks at our deepest emotions --- Friend me on Faceook
    re.compile(r'(:?<div>\s*advertisement\s*<\/div>)|(:?A\sphilosopher\slooks\sat\sour\sdeepest\semotions)'),

    # <a>Your Neurology</a> --- <a>Health</a> --- <h2>The Latest</h2> --- <a>News</a> --- <a>Sex</a> --- <h6>Loneliness</h6> --- <a>Parenting</a>
    # <h3>Today</h3> --- <a>Back</a> --- <a>Psychosis</a> --- <a>Resilience</a> --- <a>Anger</a> --- <a>Relationships</a> --- <a>Trauma</a>
    # <a>Self-Help</a> --- <a>Stress</a> --- <a>Career</a> --- <a>Personality</a> --- <a>Work</a>
    re.compile(r'(:?<a>\s?Your\sNeurology\s?</a>)|(:?<a>\s?Health\s?</a>)|(:?<h2>\s?The\sLatest\s?</h2>)|(:?<a>\s?News\s?</a>)|(:?<a>\s?Sex\s?</a>)|(:?<h6>\s?Loneliness\s?</h6>)|(:?<a>\s?Parenting\s?</a>)'),
    re.compile(r'(:?<h3>\s?Today\s?</h3>)|(:?<a>\s?Back\s?</a>)|(:?<a>\s?Psychosis\s?</a>)|(:?<a>\s?Resilience\s?</a>)|(:?<a>\s?Anger\s?</a>)|(:?<a>\s?Relationships\s?</a>)|(:?<a>\s?Trauma\s?</a>)'),
    re.compile(r'(:?<a>Self-Help\s?</a>)|(:?<a>\s?Stress\s?</a>)|(:?<a>\s?Career\s?</a>)|(:?<a>(P|p)ersonality\s?</a>)|(:?<a>\s?Work\s?</a>)'),
    # <a>Education</a> --- <a>Law and Crime</a> --- <h2>About the Author</h2> --- <h3>Exploring New Frontiers</h3>
    re.compile(r'(:?<a>\s?Education\s?</a>)|(:?<a>\s?Law\sand\sCrime\s?</a>)|(:?<h2>\s?About\sthe\sAuthor\s?</h2>)|(:?<h3>\s?Exploring\sNew\sFrontiers\s?</h3>)'),
    # <a>alcohol</a> --- <h2>Most Popular</h2> --- <a>Decision-Making</a> --- <a>Animal Behavior</a>
    re.compile(r'(:?<a>\s?alcohol\s?</a>)|(:?<h2>\s?Most\sPopular\s?</h2>)|(:?<a>\s?Decision-Making\s?</a>)|(:?<a>\s?Animal\sBehavior\s?</a>)'),

    # <li>Previous</li> --- Page 1 --- <li>Next</li> --- <h2>Read Next</h2> --- <a>press release</a>
    re.compile(r'(:?<li>Previous</li>)|(:?Page\s\d{1,3})|(:?<h2>Read\sNext</h2>)|(:?<a>press\srelease</a>)'),

    # <p>By <a>Nancy Darling Ph.D.</a> on January 11, 2022 in <a>Thinking About Kids</a></p>
    re.compile(r'^<p>By\s<a>.*</a>\son\s.*<a>.*</a>\s*</p>$'),
    # <p>Justin J. Lehmiller Ph.D.</p> --- <p><a>Uriel Abulof Ph.D.</a>, Shirley Le Penne</p>
    re.compile(r'^(:?<p>.+\sPh\.D\.\s?</p>)$|^(:?<p>\s*<a>.+\sPh\.D\.</a>,.+\s?</p>)$'),
    # <p><a>David Scharff M.D.</a></p>
    re.compile(r'^(:?<p>\s*<a>.+\s?M\.D\.</a>\s*</p>)$'),
    # <p>Tyler J. VanderWeele, Director</p>
    re.compile(r'^<p>.+\,\sDirector\s?</p>$'),
    # <div>Last updated: 02/26/2019 </div>
    re.compile(r'^<div>Last\supdated\:\s?\d+\/\d+\/\d+\s?<\/div>$'),

    # <div> <div> 0800 627 004 x11 </div> </div>
    re.compile(r'<div>\s*<div>\s*\d+\s\d+\s\d+(\sx\d+)?\s*</div>\s*</div>'),
    #  <div> <div> (01) 267 6622 x48 </div> </div>
    re.compile(r'<div>\s*<div>\s*\(\d+\)\s\d+\s\d+(\sx\d+)?\s*</div>\s*</div>'),
    # Office is near: --- ONLINE THERAPY --- Offers online therapy --- Ask about video sessions.
    re.compile(r'(:?Office\sis\snear\:)|(:?ONLINE\sTHERAPY)|(:?Offers\sonline\stherapy)|(:?Ask\sabout\svideo\ssessions)'),
    # <a> New Zealand </a> --- <a>Psychology Today</a> ---  <span>Email </span> --- <div> Email </div>
    re.compile(r'(:?<a>\s*New\sZealand\s*</a>)|(:?<a>\s*Psychology\sToday\s*</a>)|(:?<span>\s*Email\s*</span>)|(:?<div>\s*Email\s*</div>)'),
    # <p><a>Gary W. Lewandowski Jr. Ph.D.</a></p>
    re.compile(r'^(:?<p>\s*<a>.+\sPh\.D\.</a>\s*</p>)$'),
    # <p>© Ann Gold Buscho, Ph.D. 2021</p>
    re.compile(r'^(:?<p>.+\sPh\.D\.\s\d+\s*</p>)$'),
    # <p>Jade Wu Ph.D., Monica Johnson Psy.D.</p>
    re.compile(r'^(:?<p>.+\sPsy\.D\.\s*</p>)$'),
    # Facebook image:
    re.compile(r'(:?Facebook\simage\:)'),
    # <p> Posted December 15, 2021 | Reviewed by Davia Sills </p>
    re.compile(r'(:?<p>\s*Posted\s.+\s\|\sReviewed\sby\s.+</p>)'),
    # <div> <div> (267) 680-7188 x434 </div> </div>
    re.compile(r'<div>\s*\(\d+\)\s\d+\-\d+(\sx\d+)?\s*</div>'),
  ]

  # collect_data(sources, websites, invalid_patterns=invalid_patterns, debug=DEBUG)

# process_psychologytoday()
```
```python
def process_scientificamerican():
  DEBUG = True
  websites = {
    'scientificamerican.com/': True,
  }

  sources = build_source('raw_html_topm_sites_s450_x50_{:03}.jl', 25)

  invalid_patterns = [
    # Scientific American
    re.compile(r'(:?Scientific\sAmerican)'),
    # The above text is a transcript of this podcast --- Rights Permissions --- Recent Articles by --- Full Transcript
    re.compile(r'(:?The\sabove\stext\sis\sa\stranscript\sof\sthis\spodcast)|(:?Rights\sPermissions)|(:?Recent\sArticles\sby)|(:?Full\sTranscript)'),
    # Support science journalism --- Latest Stories
    re.compile(r'(:?Support\sscience\sjournalism)|(:?Latest\sStories)'),
    # An edited transcript of the interview follows --- Give the Gift of Knowledge --- Give a Gift 2021
    re.compile(r'(:?An\sedited\stranscript\sof\sthe\sinterview\sfollows)|(:?Give\sthe\sGift\sof\sKnowledge)|(:?Give\sa\sGift\s\d\d\d\d)'),
    # <a>speaker series</a> --- <div>Special Report</div> --- You can listen to all past episodes <a>here</a>
    re.compile(r'(:?<a>\s*speaker\sseries\s*<\/a>)|(:?<div>\s*Special\sReport\s*<\/div>)|(:?\s?You\scan\slisten\sto\sall\spast\sepisodes\s?<a>here</a>)'),
    # watch the entire interview here
    re.compile(r'(:?watch\sthe\sentire\sinterview\shere)'),
    # <div>June 23, 2021 — Devin Williams</div> --- <li> November 12, 2021 — Devin Williams</li>
    re.compile(r'(:?(<div>|<li>)\s?\w+\s\d+\,\s\d+\s\—\s.*(<\/div>|<\/li>))'),
    # pitchmindmatters@gmail.com
    re.compile(r'(:?\w+\@gmail\.com)'),
    # <div><div>Climate Change</div></div> --- <div><div>Neuroscience</div></div> --- <div><div>Quantum Physics</div></div>
    # <div><div>Creativity</div></div> --- <div><div>Natural Disasters</div></div> --- <div><div>Astrophysics</div></div>
    # <div><div>Epidemiology</div></div>
    re.compile(r'^(:?<div><div>Climate\sChange</div></div>)$'),
    re.compile(r'^(:?<div><div>Neuroscience</div></div>)$'),
    re.compile(r'^(:?<div><div>Quantum\sPhysics</div></div>)$'),
    re.compile(r'^(:?<div><div>Creativity</div></div>)$'),
    re.compile(r'^(:?<div><div>Natural\sDisasters</div></div>)$'),
    re.compile(r'^(:?<div><div>Astrophysics</div></div>)$'),
    re.compile(r'^(:?<div><div>Epidemiology</div></div>)$'),
    # <div>10 hours ago — Iris Berent | Opinion</div>
    re.compile(r'^(:?<div>\d+\shours?\sago\s\—.*<\/div>)$'),
  ]

  collect_data(sources, websites, invalid_patterns=invalid_patterns, debug=DEBUG, _batch_size=1000)

# process_scientificamerican()
```
