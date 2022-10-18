import re
import base64
import html

from lxml import etree

def load_text(f_path):
  with open(f_path, 'r') as f:
    text = f.read()
  return text

def uri_params(params, spider):
    # Follow instruction at https://docs.scrapy.org/en/latest/topics/feed-exports.html#std-setting-FEED_URI_PARAMS
    # This wierd thing happen because of a bug in /usr/local/lib/python3.8/dist-packages/scrapy/extensions/feedexport.py:488
    # instead of return the result of call back it return the original params
    params['spider_name'] = spider.name

def filter_html(html):
  if not html:
      return html

  pipeline = [
      (re.compile('\&amp;'), ''),
      (re.compile('\&nbsp;'), ''),
      (re.compile('\<br[^\>]*\>'), ''),
      (re.compile('\<hr[^\>]*\>'), ''),
      (re.compile('\<link[^\>]*\>'), ''),
      (re.compile('\<meta[^\>]*\>'), ''),
      (re.compile('(\\\\n|\\n|\\\\r|\\r)'), ''),
      (re.compile('\\\\t|\\t'), ''),
      (re.compile('\<\!\-\-.*?\-\-\>'), ''),
      (re.compile('\<strong[^\>]*\>|\<\/strong[^\>]*\>'), ''),
      (re.compile('\<em[^\>]*\>|\<\/em[^\>]*\>'), ''),
      (re.compile('\<del[^\>]*\>|\<\/del[^\>]*\>'), ''),
      (re.compile('\<svg[^\>]*\>.*?\<\\*\/svg\>'), ''),
      (re.compile('\<svg[^\>]*\>.*?\<\/svg\>'), ''),
      (re.compile('\<aside[^\>]*\>.*?\<\/aside\>'), ''),
      (re.compile('(\<img[^\>]*\/\>)|(\<img[^\>]*\>)'), ''),
      (re.compile('(\<input[^\>]*\/\>)|(\<input[^\>]*\>)'), ''),
      (re.compile('\<symbol[^\>]*\>.*?\<\/symbol\>'), ''),
      (re.compile('\<icon[^\>]*\>.*?\<\/icon\>'), ''),
      (re.compile('\<picture[^\>]*\>.*?\<\/picture\>'), ''),
      (re.compile('\<figcaption[^\>]*\>.*?\<\/figcaption\>'), ''),
      (re.compile('\<figure[^\>]*\>.*?\<\/figure\>'), ''),
      (re.compile('\<video[^\>]*\>.*?\<\/video\>'), ''),
      (re.compile('\<video\-js[^\>]*\>.*?\<\/video\-js\>'), ''),
      (re.compile('\<source[^\>]*\>.*?\<\/source\>'), ''),
      (re.compile('\<form[^\>]*\>.*?\<\/form\>'), ''),
      (re.compile('\<button[^\>]*\>.*?\<\/button\>'), ''),
      (re.compile('\<table[^\>]*\>.*?\<\/table\>'), ''),
      # (re.compile('\<header[^\>]*\>.*?\<\/header\>'), ''),
      # (re.compile('\<head[^\>]*\>.*?\<\/head\>'), ''),
      (re.compile('\<title[^\>]*\>.*?\<\/title\>'), ''),
      (re.compile('\<footer[^\>]*\>.*?\<\/footer\>'), ''),
      (re.compile('\<style[^\>]*\>.*?\<\/style\>'), ''),
      (re.compile('\<script[^\>]*\>.*?\<\/script\>'), ''),
      (re.compile('\<noscript[^\>]*\>.*?\<\/noscript\>'), ''),
      (re.compile('\<iframe[^\>]*\>.*?\<\/iframe\>'), ''),
      (re.compile('(<[a-zA-Z0-9_\-]+?\s)([^\>]+)(\/?\>)'), lambda matchObj : matchObj.groups()[0] + matchObj.groups()[2]), #  remove node's attributes
    ]

  _text = ' '.join(html.splitlines())
  for p, repl in pipeline:
    _text = p.sub(repl, _text)

  detect_nested_a_in_span = re.compile('(\<span[^\>]*\>.*?)\<a[^\>]*\>([^\<]+)\<\/a\>(.*?\<\/span\>)')
  detect_nested_span_in_a = re.compile('(\<a[^\>]*\>.*?)\<span[^\>]*\>([^\<]+)\<\/span\>(.*?\<\/a\>)')
  detect_nested_span_in_span = re.compile('(\<span[^\>]*\>.*?)\<span[^\>]*\>([^\<]+)\<\/span\>(.*?\<\/span\>)')
  detect_nested_nav_in_nav = re.compile('(\<nav[^\>]*\>.*?)\<nav[^\>]*\>(.*?)\<\/nav\>(.*?\<\/nav\>)')

  repeating_pipeline = [
      (detect_nested_a_in_span, lambda matchObj: matchObj.groups()[0] + matchObj.groups()[1] + matchObj.groups()[2]), #  reduce the structural complexity by removing nested node
      (detect_nested_span_in_span, lambda matchObj: matchObj.groups()[0] + matchObj.groups()[1] + matchObj.groups()[2]),
      (detect_nested_span_in_a, lambda matchObj: matchObj.groups()[0] + matchObj.groups()[1] + matchObj.groups()[2]),
      (detect_nested_nav_in_nav, lambda matchObj: matchObj.groups()[0] + matchObj.groups()[1] + matchObj.groups()[2]),
      (re.compile('\<section[^\>]*\>\s*\<\/section\>'), ''), #  remove empty nodes
      (re.compile('\<nav[^\>]*\>\s*\<\/nav\>'), ''), #  remove empty nodes
      (re.compile('\<div[^\>]*\>\s*\<\/div\>'), ''), #  remove empty nodes
      (re.compile('\<p[^\>]*\>\s*\<\/p\>'), ''), #  remove empty nodes
      (re.compile('\<span[^\>]*\>\s*\<\/span\>'), ''), #  remove empty nodes
      (re.compile('\<a[^\>]*\>\s*\<\/a\>'), ''), #  remove empty nodes
      (re.compile('\<li[^\>]*\>\s*\<\/li\>'), ''), #  remove empty nodes
      (re.compile('\<ul[^\>]*\>\s*\<\/ul\>'), ''), #  remove empty nodes
      (re.compile('\<h\d[^\>]*\>\s*\<\/h\d\>'), ''), #  remove empty nodes
      (re.compile('\<i[^\>]*\>\s*\<\/i\>'), ''), #  remove empty nodes
      (re.compile('\s\s'), ' '), #  remove space
    ]

  while True:

    old_length = len(_text)
    for p, repl in repeating_pipeline:
      _text = p.sub(repl, _text)
    new_length = len(_text)

    if old_length > new_length:
      old_length = new_length
    else:
      break

  # This round is for processing that needs nested nodes removed in order to function properly
  pipeline = [
      (re.compile('\<nav[^\>]*\>.*?\<\/nav\>'), ''), # Remove nav after all nested nav elements have been removed
    ]

  for p, repl in pipeline:
    _text = p.sub(repl, _text)

  return _text

def base64StrEncode(text: str, scheme='utf-8') -> str:
  encoded_bytes = text.encode(scheme)
  # >>> type(base64_encoded)
  # <class 'bytes'>
  base64_encoded = base64.b64encode(encoded_bytes)
  encoded_str = str(base64_encoded, scheme)

  return encoded_str

def base64StrDecode(encoded_str: str, scheme='utf-8') -> str:
  # >>> type(decoded_bytes)
  # <class 'bytes'>
  decoded_bytes = base64.b64decode(encoded_str)
  decoded_str = str(decoded_bytes, scheme)

  return decoded_str


