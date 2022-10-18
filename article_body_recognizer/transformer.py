import re
import html
import base64
import math
import io

from lxml import etree

# TEST CASES
# 1. Check if nodes are assigned number consistently
# 2. Check if data does not comply the threshold rule:
#   a) There exists a node has level higher than threshold.
#   b) There exists node has child and has level smaller than threshold
# 3. Check if there exist paths of two nodes that are overlaping (one is child of other)
# 4. Check if represented data convey comprehensively html structure.
def transform_top_level_nodes_to_sequence(lxmltree=None, html_text=''):
  STOP_WORDS = [
    "a", "about", "above", "after", "again", "against", "all","also", "am", "an", "and", "any", "are", "aren't", "are’t", "as",
    "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by","can", "can't", "can’t", "cannot",
    "could", "couldn't", "could’t", "did", "didn't", "didn’t", "do", "does", "doesn't", "doesn’t", "doing", "don't", "don’t", "down", "during", "each",
    "few", "for", "from", "further", "had", "hadn't", "hadn’t", "has", "hasn't", "hasn’t", "have", "haven't", "haven’t", "having", "he", "he'd", "he’d",
    "he'll", "he’ll", "he's", "he’s", "her", "here", "here's", "here’s", "hers", "herself", "him", "himself", "his", "how", "how's", "how’s", "i", "i'd", "i’d",
    "i'll", "i’ll", "i'm", "i’m", "i've", "i’ve", "if", "in", "into", "is", "isn't", "isn’t", "it", "it'll", "it’ll", "it's", "it’s", "its", "itself", "let's", "let’s", "me", "more",
    "most", "mustn't", "mustn’t", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "shan’t", "she", "she'd", "she’d", "she'll", "she’ll", "she's", "she’s", "should",
    "shouldn't", "shouldn’t", "so", "some", "such", "than", "that", "that's", "that’s", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "there’s", "these", "they", "they'd", "they’d", "they'll", "they’ll", "they're", "they’re", "they've", "they’ve", "this", "those", "through",
    "to", "too", "under", "until", "up", "us", "very", "was", "wasn't", "wasn’t", "we", "we'd", "we’d", "we'll", "we’ll", "we're", "we’re", "we've", "we’ve", "were", "weren't", "weren’t",
    "what", "what's", "what’s", "when", "when's","when’s", "where", "where's", "where’s", "which", "while", "who", "who's", "who’s", "whom", "why", "why's", "why’s",
    "will", "with", "won't", "won’t", "would", "wouldn't", "wouldn’t", "you", "you'd", "you’d", "you'll", "you’ll", "you're", "you’re", "you've", "you’ve", "your", "yours", "yourself", "yourselves"
  ]

  node_list = []
  if lxmltree is not None:
    _lxmltree = lxmltree
  else:
    _lxmltree = LxmlTree(html_text)

  def preprocess_text(_text, _stop_words):
    # start_punctuation = re.compile('^(\.|\,|\:|\;|\'|\‘|\"|\“|\(|\!|\?|\|(\'s)|(\’s)|(\'s)|\*|\[|\-|\—|\–|\#)+')
    # stop_punctuation_p = re.compile('(\.|\,|\:|\;|\'|\’|\"|\”|\)|\!|\?|\|(\'s)|(\’s)|(\'s)|\*|\]|\-|\—|\–|\#)+$')

    splited = _text.split()

    filtered = []
    for w in splited:
      _w = w.lower()
      # _w = stop_punctuation_p.sub('', _w)
      # _w = start_punctuation.sub('', _w)
      if _w not in _stop_words and str() != _w:
        filtered.append(_w)

    return ' '.join(filtered)


  def get_node_formatted(ind, node):
    if not isinstance(node.tag, str) or ind is None:
      return ""

    return "<{:04d}#{:>3}>".format(ind, node.tag)

  def path_to_root(node, parents):
    text = _lxmltree.node2string(node)
    text = html.unescape(text)

    # If data is too much for working algorithm to digest It is always harmful to be along with it
    # At this time the hypothesis is the designing algorithm is too simple to make use of stop words (stop words are very important and will always be)
    text = preprocess_text(text, STOP_WORDS)

    ind = _lxmltree.get_node_index(node)
    text = text + get_node_formatted(ind, node)

    for el in reversed(parents):
      ind = _lxmltree.get_node_index(el)
      text += get_node_formatted(ind, node)

    return text

  def traverse_down(node, level=1, parents=[]):
    if level >= _lxmltree.THRESHOLD_DEPTH_LV or not getattr(node, 'getchildren', None) or not len(node.getchildren()):
      node_list.append(path_to_root(node, parents))
      return True

    children = node.getchildren()
    for child in children:
      traverse_down(child, level=level+1, parents=parents + [node])

  traverse_down(_lxmltree.root)

  return '<#>'.join(node_list)


class LxmlTree():
  """
    Wrap Lxml Element to build utility functions
  """

  def __init__(self, xml_text):
    # [IMPORTANT] It synchronizes with counterpart in data-building process;
    # This hyperparameter affects how sequence represent hierarchical data.
    self.THRESHOLD_DEPTH_LV = 14

    # parser = etree.XMLParser(recover=True)
    # self.root = etree.XML(xml_text, parser=parser) # error when broken html text
    self.root = etree.HTML(xml_text)
    self.index_table = {}

    counter = 1
    for el in self.root.iter():
      self.index_nodes(el, counter)
      counter += 1

  def get_node_by_number(self, number):
    return self.index_table[number]

  def get_node_index(self, node):
    index = node.get('index')
    if index is None:
      return None

    return int(index)

  def index_nodes(self, el, counter):
    try:
      el.set('index', str(counter))
    except:
      print('Element that has wierd implementation')

    self.index_table[counter] = el

  def node2string(self, node):
    try:
      _text = etree.tostring(node)
    except:
      print('node: ', node)
      print('tag: ', node.tag)
      raise Exception("Error while get text form of node")

    _text = str(_text, encoding='utf-8')
    # to remove 'index' attribute of node
    pattern = re.compile(r'\sindex\=\"\d+?\"')
    _text = pattern.sub('', _text)

    return _text

  def calc_distributed_probability_of_truth(self, target_number, percentage_of_itinerary=1):
    assert percentage_of_itinerary > 0 and percentage_of_itinerary <= 1, "percentage_of_itinerary argument must be in (0, 1]"

    path_to_root = []
    distributed_probability = []

    target_node = self.get_node_by_number(target_number)
    path_to_root.append(target_number)

    parent = target_node.getparent()
    parent_number = self.get_node_index(parent)
    path_to_root.append(parent_number)

    while parent is not None:
      parent = parent.getparent()
      if parent is not None:
        parent_number = self.get_node_index(parent)
        path_to_root.insert(0, parent_number)

    # print('path_to_root: ', path_to_root)

    full_journey = len(path_to_root)

    if 1 <= percentage_of_itinerary:
      # to exluded root node from distribution coz it does not contain information (every node has root number as 1)
      k = full_journey - 1

      # assign probability to root node
      # k = full_journey
    else:
      k = full_journey * percentage_of_itinerary
      k = math.floor(k)

    assert k > 0, "Bad data being processed!"

    # Source of formula: https://en.wikipedia.org/wiki/1_%2B_2_%2B_4_%2B_8_%2B_%E2%8B%AF
    # 2^0 + 2^1 + ... + 2^k = 2^(k + 1) - 1
    # Find x to satisfy: x + 2x + 4x + ...+ (2^k)x = 1 => x = 1 / (2^(k + 1) - 1)
    base = 2
    multipled_base = 1 / (math.pow(base, k) - 1)

    for ind in path_to_root[(full_journey - k):]:
      distributed_probability.append((ind, multipled_base))
      multipled_base = multipled_base * 2

    return distributed_probability


def get_nodes_indexed(html_text):

  lxmltree = LxmlTree(xml_text=html_text)
  root = lxmltree.root

  counter = 1
  for el in root.iter():
    level = 1

    parent = el.getparent()
    while parent is not None:
      level += 1
      parent = parent.getparent()

    valid = level <= lxmltree.THRESHOLD_DEPTH_LV
    if isinstance(el.tag, str):
      if valid:
        el.set('class', 'valid')
        el.set('valid', 'valid')

      title = "valid: {}, node: {}, tag: {}, level: {}".format(valid, counter, el.tag, level)
      el.set('title', title)
      el.set('node_number', str(counter))

    counter += 1

  return lxmltree.node2string(root)

