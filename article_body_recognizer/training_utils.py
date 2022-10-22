import math

import numpy as np
from tensorflow.keras.preprocessing import sequence


def charrnn_encode_sequence(text, vocab, maxlen, masking_ratio=None):
    '''
    Encodes a text into the corresponding encoding for prediction with
    the model.
    '''

    oov = vocab['oov']
    masking_code = 0
    encoded = np.array([vocab.get(x, oov) for x in text])
    if masking_ratio:
      encoded = mask_squence(encoded, masking_ratio, masking_code)
    return sequence.pad_sequences([encoded], padding='post', maxlen=maxlen)


def mask_squence(sequence, masking_ratio, fill_value):
  lgth = len(sequence)
  masking_lgth = math.floor(lgth * masking_ratio)
  mask = np.full(lgth, 0)
  mask[:masking_lgth] = 1
  np.random.shuffle(mask)

  masked = np.ma.masked_array(sequence, mask=mask, fill_value=fill_value)

  return masked.filled()
