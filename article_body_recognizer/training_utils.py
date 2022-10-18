import numpy as np
from tensorflow.keras.preprocessing import sequence


def charrnn_encode_sequence(text, vocab, maxlen):
    '''
    Encodes a text into the corresponding encoding for prediction with
    the model.
    '''

    oov = vocab['oov']
    encoded = np.array([vocab.get(x, oov) for x in text])
    return sequence.pad_sequences([encoded], padding='post', maxlen=maxlen)

