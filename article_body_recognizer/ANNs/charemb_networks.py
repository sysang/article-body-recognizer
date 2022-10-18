from tensorflow.keras.layers import Input, Embedding, Dense, LSTM, Bidirectional, Dropout, LayerNormalization, Layer, BatchNormalization


class CharEmbeddingV5(Layer):
  def __init__(self, num_classes, max_length, trainable=True, **kwargs):
    dim_embeddings = 11
    dense_compressed_1_size = 5

    self.embedding_layer = Embedding(num_classes, dim_embeddings, input_length=max_length, trainable=trainable, name='embedding')
    self.dense_compressed_layer_1 = Dense(dense_compressed_1_size, name='dense_compressed_layer_1', trainable=trainable, activation='tanh')

    super(CharEmbeddingV5, self).__init__(**kwargs)

  def call(self, inputs):
    embedded = self.embedding_layer(inputs)
    dense_compressed_1 = self.dense_compressed_layer_1(embedded)

    return dense_compressed_1

  def get_config(self):
    cfg = super().get_config()
    return cfg

