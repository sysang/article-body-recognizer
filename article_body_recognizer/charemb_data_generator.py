import random
import time
import threading
import enum
import itertools
from collections import deque

import numpy as np

from article_body_recognizer.char_dict import vocabularies as vocab
from article_body_recognizer.training_utils import charrnn_encode_sequence


class SplitType(str, enum.Enum):
    TRAINING = 'training'
    VALIDATING = 'validating'


class DataBufferThread(threading.Thread):

    def __init__(
        self,
        name, dataset, model,
        buffer_size, batch_size, max_length,
        min_length, masking_ratio,
        neutral_distance_scale, close_distance_scale,
        split_type, **kwargs,
    ):

        self._stopevent = threading.Event()

        self.dataset = dataset
        self.model = model

        # instantiate primary queue and secondary queue
        # for buffuring data
        self.pribuf_size = buffer_size * 2
        self.subbuf_size = buffer_size
        print('[INFO] Primary deque maxlen: ', self.pribuf_size)
        print('[INFO] Secondary deque maxlen: ', self.subbuf_size)
        self.sub_buffer = deque(maxlen=self.subbuf_size)
        self.pri_buffer = deque(maxlen=self.pribuf_size)

        self.batch_size = batch_size
        self.max_length = max_length
        self.min_length = min_length
        self.masking_ratio = masking_ratio
        self.neutral_distance_scale = neutral_distance_scale
        self.close_distance_scale = close_distance_scale

        assert split_type in list(SplitType), "Invalid split name."
        self.split_type = split_type

        threading.Thread.__init__(self, name=name)

    def join(self, timeout=None):
        """ Stop the thread. """
        self._stopevent.set()
        threading.Thread.join(self, timeout)

    def run(self):
        max_length = self.max_length
        min_length = self.min_length
        masking_ratio = self.masking_ratio
        neutral_distance_scale = self.neutral_distance_scale
        close_distance_scale = self.close_distance_scale

        max_prev_text_quantity = 1000
        prev_texts_queue = deque(
            [
                '<p>This implementation of RMSprop uses plain momentum, \
                                                not Nesterov momentum.</p>'
            ] * max_prev_text_quantity, maxlen=max_prev_text_quantity)

        dataset = itertools.cycle(self.dataset)

        X1_batch = []
        X2_batch = []
        Y1_batch = []
        # Y2_batch = []
        count_batch = 0

        while not self._stopevent.isSet():
            seq = next(dataset)

            lgth = len(seq)
            if min_length > lgth or lgth > max_length:
                continue

            encoded_1 = charrnn_encode_sequence(
                seq,
                vocab,
                max_length,
                masking_ratio=masking_ratio
            )[0]
            encoded_2 = charrnn_encode_sequence(
                seq,
                vocab,
                max_length,
                masking_ratio=masking_ratio,
            )[0]

            X1_batch.append(encoded_1)
            X2_batch.append(encoded_2)
            Y1_batch.append(close_distance_scale)
            X1_batch, X2_batch, Y1_batch, count_batch = \
                self.queue_or_count(
                    X1_batch, X2_batch, Y1_batch, count_batch)

            # Reverse the oder to force distance symmetric
            X1_batch.append(encoded_2)
            X2_batch.append(encoded_1)
            Y1_batch.append(close_distance_scale)
            X1_batch, X2_batch, Y1_batch, count_batch = \
                self.queue_or_count(
                    X1_batch, X2_batch, Y1_batch, count_batch)

            # Invalidating prediction just count for the real
            # truth values. So just include the
            # neutral_distance_scale cases for training
            if self.is_training_split():
                prev = self.get_prev_text(seq, prev_texts_queue)

                # Try to make two comparators balanced
                if self.will_happen_with_respect_to(probability=0.5):
                    encoded_1 = charrnn_encode_sequence(
                        seq, vocab, max_length,
                        masking_ratio=masking_ratio)[0]
                    encoded_2 = charrnn_encode_sequence(
                        prev, vocab, max_length,
                        masking_ratio=masking_ratio)[0]
                else:
                    encoded_2 = charrnn_encode_sequence(
                        seq, vocab, max_length,
                        masking_ratio=masking_ratio)[0]
                    encoded_1 = charrnn_encode_sequence(
                        prev, vocab, max_length,
                        masking_ratio=masking_ratio)[0]

                X1_batch.append(encoded_1)
                X2_batch.append(encoded_2)
                Y1_batch.append(neutral_distance_scale)
                X1_batch, X2_batch, Y1_batch, count_batch = \
                    self.queue_or_count(
                        X1_batch, X2_batch, Y1_batch, count_batch)

            # ATTENTION: Big trouble for you if you put this
            # one outside of the loop
            # REMEMBER: If you put this one outside of the loop
            # everything is perfect except the performance, THAT'S IT!!
            if self.will_happen_with_respect_to(probability=0.61):
                prev_texts_queue.append(seq)

    def queue_or_count(self, _X1, _X2, _Y1, count_batch):

        # Firstly, count batch's volume
        count_batch += 1

        if count_batch % self.batch_size == 0:
            _X1 = np.array(_X1)
            _X2 = np.array(_X2)
            _X = {
                'input_1': _X1,
                'input_2': _X2,
            }

            if self.is_training_split() and self.model:
                _Y = self.measure_distance(_X, _Y1)
            else:
                _Y1 = np.array(_Y1)
                _Y = {
                    'distance_1': _Y1,
                    'distance_2': _Y1,
                    'distance_3': _Y1,
                }

            batch = (_X, _Y)

            self.pri_buffer.appendleft(batch)

            # if the primary buffer is full then wait for the
            # number of remaining items less than buffer_size
            if len(self.pri_buffer) >= self.pri_buffer.maxlen:
                while len(self.pri_buffer) > self.subbuf_size:
                    time.sleep(0.01)

            # Reset batch
            _X1 = []
            _X2 = []
            _Y1 = []
            # _Y2 = []
            count_batch = 0

            return (_X1, _X2, _Y1, count_batch)

        # not thing more than accumulate batch count
        return (_X1, _X2, _Y1, count_batch)

    def is_training_split(self):
        return SplitType.TRAINING == self.split_type

    def measure_distance(self, inputs, labels):
        # print('[DEBUG] Generating truth values ...')
        preds = self.model.predict_on_batch(inputs)
        distance_1 = np.array(preds[0]).squeeze()
        distance_2 = np.array(preds[1]).squeeze()
        distance_3 = np.array(preds[2]).squeeze()

        mask = np.array(labels) == self.close_distance_scale
        distance_1[mask] = self.close_distance_scale
        distance_2[mask] = self.close_distance_scale
        distance_3[mask] = self.close_distance_scale

        _labels = {
            'distance_1': distance_1,
            'distance_2': distance_2,
            'distance_3': distance_3,
        }

        # print('[DEBUG] Done.')

        return _labels

    def will_happen_with_respect_to(self, probability=1):
        if probability == 1:
            return True
        return random.random() <= probability

    def get_prev_text(self, text, prev_texts_queue):
        trial_times = 1000
        prev = random.choice(prev_texts_queue)

        counter = 1
        while prev == text and counter <= trial_times:
            prev = random.choice(prev_texts_queue)
            counter += 1

        return prev

    def iter(self):
        # start execution, run(), in another thread
        self.start()

        while True:
            # check if secondary buffer is ready
            # start iteration, else start reload data
            # from primary buffer
            if len(self.sub_buffer) > 0:
                yield self.sub_buffer.pop()

            # check if primary buffer is ready to
            # stack data into secondary buffer
            # if both primary buffer and secondary buffing are not
            # ready, wait for buffering (in other thread)
            elif len(self.pri_buffer) >= self.subbuf_size:
                for i in range(self.subbuf_size):
                    batch = self.pri_buffer.pop()
                    self.sub_buffer.appendleft(batch)

            else:
                time.sleep(0.01)
