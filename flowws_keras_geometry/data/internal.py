import collections

import numpy as np
from tensorflow import keras

class ScaledMSE(keras.metrics.MeanSquaredError):
    """Calculate the mean squared error with a specified scaling factor applied."""
    def __init__(self, scale=1., *args, **kwargs):
        self.scale = scale
        super().__init__(*args, **kwargs)

    def result(self):
        return super().result()*self.scale**2

    def get_config(self):
        result = super().get_config()
        result['scale'] = self.scale
        return result

class ScaledMAE(keras.metrics.MeanAbsoluteError):
    """Calculate the mean absolute error with a specific scaling factor applied."""
    def __init__(self, scale=1., *args, **kwargs):
        self.scale = scale
        super().__init__(*args, **kwargs)

    def result(self):
        return super().result()*self.scale

    def get_config(self):
        result = super().get_config()
        result['scale'] = self.scale
        return result

def grab_nested_batch(i, x):
    if isinstance(x, tuple):
        return tuple(grab_nested_batch(i, v) for v in x)
    elif isinstance(x, list):
        return list(grab_nested_batch(i, v) for v in x)
    return x[i]

def nested_batch_len(x):
    if isinstance(x, (tuple, list)):
        return nested_batch_len(x[0])
    return len(x)

def iter_nested_batch(x):
    for i in range(nested_batch_len(x)):
        yield grab_nested_batch(i, x)

class DataMixingPool:
    def __init__(self, size=512, batch_size=16):
        self.size = size
        self.batch_size = batch_size
        self.pending_batches = [[] for _ in range(self.size)]
        self.ready_batches = collections.deque()

    def sample(self, generator, seed=13):
        rng = np.random.default_rng(seed)

        for batch in generator:
            targets = rng.permutation(len(self.pending_batches))
            batch_iter = iter_nested_batch(batch)
            for (target_id, batch_elements) in zip(targets, batch_iter):
                self.pending_batches[target_id].append(batch_elements)
                if len(self.pending_batches[target_id]) >= self.batch_size:
                    self.ready_batches.append(self.pending_batches[target_id])
                    self.pending_batches[target_id] = []

            while self.ready_batches:
                yield self.ready_batches.popleft()
