import itertools

import tensorflow as tf
from tensorflow import keras

HUGE_FLOAT = 1e9

@tf.custom_gradient
def custom_norm(x):
    y = tf.linalg.norm(x, axis=-1, keepdims=True)

    def grad(dy):
        return dy * (x / (y + 1e-19))

    return y, grad

def vecvec(a, b):
    """vector*vector -> scalar + bivector"""
    products = a[..., tf.newaxis]*b[..., tf.newaxis, :]
    old_shape = tf.shape(products)
    new_shape = tf.concat([old_shape[:-2], [9]], -1)
    products = tf.reshape(products, new_shape)
    # 0 1 2
    # 3 4 5
    # 6 7 8
    swizzle = tf.constant([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, -1, 0],
        [0, 0, 0, -1],
        [1, 0, 0, 0],
    ], dtype=products.dtype)
    return tf.tensordot(products, swizzle, 1)
    return tf.linalg.matmul(products, swizzle, b_is_sparse=True)

def vecvec_invariants(p):
    result = [p[..., :1], custom_norm(p[..., 1:4])]
    return tf.concat(result, axis=-1)

def bivecvec(p, c):
    """(scalar + bivector)*vector -> vector + trivector"""
    products = p[..., tf.newaxis]*c[..., tf.newaxis, :]
    old_shape = tf.shape(products)
    new_shape = tf.concat([old_shape[:-2], [12]], -1)
    products = tf.reshape(products, new_shape)
    # 0 1 2
    # 3 4 5
    # 6 7 8
    # 9 10 11
    swizzle = tf.constant([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, -1, 0],
        [0, 0, 0, -1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
    ], dtype=products.dtype)
    return tf.tensordot(products, swizzle, 1)
    return tf.linalg.matmul(products, swizzle, b_is_sparse=True)

def bivecvec_invariants(q):
    result = [custom_norm(q[..., :3]), q[..., 3:4]]
    return tf.concat(result, axis=-1)

def trivecvec(q, d):
    """(vector + trivector)*vector -> scalar + bivector"""
    products = q[..., tf.newaxis]*d[..., tf.newaxis, :]
    old_shape = tf.shape(products)
    new_shape = tf.concat([old_shape[:-2], [12]], -1)
    products = tf.reshape(products, new_shape)
    # 0 1 2
    # 3 4 5
    # 6 7 8
    # 9 10 11
    swizzle = tf.constant([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, -1, 0],
        [0, 0, 0, -1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
    ], dtype=products.dtype)
    return tf.tensordot(products, swizzle, 1)
    return tf.linalg.matmul(products, swizzle, b_is_sparse=True)

trivecvec_invariants = vecvec_invariants

class GradientLayer(keras.layers.Layer):
    def call(self, inputs):
        return tf.gradients(inputs[0], inputs[1])

class MomentumNormalization(keras.layers.Layer):
    def __init__(self, momentum=.99, epsilon=1e-7, *args, **kwargs):
        self.momentum = momentum
        self.epsilon = epsilon
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        shape = [1]*len(input_shape)

        self.mu = self.add_weight(
            name='mu', shape=shape, initializer='zeros', trainable=False)
        self.sigma = self.add_weight(
            name='sigma', shape=shape, initializer='ones', trainable=False)

    def call(self, inputs, training=False):
        if training:
            mean = tf.math.reduce_mean(inputs, keepdims=True)
            std = tf.math.reduce_std(inputs, keepdims=True)
            self.mu.assign(self.momentum*self.mu + (1 - self.momentum)*mean)
            self.sigma.assign(self.momentum*self.sigma + (1 - self.momentum)*std)

        return (inputs - self.mu)/(self.sigma + self.epsilon)

    def get_config(self):
        result = super().get_config()
        result['momentum'] = self.momentum
        result['epsilon'] = self.epsilon
        return result

class NeighborhoodReduction(keras.layers.Layer):
    def __init__(self, mode='sum', *args, **kwargs):
        self.mode = mode

        super().__init__(*args, **kwargs)

    def call(self, inputs, mask=None):
        result = inputs
        if mask is not None:
            mask = tf.expand_dims(mask, -1)
            result = tf.where(mask, inputs, tf.zeros_like(inputs))

        if self.mode == 'sum':
            return tf.math.reduce_sum(result, axis=-2)
        elif self.mode == 'soft_max':
            return tf.math.reduce_logsumexp(result, axis=-2)
        else:
            raise NotImplementedError()

    def get_config(self):
        result = super().get_config()
        result['mode'] = self.mode
        return result

class PairwiseValueNormalization(keras.layers.Layer):
    def call(self, inputs, training=False):
        mu = tf.math.reduce_mean(inputs, axis=(-2, -3), keepdims=True)
        sigma = tf.math.reduce_std(inputs, axis=(-2, -3), keepdims=True)
        return (inputs - mu)/sigma

class PairwiseVectorDifference(keras.layers.Layer):
    def call(self, inputs):
        return inputs[..., None, :] - inputs[..., None, :, :]

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return mask

        # (..., N, 3) -> (..., N)
        mask = tf.reduce_all(tf.not_equal(inputs, 0), axis=-1)
        # (..., N, N)
        mask = tf.logical_and(mask[..., None], mask[..., None, :])
        return mask

class PairwiseVectorDifferenceSum(keras.layers.Layer):
    def call(self, inputs):
        return tf.concat([
            inputs[..., None, :] - inputs[..., None, :, :],
            inputs[..., None, :] + inputs[..., None, :, :]
        ], axis=-1)

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return mask

        mask = tf.reduce_any(tf.not_equal(inputs, 0), axis=-1)
        mask = tf.logical_and(mask[..., None], mask[..., None, :])
        return mask

class VectorAttention(keras.layers.Layer):
    def __init__(self, score_net, value_net, reduce=True,
                 merge_fun='mean', join_fun='mean', rank=2, *args, **kwargs):
        self.score_net = score_net
        self.value_net = value_net
        self.reduce = reduce
        self.merge_fun = merge_fun
        self.join_fun = join_fun
        self.rank = rank

        if merge_fun == 'mean':
            self.merge_fun_ = lambda *args: sum(args)/float(len(args))
        elif merge_fun == 'concat':
            self.merge_fun_ = lambda *args: sum(
                [tf.tensordot(x, b, 1) for (x, b) in zip(args, self.merge_kernels)])
        else:
            raise NotImplementedError()

        if join_fun == 'mean':
            self.join_fun_ = lambda *args: tf.reduce_mean(args, axis=0)
        elif join_fun == 'concat':
            self.join_fun_ = lambda *args: sum(
                [tf.tensordot(x, b, 1) for (x, b) in zip(args, self.join_kernels)])
        else:
            raise NotImplementedError()

        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        (r_shape, v_shape) = input_shape

        self.merge_kernels = None
        if self.merge_fun == 'concat':
            stdev = tf.sqrt(2./self.rank/v_shape[-1])
            self.merge_kernels = [self.add_weight(
                name='merge_kernel_{}'.format(i), shape=(v_shape[-1], v_shape[-1]),
                initializer=keras.initializers.RandomNormal(stddev=stdev)
            ) for i in range(self.rank)]

        self.join_kernels = None
        if self.join_fun == 'concat':
            # always joining neighborhood values and invariant values
            stdev = tf.sqrt(2./2/v_shape[-1])
            self.join_kernels = [self.add_weight(
                name='join_kernel_{}'.format(i), shape=(v_shape[-1], v_shape[-1]),
                initializer=keras.initializers.RandomNormal(stddev=stdev)
            ) for i in range(2)]

        return super().build(input_shape)

    def _expand_products(self, rs, vs):
        broadcast_indices = []
        for i in range(1, self.rank + 1):
            index = [Ellipsis] + [None]*(self.rank) + [slice(None)]
            index[-i - 1] = slice(None)
            broadcast_indices.append(index)

        expanded_vs = [vs[index] for index in broadcast_indices]
        expanded_rs = [rs[index] for index in broadcast_indices]

        product_funs = itertools.chain(
            [vecvec], itertools.cycle([bivecvec, trivecvec]))
        invar_funs = itertools.chain(
            [vecvec_invariants], itertools.cycle([bivecvec_invariants, trivecvec_invariants]))

        left = expanded_rs[0]

        invar_fn = custom_norm
        for (product_fn, invar_fn, right) in zip(product_funs, invar_funs, expanded_rs[1:]):
            left = product_fn(left, right)

        result = invar_fn(left)

        return broadcast_indices, result, expanded_vs

    def _intermediates(self, inputs, mask=None):
        (positions, values) = inputs
        (broadcast_indices, invariants, expanded_values) = self._expand_products(positions, values)
        neighborhood_values = self.merge_fun_(*expanded_values)
        invar_values = self.value_net(invariants)

        joined_values = self.join_fun_(invar_values, neighborhood_values)
        new_values = joined_values

        scores = self.score_net(joined_values)
        old_shape = tf.shape(scores)

        if mask is not None:
            (position_mask, value_mask) = mask
            if position_mask is not None:
                position_mask = tf.expand_dims(position_mask, -1)
                position_mask = tf.reduce_all([position_mask[idx] for idx in broadcast_indices[:-1]], axis=0)
            else:
                position_mask = True
            if value_mask is not None:
                value_mask = tf.expand_dims(value_mask, -1)
                value_mask = tf.reduce_all([value_mask[idx] for idx in broadcast_indices[:-1]], axis=0)
            else:
                value_mask = True
            product_mask = tf.logical_and(position_mask, value_mask)
            scores = tf.where(product_mask, scores, -HUGE_FLOAT)

        if self.reduce:
            dims = -(self.rank + 1)
            reduce_axes = tuple(-i - 2 for i in range(self.rank))
        else:
            dims = -self.rank
            reduce_axes = tuple(-i - 2 for i in range(self.rank - 1))

        shape = tf.concat([old_shape[:dims], tf.math.reduce_prod(old_shape[dims:], keepdims=True)], -1)
        scores = tf.reshape(scores, shape)
        attention = tf.reshape(tf.nn.softmax(scores), old_shape)
        output = tf.reduce_sum(attention*new_values, reduce_axes)

        return dict(attention=attention, output=output, invariants=invar_values)

    def attention(self, inputs):
        return self._intermediates(inputs)['attention']

    def call(self, inputs, return_invariants=False, return_attention=False, mask=None):
        intermediates = self._intermediates(inputs, mask)
        result = [intermediates['output']]
        if return_invariants:
            result.append(intermediates['invariants'])
        if return_attention:
            result.append(intermediates['attention'])

        if len(result) > 1:
            return tuple(result)
        else:
            return result[0]

    def compute_mask(self, inputs, mask=None):
        if not self.reduce or mask is None:
            return mask

        (position_mask, value_mask) = mask
        if position_mask is not None:
            return tf.reduce_any(position_mask, axis=-1)
        else:
            return tf.reduce_any(value_mask, axis=-1)

    @classmethod
    def from_config(cls, config):
        new_config = dict(config)
        for key in ('score_net', 'value_net'):
            new_config[key] = keras.models.Sequential.from_config(new_config[key])
        return cls(**new_config)

    def get_config(self):
        result = super().get_config()
        result['score_net'] = self.score_net.get_config()
        result['value_net'] = self.value_net.get_config()
        result['reduce'] = self.reduce
        result['merge_fun'] = self.merge_fun
        result['join_fun'] = self.join_fun
        result['rank'] = self.rank
        return result
