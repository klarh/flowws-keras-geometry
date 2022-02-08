import collections
import itertools

import tensorflow as tf
from tensorflow import keras

HUGE_FLOAT = 1e9

@tf.custom_gradient
def custom_norm(x):
    """Calculate the norm of a set of vector-like quantities, with some
    numeric stabilization applied to the gradient."""
    y = tf.linalg.norm(x, axis=-1, keepdims=True)

    def grad(dy):
        return dy * (x / (y + 1e-19))

    return y, grad

def bivec_dual(b):
    """scalar + bivector -> vector + trivector

    Calculates the dual of an input value, expressed as (scalar,
    bivector) with basis (1, e12, e13, e23).

    """
    swizzle = tf.constant([
        [0, 0, 0, -1],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [1, 0, 0, 0]
    ], dtype=b.dtype)
    return tf.tensordot(b, swizzle, 1)

def vecvec(a, b):
    """vector*vector -> scalar + bivector

    Calculates the product of two vector inputs with basis (e1, e2,
    e3). Produces a (scalar, bivector) output with basis (1, e12, e13,
    e23).

    """
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

def vecvec_invariants(p):
    """Calculates rotation-invariant attributes of a (scalar, bivector) quantity.

    Returns a 2D output: the scalar and norm of the bivector.

    """
    result = [p[..., :1], custom_norm(p[..., 1:4])]
    return tf.concat(result, axis=-1)

def vecvec_covariants(p):
    """Calculates rotation-covariant attributes of a (scalar, bivector) quantity.

    Converts the bivector to a vector by taking the dual.

    """
    dual = bivec_dual(p)
    return dual[..., :3]

def bivecvec(p, c):
    """(scalar + bivector)*vector -> vector + trivector

    Calculates the product of a (scalar + bivector) and a vector. The
    two inputs are expressed in terms of the basis (1, e12, e13, e23)
    and (e1, e2, e3); the output is expressed in terms of the basis
    (e1, e2, e3, e123).

    """
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

def bivecvec_invariants(q):
    """Calculates rotation-invariant attributes of a (vector, trivector) quantity.

    Returns a 2D output: the norm of the vector and the trivector.

    """
    result = [custom_norm(q[..., :3]), q[..., 3:4]]
    return tf.concat(result, axis=-1)

def bivecvec_covariants(q):
    """Calculates rotation-covariant attributes of a (vector, trivector) quantity.

    Returns the vector.

    """
    return q[..., :3]

def trivecvec(q, d):
    """(vector + trivector)*vector -> scalar + bivector

    Calculates the product of a (vector + trivector) and a vector. The
    two inputs are expressed in terms of the basis (e1, e2, e3, e123)
    and (e1, e2, e3); the output is expressed in terms of the basis
    (1, e12, e13, e23).

    """
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

trivecvec_invariants = vecvec_invariants

trivecvec_covariants = vecvec_covariants

def custom_all(xs):
    """Reduces the arguments with multiplication. Convenience for broadcasting."""
    result = True
    for x in xs:
        result = tf.math.logical_and(result, x)
    return result

class GradientLayer(keras.layers.Layer):
    """Calculates the gradient of one input with respect to the other."""
    def call(self, inputs):
        return tf.gradients(inputs[0], inputs[1])

class MomentumNormalization(keras.layers.Layer):
    """Exponential decay normalization.

    Computes the mean and standard deviation all axes but the last and
    normalizes values to have mean 0 and variance 1; suitable for
    normalizing a vector of real-valued quantities with differing
    units.

    """
    def __init__(self, momentum=.99, epsilon=1e-7, use_mean=True,
                 use_std=True, *args, **kwargs):
        self.momentum = momentum
        self.epsilon = epsilon
        self.use_mean = use_mean
        self.use_std = use_std
        self.supports_masking = True
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        shape = [input_shape[-1]]

        self.mu = self.add_weight(
            name='mu', shape=shape, initializer='zeros', trainable=False)
        self.sigma = self.add_weight(
            name='sigma', shape=shape, initializer='ones', trainable=False)

    def call(self, inputs, training=False, mask=None):
        if training:
            axes = range(len(inputs.shape) - 1)
            if mask is not None:
                values = tf.ragged.boolean_mask(inputs, mask=mask)
            else:
                values = inputs
            mean = tf.math.reduce_mean(values, axis=axes, keepdims=False)
            std = tf.math.reduce_std(values, axis=axes, keepdims=False)
            self.mu.assign(self.momentum*self.mu + (1 - self.momentum)*mean)
            self.sigma.assign(self.momentum*self.sigma + (1 - self.momentum)*std)

        mu = self.mu*tf.cast(self.use_mean, tf.float32)
        use_std = tf.cast(self.use_std, tf.float32)
        denominator = use_std*(self.sigma + self.epsilon) + (1 - use_std)*1.
        result = (inputs - mu)/denominator
        if mask is not None:
            return tf.where(mask, result, inputs)
        return result

    def get_config(self):
        result = super().get_config()
        result['momentum'] = self.momentum
        result['epsilon'] = self.epsilon
        result['use_mean'] = self.use_mean
        result['use_std'] = self.use_std
        return result

class NeighborDistanceNormalization(keras.layers.Layer):
    def __init__(self, mode='min', *args, **kwargs):
        self.mode = mode
        super().__init__(*args, **kwargs)

    def call(self, inputs):
        if self.mode == 'min':
            distances = tf.linalg.norm(inputs, axis=-1, keepdims=True)
            scale = 1./tf.math.reduce_min(distances, axis=-2, keepdims=True)
        elif self.mode == 'mean':
            distances = tf.linalg.norm(inputs, axis=-1, keepdims=True)
            scale = 1./tf.math.reduce_mean(distances, axis=-2, keepdims=True)
        else:
            raise NotImplementedError()

        return inputs*scale

    def get_config(self):
        result = super().get_config()
        result['mode'] = self.mode
        return result

class NeighborhoodReduction(keras.layers.Layer):
    """Reduce values over the local neighborhood axis."""
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
    """Normalize values over the -2 and -3 dimensions of output."""
    def call(self, inputs, training=False):
        mu = tf.math.reduce_mean(inputs, axis=(-2, -3), keepdims=True)
        sigma = tf.math.reduce_std(inputs, axis=(-2, -3), keepdims=True)
        return (inputs - mu)/sigma

class PairwiseVectorDifference(keras.layers.Layer):
    """Calculate the difference of all pairs of vectors in the neighborhood axis."""
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
    """Calculate the symmetric difference and sum of all pairs of vectors in the neighborhood axis."""
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

_AttentionInputType = collections.namedtuple(
    'AttentionInputType', ['positions', 'values', 'weights'])

class VectorAttention(keras.layers.Layer):
    """Calculates geometric product attention.

    This layer implements a set of geometric products over all tuples
    of length `rank`, then sums over them using an attention mechanism
    to perform a permutation-covariant (`reduce=False`) or
    permutation-invariant (`reduce=True`) result.

    :param score_net: keras `Sequential` network producing logits for the attention mechanism
    :param value_net: keras `Sequential` network producing values in the embedding dimension of the network
    :param reduce: if `True`, produce a permutation-invariant result; otherwise, produce a permutation-covariant result
    :param merge_fun: Function used to merge the input values of each tuple before being passed to `join_fun`: 'mean' (no parameters) or 'concat' (learned projection for each tuple position)
    :param join_fun: Function used to join the representations of the rotation-invariant quantities (produced by `value_net`) and the tuple summary (produced by `merge_fun`): 'mean' (no parameters) or 'concat' (learned projection for each representation)
    :param rank: Degree of correlations to consider. 2 for pairwise attention, 3 for triplet-wise attention, and so on. Memory and computational complexity scales as `N**rank`

    """
    def __init__(self, score_net, value_net, reduce=True,
                 merge_fun='mean', join_fun='mean', rank=2,
                 *args, **kwargs):
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
            self.join_fun_ = lambda *args: sum(args)/float(len(args))
        elif join_fun == 'concat':
            self.join_fun_ = lambda *args: sum(
                [tf.tensordot(x, b, 1) for (x, b) in zip(args, self.join_kernels)])
        else:
            raise NotImplementedError()

        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        shapes = self._parse_inputs(input_shape)
        r_shape = shapes.positions
        v_shape = shapes.values

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
            broadcast_indices.append(tuple(index))

        expanded_vs = [vs[index] for index in broadcast_indices]
        expanded_rs = [rs[index] for index in broadcast_indices]

        product_funs = itertools.chain(
            [vecvec], itertools.cycle([bivecvec, trivecvec]))
        invar_funs = itertools.chain(
            [vecvec_invariants], itertools.cycle([bivecvec_invariants, trivecvec_invariants]))
        covar_funs = itertools.chain(
            [vecvec_covariants], itertools.cycle([bivecvec_covariants, trivecvec_covariants]))

        left = expanded_rs[0]

        invar_fn = custom_norm
        covar_fn = lambda x: x
        for (product_fn, invar_fn, covar_fn, right) in zip(
                product_funs, invar_funs, covar_funs, expanded_rs[1:]):
            left = product_fn(left, right)

        invar = invar_fn(left)
        covar = covar_fn(left)

        return broadcast_indices, invar, covar, expanded_vs

    def _intermediates(self, inputs, mask=None):
        parsed_inputs = self._parse_inputs(inputs)
        positions = parsed_inputs.positions
        values = parsed_inputs.values
        (broadcast_indices, invariants, _, expanded_values) = \
            self._expand_products(positions, values)
        neighborhood_values = self.merge_fun_(*expanded_values)
        invar_values = self.value_net(invariants)

        joined_values = self.join_fun_(invar_values, neighborhood_values)
        new_values = joined_values

        tuple_weights = self._make_tuple_weights(broadcast_indices, parsed_inputs.weights)
        scores = self.score_net(joined_values)
        old_shape = tf.shape(scores)

        if mask is not None:
            parsed_mask = self._parse_inputs(mask)
            position_mask = parsed_mask.positions
            value_mask = parsed_mask.values
            if position_mask is not None:
                masks = [position_mask[..., None][idx] for idx in broadcast_indices]
                position_mask = sum(masks) == self.rank
            else:
                position_mask = True
            if value_mask is not None:
                masks = [value_mask[..., None][idx] for idx in broadcast_indices]
                value_mask = sum(masks) == self.rank
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

        shape = tf.concat([old_shape[:dims],
                           tf.math.reduce_prod(old_shape[dims:], keepdims=True)], -1)
        scores = tf.reshape(scores, shape)
        attention = tf.reshape(tf.nn.softmax(scores), old_shape)
        output = tf.reduce_sum(attention*tuple_weights*new_values, reduce_axes)

        return dict(attention=attention, output=output, invariants=invar_values)

    def _make_tuple_weights(self, broadcast_indices, weights):
        if isinstance(weights, int):
            return weights
        expanded_weights = [weights[..., None][idx] for idx in broadcast_indices]
        result = 1
        for w in expanded_weights:
            result = result*w
        return tf.math.pow(result, 1./self.rank)

    def _parse_inputs(self, inputs):
        if len(inputs) == 2:
            (r, v) = inputs
            w = 1
        elif len(inputs) == 3:
            (r, v, w) = inputs
        else:
            raise NotImplementedError(inputs)
        return _AttentionInputType(r, v, w)

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

        parsed_mask = self._parse_inputs(mask)
        position_mask = parsed_mask.positions
        value_mask = parsed_mask.values
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

class Vector2VectorAttention(VectorAttention):
    """Adaptation of `VectorAttention` to produce vector outputs.

    Cf. `VectorAttention` for most arguments.

    :param scale_net: keras `Sequential` network producing scalar rescaling factors for each vector
    :param use_product_vectors: Use the vector extracted from the geometric product of each tuple within the attention mechanism
    :param use_input_vectors: Use the input vectors for each tuple within the attention mechanism. A scalar weight will be learned for each (i, j, k, ...) element of the tuple
    :param learn_vector_projection: Use a learned linear projection combination factor for the summary value (for the product vector) and the input values (for the input vectors)

    """

    def __init__(self, score_net, value_net, scale_net, *args,
                 use_product_vectors=True, use_input_vectors=False,
                 learn_vector_projection=False,
                 **kwargs):
        super().__init__(score_net, value_net, *args, **kwargs)

        self.scale_net = scale_net
        self.use_product_vectors = use_product_vectors
        self.use_input_vectors = use_input_vectors
        self.learn_vector_projection = learn_vector_projection

        self.input_vector_count = (
            int(self.use_product_vectors) + self.rank*self.use_input_vectors)
        if self.input_vector_count < 1:
            raise ValueError('At least one of use_product_vectors or '
                             'use_input_vectors must be True')

    def _covariants(self, covariants_, positions, broadcast_indices, expanded_values,
                    joined_values):
        covariant_vectors = []
        input_values = []
        if self.use_product_vectors:
            covariant_vectors.append(covariants_)
            input_values.append(joined_values)
        if self.use_input_vectors:
            covariant_vectors.extend(
                [positions[idx] for idx in broadcast_indices])
            input_values.extend(expanded_values)

        scalars = self.vector_kernels
        if self.learn_vector_projection:
            scalars = [tf.tensordot(v, kernel, 1) + self.vector_biases[i]
                       for i, (v, kernel) in
                       enumerate(zip(input_values, self.vector_kernels))]

        covariants = [
            vec*scalars[i] for (i, vec) in enumerate(covariant_vectors)]
        return sum(covariants)

    def _intermediates(self, inputs, mask=None):
        parsed_inputs = self._parse_inputs(inputs)
        positions = parsed_inputs.positions
        values = parsed_inputs.values
        (broadcast_indices, invariants, covariants, expanded_values) = \
            self._expand_products(positions, values)
        neighborhood_values = self.merge_fun_(*expanded_values)
        invar_values = self.value_net(invariants)

        joined_values = self.join_fun_(invar_values, neighborhood_values)
        covariants = self._covariants(
            covariants, positions, broadcast_indices, expanded_values, joined_values)

        tuple_weights = self._make_tuple_weights(broadcast_indices, parsed_inputs.weights)
        scales = self.scale_net(joined_values)
        scores = self.score_net(joined_values)
        old_shape = tf.shape(scores)

        if mask is not None:
            parsed_mask = self._parse_inputs(mask)
            position_mask = parsed_mask.positions
            value_mask = parsed_mask.values
            if position_mask is not None:
                position_mask = tf.expand_dims(position_mask, -1)
                position_mask = custom_all([position_mask[idx] for idx in broadcast_indices[:-1]])
            else:
                position_mask = True
            if value_mask is not None:
                value_mask = tf.expand_dims(value_mask, -1)
                value_mask = custom_all([value_mask[idx] for idx in broadcast_indices[:-1]])
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
        output = tf.reduce_sum(attention*tuple_weights*covariants*scales, reduce_axes)

        return dict(attention=attention, output=output, invariants=invariants)

    def build(self, input_shape):
        parsed_inputs = self._parse_inputs(input_shape)
        value_shape = parsed_inputs.values
        value_dim = value_shape[-1]

        self.vector_kernels = [1]
        self.vector_biases = [0]
        if self.use_input_vectors:
            if self.learn_vector_projection:
                self.vector_kernels = [self.add_weight(
                    name='vector_kernels_{}'.format(i), shape=[value_dim, 1])
                    for i in range(self.input_vector_count)]
                self.vector_biases = self.add_weight(
                    name='vector_biases', shape=[self.input_vector_count],
                    initializer=keras.initializers.RandomNormal(mean=0.))
            else:
                self.vector_kernels = self.add_weight(
                    name='vector_kernels', shape=[self.input_vector_count],
                    initializer=keras.initializers.RandomNormal(mean=0.))
        return super().build(input_shape)

    @classmethod
    def from_config(cls, config):
        new_config = dict(config)
        for key in ('scale_net',):
            new_config[key] = keras.models.Sequential.from_config(new_config[key])
        return super(Vector2VectorAttention, cls).from_config(new_config)

    def get_config(self):
        result = super().get_config()
        result['scale_net'] = self.scale_net.get_config()
        result['use_product_vectors'] = self.use_product_vectors
        result['use_input_vectors'] = self.use_input_vectors
        result['learn_vector_projection'] = self.learn_vector_projection
        return result
