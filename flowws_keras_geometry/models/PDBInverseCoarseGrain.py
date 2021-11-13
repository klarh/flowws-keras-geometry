
import flowws
from flowws import Argument as Arg
import tensorflow as tf
from tensorflow import keras

from .internal import HUGE_FLOAT, PairwiseVectorDifference, \
    PairwiseVectorDifferenceSum, VectorAttention, Vector2VectorAttention

class CoarseGrainAttention(Vector2VectorAttention):
    def build(self, input_shape):
        v_shape = input_shape[1]
        result = super().build(input_shape[:-1])

        if self.join_fun == 'concat':
            # always joining neighborhood values and invariant values
            stdev = tf.sqrt(2./3/v_shape[-1])
            self.join_kernels.append(self.add_weight(
                name='join_kernel_{}'.format(3), shape=(v_shape[-1], v_shape[-1]),
                initializer=keras.initializers.RandomNormal(stddev=stdev)
            ))
        return result

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return
        (r_mask, v_mask, cv_mask) = mask
        return cv_mask

    def _expand_products(self, positions, values):
        (bcast, invars, covars, vs) = super()._expand_products(positions, values)
        new_bcast = []
        for idx in bcast:
            idx = list(idx)
            idx.insert(-1 - self.rank, None)
            new_bcast.append(idx)

        invars = tf.expand_dims(invars, -2 - self.rank)
        covars = tf.expand_dims(covars, -2 - self.rank)
        new_vs = [tf.expand_dims(v, -2 - self.rank) for v in vs]
        return new_bcast, invars, covars, new_vs

    def _intermediates(self, inputs, mask=None):
        (positions, values, child_values) = inputs
        (broadcast_indices, invariants, covariants, expanded_values) = \
            self._expand_products(positions, values)
        neighborhood_values = self.merge_fun_(*expanded_values)
        invar_values = self.value_net(invariants)

        swap_i = -self.rank - 1
        swap_j = swap_i - 1
        child_expand_indices = list(broadcast_indices[-1])
        child_expand_indices[swap_i], child_expand_indices[swap_j] = \
            child_expand_indices[swap_j], child_expand_indices[swap_i]
        child_values = child_values[child_expand_indices]

        joined_values = self.join_fun_(child_values, invar_values, neighborhood_values)

        scales = self.scale_net(joined_values)
        scores = self.score_net(joined_values)
        old_shape = tf.shape(scores)

        if mask is not None:
            (position_mask, value_mask, child_value_mask) = mask
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
        output = tf.reduce_sum(attention*covariants*scales, reduce_axes)

        return dict(attention=attention, output=output, invariants=invariants)

@flowws.add_stage_arguments
class PDBInverseCoarseGrain(flowws.Stage):
    """Build a geometric attention network for a coarse-grain backmapping task.

    This module specifies the architecture of a network to produce
    atomic coordinates from a set of coarse-grained beads.

    """

    ARGS = [
        Arg('rank', None, int, 2,
            help='Degree of correlations (n-vectors) to consider'),
        Arg('n_dim', '-n', int, 32,
            help='Working dimensionality of point representations'),
        Arg('dilation', None, float, 2,
            help='Working dimension dilation factor for MLP components'),
        Arg('merge_fun', '-m', str, 'concat',
            help='Method to merge point representations'),
        Arg('join_fun', '-j', str, 'concat',
            help='Method to join invariant and point representations'),
        Arg('n_blocks_coarse', None, int, 2,
            help='Number of deep blocks to use in the coarse-grain space'),
        Arg('n_blocks_fine', None, int, 2,
            help='Number of deep blocks to use in the coarse-grain space'),
        Arg('block_nonlinearity', None, bool, True,
            help='If True, add a nonlinearity to the end of each block'),
        Arg('residual', '-r', bool, True,
            help='If True, use residual connections within blocks'),
        Arg('activation', '-a', str, 'relu',
            help='Activation function to use inside the network'),
        Arg('attention_vector_inputs', None, bool, False,
            help='Use input vectors for vector-vector attention'),
        Arg('attention_learn_projection', None, bool, False,
            help='Use learned projection weights for vector-vector attention'),
    ]

    def run(self, scope, storage):
        rank = self.arguments['rank']
        n_dim = self.arguments['n_dim']
        merge_fun = self.arguments['merge_fun']
        join_fun = self.arguments['join_fun']

        train_data = scope['train_generator']
        sample_batch = next(train_data)

        x_in = keras.layers.Input(sample_batch[0][0].shape[1:], name='rij')
        v_in = keras.layers.Input(sample_batch[0][1].shape[1:], name='tij')
        cv_in = keras.layers.Input(sample_batch[0][2].shape[1:], name='child_t')

        cv_emb = keras.layers.Embedding(len(scope['child_type_names']), n_dim, mask_zero=True)(cv_in)

        dilation_dim = round(n_dim*self.arguments['dilation'])

        def make_scorefun():
            layers = [keras.layers.Dense(dilation_dim)]

            layers.append(keras.layers.Activation(self.arguments['activation']))

            layers.append(keras.layers.Dense(1))
            return keras.models.Sequential(layers)

        def make_valuefun(dim):
            layers = [keras.layers.Dense(dilation_dim)]
            layers.append(keras.layers.LayerNormalization())

            layers.append(keras.layers.Activation(self.arguments['activation']))

            layers.append(keras.layers.Dense(dim))
            return keras.models.Sequential(layers)

        def make_block(last):
            residual_in = last
            last = VectorAttention(
                make_scorefun(), make_valuefun(n_dim), False, rank=rank,
                join_fun=join_fun,
                merge_fun=merge_fun)([x_in, last])

            if self.arguments['block_nonlinearity']:
                last = make_valuefun(n_dim)(last)

            if self.arguments['residual']:
                last = last + residual_in

            return last

        def make_vector_block(vec):
            residual_in = vec

            vec = PairwiseVectorDifference()(vec)
            (vec, ivs, att) = Vector2VectorAttention(
                make_scorefun(), make_valuefun(n_dim), make_valuefun(1), True, rank=rank,
                join_fun=join_fun, merge_fun=merge_fun,
                use_input_vectors=self.arguments['attention_vector_inputs'],
                learn_vector_projection=self.arguments['attention_learn_projection'])(
                    [vec, delta_v], return_invariants=True, return_attention=True)

            if self.arguments['residual']:
                vec = residual_in + vec

            return vec

        last = keras.layers.Dense(n_dim)(v_in)
        for _ in range(self.arguments['n_blocks_coarse']):
            last = make_block(last)

        (vec, ivs, att) = CoarseGrainAttention(
            make_scorefun(), make_valuefun(n_dim), make_valuefun(1), True, name='final_attention',
            rank=1,
            join_fun=join_fun,
            merge_fun=merge_fun)(
            [x_in, last, cv_emb], return_invariants=True, return_attention=True)

        delta_v = PairwiseVectorDifferenceSum()(cv_emb)
        delta_v = keras.layers.Dense(n_dim)(delta_v)

        for _ in range(self.arguments['n_blocks_fine']):
            vec = make_vector_block(vec)

        scope['input_symbol'] = [x_in, v_in, cv_in]
        scope['output'] = vec
        scope['loss'] = 'mse'
        scope['attention_model'] = keras.models.Model([x_in, v_in, cv_in], att)
        scope['invariant_model'] = keras.models.Model([x_in, v_in, cv_in], ivs)
