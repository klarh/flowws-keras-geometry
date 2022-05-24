from .internal import GradientLayer, \
    NeighborhoodReduction, \
    PairwiseValueNormalization, PairwiseVectorDifference, \
    PairwiseVectorDifferenceSum

import flowws
from flowws import Argument as Arg
from geometric_algebra_attention import keras as gala
import numpy as np
import tensorflow as tf
from tensorflow import keras

LAMBDA_ACTIVATIONS = {
    'leakyswish': lambda x: tf.nn.swish(x) - 1e-2*tf.nn.swish(-x),
    'log1pswish': lambda x: tf.math.log1p(tf.nn.swish(x)),
    'sin': tf.sin,
}

NORMALIZATION_LAYERS = {
    None: lambda _: [],
    'none': lambda _: [],
    'batch': lambda _: [keras.layers.BatchNormalization()],
    'layer': lambda _: [keras.layers.LayerNormalization()],
    'momentum': lambda _: [gala.MomentumNormalization()],
    'momentum_layer': lambda _: [gala.MomentumLayerNormalization()],
}

NORMALIZATION_LAYER_DOC = ' (any of {})'.format(
    ','.join(map(str, NORMALIZATION_LAYERS))
)

class NormalizeLayerBy(keras.layers.Layer):
    def __init__(self, power=1, *args, **kwargs):
        self.power = power
        super().__init__(*args, **kwargs)

    def call(self, x):
        from geometric_algebra_attention.tensorflow.geometric_algebra import custom_norm
        (left, right) = x
        norm = custom_norm(right)
        if self.power != 1:
            norm = norm**self.power
        norm = tf.clip_by_value(norm, 1e-7, np.inf)
        return left/norm

    def get_config(self):
        result = super().get_config()
        result['power'] = self.power
        return result

@flowws.add_stage_arguments
class GalaMoleculeForceRegression(flowws.Stage):
    """Build a geometric attention network for the molecular force regression task.

    This module specifies the architecture of a network to calculate
    atomic forces given the coordinates and types of atoms in a
    molecule. Conservative forces are computed by calculating the
    gradient of a scalar.

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
        Arg('dropout', '-d', float, 0,
            help='Dropout rate to use, if any'),
        Arg('mlp_layers', None, int, 1,
            help='Number of hidden layers for score/value MLPs'),
        Arg('n_blocks', '-b', int, 2,
            help='Number of deep blocks to use'),
        Arg('block_nonlinearity', None, bool, True,
            help='If True, add a nonlinearity to the end of each block'),
        Arg('residual', '-r', bool, True,
            help='If True, use residual connections within blocks'),
        Arg('activation', '-a', str, 'swish',
            help='Activation function to use inside the network'),
        Arg('final_activation', None, str, 'swish',
            help='Final activation function to use within the network'),
        Arg('score_normalization', None, str, 'layer',
            help=('Normalizations to apply to score (attention) function' +
                  NORMALIZATION_LAYER_DOC)),
        Arg('value_normalization', None, str, 'layer',
            help=('Normalizations to apply to value function' +
                  NORMALIZATION_LAYER_DOC)),
        Arg('block_normalization', None, str, 'layer',
            help=('Normalizations to apply to the output of each attention block' +
                  NORMALIZATION_LAYER_DOC)),
        Arg('invariant_value_normalization', None, str, 'momentum',
            help=('Normalizations to apply to value function, before MLP layers' +
                  NORMALIZATION_LAYER_DOC)),
        Arg('equivariant_value_normalization', None, str, 'layer',
            help=('Normalizations to apply to equivariant results' +
                  NORMALIZATION_LAYER_DOC)),
        Arg('invariant_mode', None, str, 'single',
           help='Attention invariant_mode to use'),
        Arg('covariant_mode', None, str, 'single',
           help='Multivector2MultivectorAttention covariant_mode to use'),
        Arg('use_multivectors', None, bool, False,
            help='If True, use multivector intermediates for calculations'),
        Arg('include_normalized_products', None, bool, False,
           help='Also include normalized geometric product terms'),
        Arg('normalize_equivariant_values', None, bool, False,
           help='If True, divide equivariant values by magnitude of inputs after each attention step'),
    ]

    def run(self, scope, storage):
        use_weights = scope.get('use_bond_weights', False)
        n_dim = self.arguments['n_dim']
        dilation = self.arguments['dilation']
        block_nonlin = self.arguments['block_nonlinearity']
        residual = self.arguments['residual']
        join_fun = self.arguments['join_fun']
        merge_fun = self.arguments['merge_fun']
        dropout = self.arguments['dropout']
        num_blocks = self.arguments['n_blocks']
        rank = self.arguments['rank']
        activation = self.arguments['activation']
        distance_norm = self.arguments.get('normalize_distances', None)
        invar_mode = self.arguments['invariant_mode']
        covar_mode = self.arguments['covariant_mode']
        DropoutLayer = scope.get('dropout_class', keras.layers.Dropout)

        normalization_getter = lambda key: (
            NORMALIZATION_LAYERS[self.arguments.get(key + '_normalization', None)](rank)
        )

        if self.arguments['use_multivectors']:
            Attention = gala.MultivectorAttention
            AttentionVector = gala.Multivector2MultivectorAttention
            maybe_upcast_vector = gala.Vector2Multivector()
            maybe_downcast_vector = gala.Multivector2Vector()
        else:
            Attention = gala.VectorAttention
            AttentionVector = gala.Vector2VectorAttention
            maybe_upcast_vector = lambda x: x


        if self.arguments['activation'] in LAMBDA_ACTIVATIONS:
            activation_layer = lambda: keras.layers.Lambda(
                LAMBDA_ACTIVATIONS[self.arguments['activation']])
        else:
            activation_layer = lambda: keras.layers.Activation(
                self.arguments['activation'])

        if self.arguments['final_activation'] in LAMBDA_ACTIVATIONS:
            final_activation_layer = lambda: keras.layers.Lambda(
                LAMBDA_ACTIVATIONS[self.arguments['final_activation']])
        else:
            final_activation_layer = lambda: keras.layers.Activation(
                self.arguments['final_activation'])

        dilation_dim = int(np.round(n_dim*dilation))

        def make_scorefun():
            layers = [keras.layers.Dense(dilation_dim)]

            layers.extend(normalization_getter('score'))

            layers.append(activation_layer())
            if dropout:
                layers.append(DropoutLayer(dropout))

            layers.append(keras.layers.Dense(1))
            return keras.models.Sequential(layers)

        def make_valuefun(dim, in_network=True):
            layers = []

            if in_network:
                layers.extend(normalization_getter('invariant_value'))

            layers.append(keras.layers.Dense(dilation_dim))
            layers.extend(normalization_getter('value'))

            layers.append(activation_layer())
            if dropout:
                layers.append(DropoutLayer(dropout))

            layers.append(keras.layers.Dense(dim))
            return keras.models.Sequential(layers)

        def make_block(last_x, last):
            residual_in_x = last_x
            residual_in = last
            if self.arguments['use_multivectors']:
                arg = [last_x, last, w_in] if use_weights else [last_x, last]
                last_x = gala.Multivector2MultivectorAttention(
                    make_scorefun(),
                    make_valuefun(n_dim),
                    make_valuefun(1),
                    False,
                    rank=rank,
                    join_fun=join_fun,
                    merge_fun=merge_fun,
                    invariant_mode=invar_mode,
                    covariant_mode=covar_mode,
                    include_normalized_products=self.arguments['include_normalized_products'],
                )(arg)

            arg = [last_x, last, w_in] if use_weights else [last_x, last]
            last = Attention(
                make_scorefun(),
                make_valuefun(n_dim),
                False,
                rank=rank,
                join_fun=join_fun,
                merge_fun=merge_fun,
                invariant_mode=invar_mode,
                covariant_mode=covar_mode,
                include_normalized_products=self.arguments['include_normalized_products'],
            )(arg)

            if block_nonlin:
                last = make_valuefun(n_dim, in_network=False)(last)

            if residual:
                last = last + residual_in

            for layer in normalization_getter('block'):
                last = layer(last)

            if self.arguments['use_multivectors']:
                if self.arguments['normalize_equivariant_values']:
                    last_x = NormalizeLayerBy(rank - 1)([last_x, residual_in_x])
                if residual:
                    last_x = residual_in_x + last_x
                for layer in normalization_getter('equivariant_value'):
                    last_x = layer(last_x)

            return last_x, last

        x_in = keras.layers.Input((scope['neighborhood_size'], 3))
        v_in = keras.layers.Input((scope['neighborhood_size'], scope['num_types']))
        w_in = None
        inputs = [x_in, v_in]
        if use_weights:
            w_in = keras.layers.Input((None,), name='wij')
            inputs = [x_in, v_in, w_in]

        delta_x = PairwiseVectorDifference()(x_in)
        delta_v = PairwiseVectorDifferenceSum()(v_in)

        last_x = maybe_upcast_vector(delta_x)
        last = keras.layers.Dense(n_dim)(delta_v)
        for _ in range(num_blocks):
            last_x, last = make_block(last_x, last)

        arg = [last_x, last, w_in] if use_weights else [last_x, last]
        (last, ivs, att) = Attention(
            make_scorefun(),
            make_valuefun(n_dim),
            True,
            name='final_attention',
            rank=rank,
            join_fun=join_fun,
            merge_fun=merge_fun,
            invariant_mode=invar_mode,
            covariant_mode=covar_mode,
        )(arg, return_invariants=True, return_attention=True)

        last = keras.layers.Dense(dilation_dim, name='final_mlp')(last)
        last = final_activation_layer()(last)
        last = NeighborhoodReduction()(last)
        last = keras.layers.Dense(1, name='energy_projection', use_bias=False)(last)
        last = GradientLayer()((last, x_in))

        scope['input_symbol'] = [x_in, v_in]
        scope['output'] = last
        scope['loss'] = 'mse'
        scope['attention_model'] = keras.models.Model([x_in, v_in], att)
        scope['invariant_model'] = keras.models.Model([x_in, v_in], ivs)
