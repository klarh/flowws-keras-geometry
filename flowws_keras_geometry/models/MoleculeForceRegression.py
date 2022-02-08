from .internal import GradientLayer, MomentumNormalization, \
    NeighborhoodReduction, \
    PairwiseValueNormalization, PairwiseVectorDifference, \
    PairwiseVectorDifferenceSum, VectorAttention

import flowws
from flowws import Argument as Arg
import numpy as np
import tensorflow as tf
from tensorflow import keras

LAMBDA_ACTIVATIONS = {
    'log1pswish': lambda x: tf.math.log1p(tf.nn.swish(x)),
    'sin': tf.sin,
}

NORMALIZATION_LAYERS = {
    'layer': lambda _: keras.layers.LayerNormalization(),
    'layer_all': lambda rank: keras.layers.LayerNormalization(axis=[-i - 1 for i in range(rank + 1)]),
    'momentum': lambda _: MomentumNormalization(),
    'pairwise': lambda _: PairwiseValueNormalization(),
}

NORMALIZATION_LAYER_DOC = ' (any of {})'.format(','.join(NORMALIZATION_LAYERS))

@flowws.add_stage_arguments
class MoleculeForceRegression(flowws.Stage):
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
        Arg('score_normalization', None, [str], [],
            help=('Normalizations to apply to score (attention) function' +
                  NORMALIZATION_LAYER_DOC)),
        Arg('value_normalization', None, [str], [],
            help=('Normalizations to apply to value function' +
                  NORMALIZATION_LAYER_DOC)),
        Arg('invar_normalization', None, [str], [],
            help=('Normalizations to apply to invariants in value function' +
                  NORMALIZATION_LAYER_DOC)),
        Arg('normalize_distances', None, str,
            help='Method to use to normalize pairwise distances'),
        Arg('predict_energy', None, bool, False,
            help='If True, predict energies instead of forces'),
        Arg('energy_bias', None, bool, False,
            help='If True, learn a bias term for energy prediction'),
    ]

    def run(self, scope, storage):
        rank = self.arguments['rank']

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

        n_dim = self.arguments['n_dim']
        dilation_dim = int(np.round(n_dim*self.arguments['dilation']))

        def make_scorefun():
            layers = []

            for _ in range(self.arguments['mlp_layers']):
                layers.append(keras.layers.Dense(dilation_dim))

                for name in self.arguments['score_normalization']:
                    layers.append(NORMALIZATION_LAYERS[name](rank))

                layers.append(activation_layer())

                if self.arguments.get('dropout', 0):
                    layers.append(keras.layers.Dropout(self.arguments['dropout']))

            layers.append(keras.layers.Dense(1))
            return keras.models.Sequential(layers)

        def make_valuefun(uses_invars=False):
            layers = []

            if uses_invars:
                for name in self.arguments['invar_normalization']:
                    layers.append(NORMALIZATION_LAYERS[name](rank))

            for _ in range(self.arguments['mlp_layers']):
                layers.append(keras.layers.Dense(dilation_dim))

                for name in self.arguments['value_normalization']:
                    layers.append(NORMALIZATION_LAYERS[name](rank))

                layers.append(activation_layer())

                if self.arguments.get('dropout', 0):
                    layers.append(keras.layers.Dropout(self.arguments['dropout']))

            layers.append(keras.layers.Dense(n_dim))
            return keras.models.Sequential(layers)

        def make_block(last):
            residual_in = last
            last = VectorAttention(
                make_scorefun(), make_valuefun(True), False, rank=rank,
                join_fun=self.arguments['join_fun'],
                merge_fun=self.arguments['merge_fun'])([delta_x, last])

            if self.arguments['block_nonlinearity']:
                last = make_valuefun()(last)

            if self.arguments['residual']:
                last = last + residual_in

            return last

        x_in = keras.layers.Input((scope['neighborhood_size'], 3))
        v_in = keras.layers.Input((scope['neighborhood_size'], scope['num_types']))

        delta_x = PairwiseVectorDifference()(x_in)
        delta_v = PairwiseVectorDifferenceSum()(v_in)

        if 'normalize_distances' in self.arguments:
            mode = self.arguments['normalize_distances']

            if mode == 'momentum':
                delta_x = MomentumNormalization()(delta_x)
            else:
                raise NotImplementedError(mode)

        last = keras.layers.Dense(n_dim)(delta_v)
        for _ in range(self.arguments['n_blocks']):
            last = make_block(last)

        (last, ivs, att) = VectorAttention(
            make_scorefun(), make_valuefun(True), True, name='final_attention',
            rank=rank,
            join_fun=self.arguments['join_fun'],
            merge_fun=self.arguments['merge_fun'])(
            [delta_x, last], return_invariants=True, return_attention=True)

        last = keras.layers.Dense(dilation_dim, name='final_mlp')(last)
        last = final_activation_layer()(last)
        last = NeighborhoodReduction()(last)
        use_bias = self.arguments.get('energy_bias', False)
        last = keras.layers.Dense(1, name='energy_projection', use_bias=use_bias)(last)
        if not self.arguments['predict_energy']:
            last = GradientLayer()((last, x_in))

        scope['input_symbol'] = [x_in, v_in]
        scope['output'] = last
        scope['loss'] = 'mse'
        scope['attention_model'] = keras.models.Model([x_in, v_in], att)
        scope['invariant_model'] = keras.models.Model([x_in, v_in], ivs)
