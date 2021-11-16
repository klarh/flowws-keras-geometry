import flowws
from flowws import Argument as Arg
import numpy as np
from tensorflow import keras

from .internal import VectorAttention, NeighborDistanceNormalization

@flowws.add_stage_arguments
class CrystalStructureClassification(flowws.Stage):
    """Build a geometric attention network for the structure identification task.

    This module specifies the architecture of a network to classify
    local environments of crystal structures in a rotation-invariant
    manner.

    """

    ARGS = [
        Arg('rank', None, int, 2,
            help='Degree of correlations (n-vectors) to consider'),
        Arg('n_dim', '-n', int, 32,
            help='Working dimensionality of point representations'),
        Arg('dilation', None, float, 2,
            help='Working dimension dilation factor for MLP components'),
        Arg('merge_fun', '-m', str, 'mean',
            help='Method to merge point representations'),
        Arg('join_fun', '-j', str, 'mean',
            help='Method to join invariant and point representations'),
        Arg('dropout', '-d', float, 0,
            help='Dropout rate to use, if any'),
        Arg('n_blocks', '-b', int, 2,
            help='Number of deep blocks to use'),
        Arg('block_nonlinearity', None, bool, True,
            help='If True, add a nonlinearity to the end of each block'),
        Arg('residual', '-r', bool, True,
            help='If True, use residual connections within blocks'),
        Arg('activation', '-a', str, 'relu',
            help='Activation function to use inside the network'),
        Arg('final_activation', None, str, 'relu',
            help='Final activation function to use within the network'),
        Arg('scale_invariant', None, str,
            help='Make model scale-invariant by normalizing point clouds by the given distance (one of "min", "mean")'),
    ]

    def run(self, scope, storage):
        n_dim = self.arguments['n_dim']
        dilation_dim = int(np.round(n_dim*self.arguments['dilation']))

        def make_scorefun():
            layers = [
                keras.layers.Dense(dilation_dim),
                keras.layers.Activation(self.arguments['activation'])
            ]

            if self.arguments.get('dropout', 0):
                layers.append(keras.layers.Dropout(self.arguments['dropout']))

            layers.append(keras.layers.Dense(1))
            return keras.models.Sequential(layers)

        def make_valuefun():
            layers = [
                keras.layers.Dense(dilation_dim),
                keras.layers.LayerNormalization(),
                keras.layers.Activation(self.arguments['activation']),
            ]

            if self.arguments.get('dropout', 0):
                layers.append(keras.layers.Dropout(self.arguments['dropout']))

            layers.append(keras.layers.Dense(n_dim))
            return keras.models.Sequential(layers)

        def make_block(last):
            residual_in = last
            last = VectorAttention(
                make_scorefun(), make_valuefun(), False, rank=self.arguments['rank'],
                join_fun=self.arguments['join_fun'],
                merge_fun=self.arguments['merge_fun'])([x, last])

            if self.arguments['block_nonlinearity']:
                last = make_valuefun()(last)

            if self.arguments['residual']:
                last = last + residual_in

            return last

        if 'x_train' in scope:
            (xs, ts) = scope['x_train']
        elif 'train_generator' in scope:
            sample_batch = next(scope['train_generator'])
            ((xs, ts), ys) = sample_batch
        else:
            raise NotImplementedError()

        x_in = keras.layers.Input(xs[0].shape)
        v_in = keras.layers.Input(ts[0].shape)

        if self.arguments.get('scale_invariant', None):
            x = NeighborDistanceNormalization(mode=self.arguments['scale_invariant'])(x_in)
        else:
            x = x_in

        last = keras.layers.Dense(n_dim)(v_in)
        for _ in range(self.arguments['n_blocks']):
            last = make_block(last)

        (last, ivs, att) = VectorAttention(
            make_scorefun(), make_valuefun(), True, name='final_attention',
            rank=self.arguments['rank'], join_fun=self.arguments['join_fun'],
            merge_fun=self.arguments['merge_fun'])(
            [x, last], return_invariants=True, return_attention=True)
        last = keras.layers.Dense(dilation_dim, name='final_mlp')(last)
        if self.arguments.get('dropout', 0):
            last = keras.layers.Dropout(self.arguments['dropout'])(last)
        last = keras.layers.Activation(self.arguments['final_activation'])(last)
        last = keras.layers.Dense(scope['num_classes'], activation='softmax')(last)

        scope['input_symbol'] = [x_in, v_in]
        scope['output'] = last
        scope['loss'] = 'sparse_categorical_crossentropy'
        scope['attention_model'] = keras.models.Model([x_in, v_in], att)
        scope['invariant_model'] = keras.models.Model([x_in, v_in], ivs)
        scope.setdefault('metrics', []).append('accuracy')
