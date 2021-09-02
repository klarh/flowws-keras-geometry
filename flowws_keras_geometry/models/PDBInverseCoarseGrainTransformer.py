
import flowws
from flowws import Argument as Arg
import numpy as np
from tensorflow import keras

def convert_generator(gen):
    for (r, v, cv), y in gen:
        rv = np.concatenate([r, v], axis=-1)
        yield (rv, cv), y

@flowws.add_stage_arguments
class PDBInverseCoarseGrainTransformer(flowws.Stage):
    """Build a transformer-based (naive, non rotation-equivariant) neural network for a coarse-grain backmapping task.

    This module specifies the architecture of a network to produce
    atomic coordinates from a set of coarse-grained beads.

    """

    ARGS = [
        Arg('n_dim', '-n', int, 32,
            help='Working dimensionality of point representations'),
        Arg('dilation', None, float, 2,
            help='Working dimension dilation factor for MLP components'),
        Arg('n_blocks_coarse', None, int, 2,
            help='Number of deep blocks to use in the coarse-grain space'),
        Arg('n_blocks_fine', None, int, 2,
            help='Number of deep blocks to use in the coarse-grain space'),
        Arg('block_nonlinearity', None, bool, True,
            help='If True, add a nonlinearity to the end of each block'),
        Arg('residual', '-r', bool, True,
            help='If True, use residual connections within blocks'),
        Arg('activation', '-a', str, 'swish',
            help='Activation function to use inside the network'),
    ]

    def run(self, scope, storage):
        n_dim = self.arguments['n_dim']

        for name in ['train', 'validation', 'test']:
            key = '{}_generator'.format(name)
            if key in scope:
                scope[key] = convert_generator(scope[key])

        train_data = scope['train_generator']
        sample_batch = next(train_data)

        rv_in = keras.layers.Input(sample_batch[0][0].shape[1:], name='rv')
        cv_in = keras.layers.Input(sample_batch[0][1].shape[1:], name='child_t')

        cv_emb = keras.layers.Embedding(len(scope['child_type_names']), n_dim, mask_zero=True)(cv_in)

        dilation_dim = round(n_dim*self.arguments['dilation'])

        def make_block(right, last=None, residual=False):
            if last is None:
                last = right

            residual_in = last
            last = keras.layers.Attention()([right, last, last])

            if self.arguments['block_nonlinearity']:
                last = keras.layers.Dense(dilation_dim)(last)
                last = keras.layers.Activation(self.arguments['activation'])(last)
                last = keras.layers.Dense(n_dim)(last)

            if residual:
                last = last + residual_in

            return last

        last = keras.layers.Dense(n_dim)(rv_in)
        for _ in range(self.arguments['n_blocks_coarse']):
            last = make_block(last, residual=self.arguments['residual'])

        last = make_block(cv_emb, last, False)

        for _ in range(self.arguments['n_blocks_fine']):
            last = make_block(last, residual=self.arguments['residual'])

        last = keras.layers.Dense(dilation_dim)(last)
        last = keras.layers.Activation(self.arguments['activation'])(last)
        vec = keras.layers.Dense(3)(last)

        scope['input_symbol'] = [rv_in, cv_in]
        scope['output'] = vec
        scope['loss'] = 'mse'
