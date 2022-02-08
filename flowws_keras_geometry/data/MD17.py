import os
import subprocess

import flowws
from flowws import Argument as Arg
import numpy as np
from tensorflow import keras

from .internal import ScaledMSE, ScaledMAE

@flowws.add_stage_arguments
class MD17(flowws.Stage):
    """Load data from the MD17 dataset for molecular force regression.

    This module downloads files from the MD17 dataset as required. It
    randomly selects molecule snapshots for the specified molecules to
    produce training, testing, and validation sets.

    For benzene in particular, the "old" benzene calculation of the
    original MD17 dataset is used.

    """

    ARGS = [
        Arg('n_train', '-n', int, 1000,
            help='Number of frames to take for training'),
        Arg('n_val', None, int, 1000,
            help='Number of frames to take for validation'),
        Arg('n_test', None, int, 1000,
            help='Number of frames to take for testing'),
        Arg('cache_dir', '-c', str, '/tmp/md17',
            help='Directory to store trajectory data'),
        Arg('molecules', '-m', [str], [],
            help='List of molecules to use'),
        Arg('seed', '-s', int, 13,
            help='Random number seed to use'),
        Arg('units', None, str, 'meV',
            help='Energy units to use (meV or kcal/mol)'),
        Arg('y_scale_reduction', None, float, 16,
            help='Factor by which to scale forces for training purposes'),
        Arg('x_scale_reduction', None, float,
            help='Factor by which to scale input distances for training purposes'
            ' (negative to auto-scale from training data)')
    ]

    def run(self, scope, storage):
        np.random.seed(self.arguments['seed'])

        energy_conversion = 1.
        if self.arguments['units'] == 'meV':
            energy_conversion = 43.36
        elif self.arguments['units'] == 'kcal/mol':
            pass
        else:
            raise NotImplementedError(self.arguments['units'])

        loaded_files = {}
        train_indices, val_indices, test_indices = {}, {}, {}
        N_train = self.arguments['n_train']
        N_val = self.arguments['n_val']
        N_test = self.arguments['n_test']

        max_atoms = 0
        seen_types = set()

        for name in self.arguments['molecules']:
            fname = self._download(name)
            loaded_files[fname] = data = np.load(fname, mmap_mode='r')
            (frames, size, types) = self.get_trajectory_size_types(data)
            max_atoms = max(max_atoms, size)

            indices = np.arange(frames)
            np.random.shuffle(indices)
            train_indices[fname] = indices[:N_train]
            val_indices[fname] = indices[N_train:N_train + N_val]
            test_indices[fname] = indices[N_train + N_val:N_train + N_val + N_test]
            seen_types.update(types)

        all_types = [0] + list(sorted(seen_types))
        type_map = {t: i for (i, t) in enumerate(all_types)}
        num_types = len(all_types)

        datasets = {}
        for name in ['train', 'val', 'test']:
            dset_xs, dset_ts, dset_ys = [], [], []
            for fname in sorted(loaded_files):
                indices = locals()['{}_indices'.format(name)]
                (xs, ts), Us = self.get_encoding(
                    loaded_files[fname], max_atoms, type_map, indices[fname], energy_conversion)
                dset_xs.append(xs)
                dset_ts.append(ts)
                dset_ys.append(Us)

            dset_xs = np.concatenate(dset_xs, axis=0)
            dset_ts = np.concatenate(dset_ts, axis=0)
            dset_ys = np.concatenate(dset_ys, axis=0)

            indices = np.arange(len(dset_xs))
            np.random.shuffle(indices)
            dset_xs = dset_xs[indices]
            dset_ts = dset_ts[indices]
            dset_ys = dset_ys[indices]

            datasets[name] = (dset_xs, dset_ts, dset_ys)

        yscale = np.std(datasets['train'][-1])*self.arguments['y_scale_reduction']

        for (_, _, y) in datasets.values():
            y /= yscale

        scaled_mse = ScaledMSE(yscale)
        scaled_mae = ScaledMAE(yscale)

        xscale = 1.
        if 'x_scale_reduction' in self.arguments:
            xscale = self.arguments['x_scale_reduction']

            if xscale <= 0:
                delta = datasets['train'][0] - datasets['train'][0][:, :1]
                delta = delta.reshape((-1, 3))
                filt = np.logical_and(
                    np.any(datasets['train'][0].reshape((-1, 3)) != 0, axis=-1),
                    np.any(delta != 0, axis=-1))
                delta = np.linalg.norm(delta[filt], axis=-1)
                xscale = np.std(delta)

            for (x, _, _) in datasets.values():
                x /= xscale

        scope['y_scale'] = yscale
        scope['x_scale'] = xscale
        scope['neighborhood_size'] = max_atoms
        scope['num_types'] = num_types
        scope['x_train'] = datasets['train'][:2]
        scope['y_train'] = datasets['train'][-1]
        scope['x_test'] = datasets['test'][:2]
        scope['y_test'] = datasets['test'][-1]
        scope['validation_data'] = (datasets['val'][:2], datasets['val'][-1])
        scope['type_map'] = type_map
        scope.setdefault('metrics', []).extend([scaled_mse, scaled_mae])

    def _download(self, name):
        url = self.get_url(name)
        fname = url.split('/')[-1]

        if fname not in os.listdir(self.arguments['cache_dir']):
            command = ['wget', '-c', '-P', self.arguments['cache_dir'], url]
            subprocess.check_call(command)

        return os.path.join(self.arguments['cache_dir'], fname)

    @staticmethod
    def get_encoding(data, max_atoms, type_map, indices=None, energy_conversion=1.):
        coords = data['R']
        forces = data['F']*energy_conversion
        types = np.zeros(max_atoms, dtype=np.uint32)
        types[:coords.shape[1]] = [type_map[t] for t in data['z']]

        if indices is not None:
            coords = coords[indices]
            forces = forces[indices]

        rs = np.zeros((len(coords), max_atoms, 3))
        rs[:, :coords.shape[1], :] = coords

        Fs = np.zeros((len(coords), max_atoms, 3))
        Fs[:, :coords.shape[1], :] = forces

        types_onehot = np.eye(len(type_map))[types]
        types_onehot = np.tile(types_onehot[np.newaxis, ...], (len(coords), 1, 1))
        return (rs, types_onehot), Fs

    @staticmethod
    def get_trajectory_size_types(data):
        (frames, size, _) = data['R'].shape
        return (frames, size, data['z'])

    @staticmethod
    def get_url(name):
        remap = dict(benzene='benzene2017_dft.npz')

        base = 'http://quantum-machine.org/gdml/data/npz/'

        return base + remap.get(name, '{}_dft.npz'.format(name.split('_')[0]))
