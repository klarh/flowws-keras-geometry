import os
import subprocess

import flowws
from flowws import Argument as Arg
import numpy as np
from tensorflow import keras

from .internal import ScaledMSE, ScaledMAE
from .MD17 import MD17

@flowws.add_stage_arguments
class RMD17(MD17):
    """Load data from the RMD17 dataset for molecular force regression.

    This module downloads the entire RMD17 dataset. It randomly
    selects molecule snapshots for the specified molecules to produce
    training, testing, and validation sets.

    """

    def _download(self, name):
        fname = 'rmd17_{}.npz'.format(name.split('_')[0])

        if fname not in os.listdir(self.arguments['cache_dir']):
            url = 'https://ndownloader.figshare.com/files/23950376'
            output_name = os.path.join(self.arguments['cache_dir'], 'RMD17.tar.bz2')
            command = ['wget', '-c', '-O', output_name, url]
            subprocess.check_call(command)

            command = ['tar', 'xf', output_name, '-C', self.arguments['cache_dir'], '--strip', '2']
            subprocess.check_call(command)

        return os.path.join(self.arguments['cache_dir'], fname)

    @staticmethod
    def get_encoding(data, max_atoms, type_map, indices=None, energy_conversion=1.):
        coords = data['coords']
        forces = data['forces']*energy_conversion
        types = np.zeros(max_atoms, dtype=np.uint32)
        types[:coords.shape[1]] = [type_map[t] for t in data['nuclear_charges']]

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
        (frames, size, _) = data['coords'].shape
        return (frames, size, data['nuclear_charges'])
