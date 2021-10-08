import collections

import flowws
from flowws import Argument as Arg
import freud
import numpy as np
import pyriodic

def encode_types(source_types, dest_types, N, max_types):
    onehot_src = np.eye(max_types)[source_types]
    onehot_dest = np.eye(max_types)[dest_types]

    minus = onehot_dest - onehot_src
    minus = minus.reshape((-1, N, max_types))
    plus = onehot_dest + onehot_src
    plus = plus.reshape((-1, N, max_types))

    return np.concatenate([minus, plus], axis=-1)

@flowws.add_stage_arguments
class PyriodicDataset(flowws.Stage):
    """Load crystal structures from `pyriodic` for training data.

    This module takes a specified set of structures from the
    `pyriodic` default database, replicates them up to the given size,
    applies Gaussian noise to each structure one or more times, and
    extracts local environments of these structures for training data.

    """

    ARGS = [
        Arg('num_neighbors', '-n', int, 12,
           help='Number of nearest neighbors to use'),
        Arg('structures', '-s', [str],
           help='Name of structures to take'),
        Arg('size', None, int, 512,
           help='Number of particles to replicate structures up to'),
        Arg('noise', None, [float], [1e-2, 5e-2, .1],
           help='Noise standard deviation to apply to structures'),
        Arg('test_fraction', '-t', float, 0,
           help='Fraction of data to hold back as test data'),
        Arg('seed', None, int, 13,
           help='Random seed to use for shuffling training data'),
    ]

    def run(self, scope, storage):
        np.random.seed(self.arguments['seed'])

        xs = []
        ts = []
        ys = []

        name_map = collections.defaultdict(lambda: len(name_map))

        structures = list(self.arguments['structures'])
        max_types = 0
        for name in structures:
            structure = None
            for (structure,) in pyriodic.db.query('select structure from unit_cells where name = ?', (name,)):
                pass
            if structure is None:
                raise ValueError('Structure {} not found'.format(name))
            max_types = max(max_types, len(set(structure.types)))

        for name in structures:
            for (structure,) in pyriodic.db.query('select structure from unit_cells where name = ?', (name,)):
                pass

            if name in name_map:
                continue

            for noise in self.arguments['noise']:
                structure = structure.rescale_shortest_distance(1)
                structure = structure.replicate_upto(self.arguments['size'])
                structure = structure.add_gaussian_noise(noise)

                q = freud.locality.AABBQuery(structure.box, structure.positions)
                qr = q.query(
                    structure.positions, dict(
                        num_neighbors=self.arguments['num_neighbors'], exclude_ii=True))
                nl = qr.toNeighborList()
                rijs = (structure.positions[nl.point_indices] -
                        structure.positions[nl.query_point_indices])
                fbox = freud.box.Box(*structure.box)
                rijs = fbox.wrap(rijs)
                rijs = rijs.reshape(
                    (len(structure), self.arguments['num_neighbors'], 3))

                tijs = encode_types(
                    structure.types[nl.query_point_indices],
                    structure.types[nl.point_indices],
                    self.arguments['num_neighbors'], max_types)

                shuf = np.arange(len(rijs))
                np.random.shuffle(shuf)
                shuf = shuf[:self.arguments['size']]

                xs.append(rijs[shuf])
                ts.append(tijs[shuf])
                ys.append(name_map[name])

        ys = np.repeat(ys, [len(v) for v in xs])
        xs = np.concatenate(xs, axis=0)
        ts = np.concatenate(ts, axis=0)

        shuf = np.arange(len(xs))
        np.random.shuffle(shuf)
        N_test = int(self.arguments['test_fraction']*len(shuf))
        test_split, train_split = shuf[:N_test], shuf[N_test:]
        xs_test = xs[test_split]
        ts_test = ts[test_split]
        ys_test = ys[test_split]
        xs = xs[train_split]
        ts = ts[train_split]
        ys = ys[train_split]

        scope['x_train'] = (xs, ts)
        scope['y_train'] = ys
        scope['x_test'] = (xs_test, ts_test)
        scope['y_test'] = ys_test
        scope['num_classes'] = len(name_map)
        scope['type_map'] = dict(name_map)
        scope['neighborhood_size'] = self.arguments['num_neighbors']
