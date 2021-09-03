import collections
import random

import flowws
from flowws import Argument as Arg
import freud
import numpy as np

from .internal import ScaledMSE, ScaledMAE

CoarseSystem = collections.namedtuple('CoarseSystem',
   ['box', 'nlist', 'positions', 'types', 'type_names',
    'child_positions', 'child_types', 'child_type_names'])

def coarse_grain(record, num_neighbors=4, x_scale=1.):
    positions = []
    types = []
    child_positions = []
    child_types = []
    index_groups = np.split(np.arange(len(record.residue_ids)),
                            np.unique(record.residue_ids, True)[1])[1:]
    for group in index_groups:
        group_child_positions = record.positions[group]/x_scale
        group_child_types = record.types[group]
        center_of_mass = np.mean(group_child_positions, axis=0)
        positions.append(center_of_mass)
        types.append(record.residue_types[group[0]])
        child_positions.append(group_child_positions)
        child_types.append(group_child_types)

    positions = np.array(positions, dtype=np.float32)
    types = np.array(types, dtype=np.uint32)
    box = np.array(record.box, dtype=np.float32)/x_scale

    aabb = freud.locality.AABBQuery(box, positions)
    query = aabb.query(positions, dict(num_neighbors=num_neighbors, exclude_ii=True, mode='nearest'))
    nlist = query.toNeighborList()

    return CoarseSystem(
        box, nlist, positions, types, record.residue_type_names,
        child_positions, child_types, record.type_names)

def loop_neighborhood_environments(
        rec, neighborhood_size, seed=13, fraction_range=(0, 2.),
        randomly_rotate=False):

    rand = np.random.default_rng(seed)
    index_i = rec.nlist.query_point_indices
    index_j = rec.nlist.point_indices

    shuffle_indices = np.arange(len(rec.positions))
    fraction_assignments = np.linspace(0, 1, len(rec.positions), endpoint=False)
    rand.shuffle(fraction_assignments)
    filt = np.logical_and(fraction_range[0] <= fraction_assignments,
                          fraction_assignments < fraction_range[1])
    if not np.sum(filt):
        raise ValueError(
            'No particles found for fraction_range: {}'.format(fraction_range))
    shuffle_indices = shuffle_indices[filt]

    while True:
        rand.shuffle(shuffle_indices)

        for i in shuffle_indices:
            bond_start = rec.nlist.find_first_index(i)
            bond_stop = rec.nlist.find_first_index(i + 1)
            bonds = slice(bond_start, bond_stop)
            r0 = rec.positions[index_i[bond_start]]
            rij = (rec.positions[index_j[bonds]] - rec.positions[index_i[bonds]])

            types_j = rec.types[index_j[bonds]]
            types_i = rec.types[index_i[bonds]]

            rchildren = rec.child_positions[i] - r0
            tchildren = rec.child_types[i]

            if randomly_rotate:
                import rowan
                q = rowan.random.rand(1)[None]
                rij = rowan.rotate(q, rij)
                rchildren = rowan.rotate(q, rchildren)

            yield rij, types_i, types_j, rchildren, tchildren

def randomly_loop_iter(xs, seed):
    rand = random.Random(seed)
    xs = list(xs)
    while True:
        rand.shuffle(xs)
        yield from xs

def make_batches(cg_records, batch_size, neighborhood_size,
                 max_atoms, max_types, global_type_remaps, y_scale=1.,
                 fraction_range=(0, 2.), seed=13, randomly_rotate=False):
    rand = random.Random(seed)
    name_iter = randomly_loop_iter(sorted(cg_records), rand.randint(0, 2**32))

    iterators = {}
    for (name, rec) in sorted(cg_records.items()):
        iterators[name] = loop_neighborhood_environments(
            rec, neighborhood_size, seed + 1, fraction_range,
            randomly_rotate=randomly_rotate)

    while True:
        cg_rij = np.zeros((batch_size, neighborhood_size, 3), dtype=np.float32)
        cg_tij = np.zeros((batch_size, neighborhood_size, 2*max_types), dtype=np.float32)
        fg_tchild = np.zeros((batch_size, max_atoms), dtype=np.uint32)
        fg_rchild = np.zeros((batch_size, max_atoms, 3), dtype=np.float32)

        for batch_i in range(batch_size):
            name = next(name_iter)
            (res_type_remap, atom_type_remap) = global_type_remaps[name]
            (rij, types_i, types_j, rchildren, tchildren) = next(iterators[name])

            types_i, types_j = res_type_remap[types_i], res_type_remap[types_j]
            types_i = np.eye(max_types)[types_i]
            types_j = np.eye(max_types)[types_j]

            cg_rij[batch_i, :len(rij)] = rij
            cg_tij[batch_i, :len(rij), :max_types] = types_j + types_i
            cg_tij[batch_i, :len(rij), max_types:] = types_j - types_i
            fg_tchild[batch_i, :len(rchildren)] = atom_type_remap[tchildren]
            fg_rchild[batch_i, :len(rchildren)] = rchildren/y_scale

        yield (cg_rij, cg_tij, fg_tchild), fg_rchild

@flowws.add_stage_arguments
class PDBCoarseGrained(flowws.Stage):
    """Generate data for backmapping coarse-graining of structures from the PDB.

    This module calculates coarse-grained versions of PDB records,
    loaded using `PDBCache`. Generators are produced for model
    training.

    """

    ARGS = [
        Arg('neighborhood_size', '-n', int,
           help='Neighborhood size (number of input amino acid coordinates) to use'),
        Arg('batch_size', '-b', int, 32,
           help='Batch size to use'),
        Arg('seed', '-s', int, 14,
           help='Random seed to use'),
        Arg('validation_fraction', '-v', float, .3,
           help='Fraction of record names to be assigned to validation set'),
        Arg('test_fraction', '-t', float,
           help='Fraction of record names to be assigned to validation set'),
        Arg('x_scale', '-x', float, 64.,
           help='Factor by which to decrease residue length scales'),
        Arg('y_scale', '-y', float, 8.,
           help='Factor by which to decrease atomic length scales'),
        Arg('randomly_rotate', '-r', bool, False,
            help='If True, randomly rotate environments'),
    ]

    def run(self, scope, storage):

        all_records = scope['pdb_records']
        coarse_records = {}
        skipped_records = []
        for (name, rec) in all_records.items():
            coarse = coarse_grain(
                rec, self.arguments['neighborhood_size'], self.arguments['x_scale'])

            if any(np.max(np.bincount(ts)) > 1 for ts in coarse.child_types):
                skipped_records.append((name, 'duplicate child types'))
                continue
            if len(coarse.positions) <= self.arguments['neighborhood_size']:
                skipped_records.append((name, 'too few positions'))
                continue
            coarse_records[name] = coarse

        scope['coarse_records'] = coarse_records
        scope['skipped_records'] = skipped_records
        print('{} final records'.format(len(coarse_records)))
        print('{} skipped records'.format(len(skipped_records)))

        max_atoms = 0
        all_residue_types, all_atom_types = set(), set()
        for rec in coarse_records.values():
            all_residue_types.update(rec.type_names)
            all_atom_types.update(rec.child_type_names)
            max_atoms = max(max_atoms, max(len(v) for v in rec.child_positions))

        all_residue_types = ['NORES'] + list(sorted(all_residue_types))
        residue_type_map = {name: i for (i, name) in enumerate(all_residue_types)}
        all_atom_types = ['NOATM'] + list(sorted(all_atom_types))
        atom_type_map = {name: i for (i, name) in enumerate(all_atom_types)}

        global_type_remaps = {}
        for (name, rec) in coarse_records.items():
            res_type_remap = [residue_type_map[name] for name in rec.type_names]
            res_type_remap = np.array(res_type_remap, dtype=np.uint32)
            atom_type_remap = [atom_type_map[name] for name in rec.child_type_names]
            atom_type_remap = np.array(atom_type_remap, dtype=np.uint32)
            global_type_remaps[name] = (res_type_remap, atom_type_remap)

        print('Max number of atoms in a residue:', max_atoms)

        scaled_mse = ScaledMSE(self.arguments['y_scale'])
        scaled_mae = ScaledMAE(self.arguments['y_scale'])
        y_scale = self.arguments['y_scale']/self.arguments['x_scale']

        ranges, labels = [0], []
        if self.arguments['validation_fraction']:
            ranges.append(self.arguments['validation_fraction'])
            labels.append('validation')

        if 'test_fraction' in self.arguments:
            ranges.append(self.arguments['test_fraction'])
            labels.append('test')

        ranges.append(2.)
        labels.append('train')

        cumulative_ranges = np.cumsum(ranges)
        label_ranges = {name: (start, stop) for (name, start, stop)
                        in zip(labels, cumulative_ranges[:-1], cumulative_ranges[1:])}

        for (name, fraction_range) in label_ranges.items():
            scope['{}_generator'.format(name)] = make_batches(
                coarse_records, self.arguments['batch_size'], self.arguments['neighborhood_size'],
                max_atoms, len(all_residue_types), global_type_remaps, y_scale,
                fraction_range, self.arguments['seed'],
                randomly_rotate=self.arguments['randomly_rotate'])

        if 'validation_generator' not in scope:
            scope['validation_generator'] = scope['train_generator']

        scope['x_scale'] = self.arguments['x_scale']
        scope['y_scale'] = self.arguments['y_scale']
        scope['type_names'] = all_residue_types
        scope['type_name_map'] = residue_type_map
        scope['child_type_names'] = all_atom_types
        scope['child_type_name_map'] = atom_type_map
        scope.setdefault('metrics', []).extend([scaled_mse, scaled_mae])
