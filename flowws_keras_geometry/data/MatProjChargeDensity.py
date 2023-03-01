import collections
import json

import flowws
from flowws import Argument as Arg
import freud
import gtar
import numpy as np
import pyriodic
import rowan

from .internal import DataMixingPool

ChargeDensityRecord = collections.namedtuple('ChargeDensityRecord',
    ['grid', 'box_matrix', 'point_density', 'positions', 'types', 'type_names'])

def _fix_box(box_matrix, dtype=np.float32):
    """Convert box into a right-handed coordinate frame with
    only upper triangular entries. Return a new box, rotation
    quaternion, and transformation function for coordinates."""
    # First use QR decomposition to compute the new basis
    Q, R = np.linalg.qr(box_matrix)
    Q = Q.astype(dtype)
    R = R.astype(dtype)

    if (not np.allclose(Q, np.eye(3)) or np.any(np.diag(R) < 0)):
        # Since we'll be performing a quaternion operation,
        # we have to ensure that Q is a pure rotation
        sign = np.linalg.det(Q)
        Q = Q*sign
        R = R*sign

        quat = rowan.from_matrix(Q.T)

        # Now we have to ensure that the box is right-handed. We
        # do this as a second step to avoid introducing reflections
        # into the rotation matrix before making the quaternion
        signs = np.diag(np.diag(np.where(R < 0, -np.ones(R.shape), np.ones(R.shape))))
        box = R.dot(signs)
    else:
        quat = np.asarray([1., 0, 0, 0])
        signs = np.eye(3)
        box = box_matrix

    # Construct the box
    Lx, Ly, Lz = np.diag(box).flatten().tolist()
    xy = box[0, 1]/Ly
    xz = box[0, 2]/Lz
    yz = box[1, 2]/Lz
    box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, xy=xy, xz=xz, yz=yz)

    transform = lambda r, Q=Q, signs=signs: r.dot(Q).dot(signs)
    return box, quat, transform

@flowws.add_stage_arguments
class MatProjChargeDensity(flowws.Stage):
    """Read charge density data from a Materials Project archive.

    This module loads a previously-saved getar archive of materials
    project data (using :py:class:`ImportMatProjChargeDensity`) and
    generates point clouds for charge density regression.
    """

    ARGS = [
        Arg('filename', '-f', str, 'materials_project_charge.sqlite',
            help='Filename of imported data to load'),
        Arg('batch_size', '-b', int, 16,
            help='Number of point clouds in each batch'),
        Arg('pool_size', '-p', int, 512,
            help='Number of batches to mix in memory pools'),
        Arg('num_neighbors', '-n', int, 16,
            help='Number of atoms to use for charge density calculation'),
        Arg('seed', '-s', int, 13,
            help='RNG seed for dataset generation'),
        Arg('validation_split', '-v', float, .3,
            help='Fraction of records to use for validation'),
        Arg('samples_per_system', None, int, 1024,
            help='Number of points to draw for each system'),
        Arg('no_keras', '-k', bool, False,
            help='If True, don\'t load/import keras'),
    ]

    def run(self, scope, storage):
        freud.parallel.set_num_threads(1)
        rng = np.random.default_rng(self.arguments['seed'])

        traj = gtar.GTAR(self.arguments['filename'], 'r')

        all_type_names = traj.readStr('all_type_names.json')
        all_type_names = json.loads(all_type_names)
        type_name_map = {t: i + 1 for (i, t) in enumerate(all_type_names)}

        density_scale = traj.readPath('density_scale.f32.uni')

        all_groups = list(sorted(
            {rec.getGroup() for rec in traj.getRecordTypes() if rec.getGroup()}))
        group_indices = rng.permutation(len(all_groups))

        N_val = int(self.arguments['validation_split']*len(group_indices))
        train_groups = [all_groups[i] for i in group_indices[N_val:]]
        val_groups = [all_groups[i] for i in group_indices[:N_val]]

        train_pool = DataMixingPool(self.arguments['pool_size'], self.arguments['batch_size'])
        train_gen_ = self.dataset_generator(
            traj, train_groups, rng.integers(0, 2**32), type_name_map,
            self.arguments['samples_per_system'], self.arguments['num_neighbors'],
            density_scale)
        train_gen = map(
            self.collate_batch, train_pool.sample(train_gen_, rng.integers(0, 2**32)))
        val_pool = DataMixingPool(self.arguments['pool_size'], self.arguments['batch_size'])
        val_gen_ = self.dataset_generator(
            traj, val_groups, rng.integers(0, 2**32), type_name_map,
            self.arguments['samples_per_system'], self.arguments['num_neighbors'],
            density_scale)
        val_gen = map(
            self.collate_batch, val_pool.sample(val_gen_, rng.integers(0, 2**32)))

        if not self.arguments['no_keras']:
            from .internal import ScaledMSE, ScaledMAE
            metrics = scope.setdefault('metrics', [])
            scaled_mae = ScaledMAE(density_scale, name='scaled_mae')
            metrics.append(scaled_mae)

        scope['loss'] = 'mse'
        scope['train_generator'] = train_gen
        scope['validation_generator'] = val_gen
        scope['type_map'] = dict(type_name_map)
        scope['max_types'] = len(type_name_map) + 1
        scope['type_embedding_size'] = scope['max_types']

    @staticmethod
    def collate_batch(batch):
        rijs, tjs, ys = [], [], []
        for (rij, tj, y) in batch:
            rijs.append(rij)
            tjs.append(tj)
            ys.append(y)

        return (np.array(rijs), np.array(tjs)), np.array(ys)

    @classmethod
    def dataset_generator(cls, traj, groups, seed, type_name_map,
                          samples_per_system, num_neighbors, density_scale):
        rng = np.random.default_rng(seed)
        query_args = dict(
            mode='nearest', num_neighbors=num_neighbors, exclude_ii=True)

        while True:
            rng.shuffle(groups)
            for group in groups:
                record = cls.read_gtar_entry(traj, group)
                (box, coords, positions) = cls.translate_to_freud(
                    record.grid, record.box_matrix, record.positions)
                boxarr = [box.Lx, box.Ly, box.Lz, box.xy, box.xz, box.yz]
                struc = pyriodic.Structure(positions, record.types, boxarr)
                Nrep = 64//2
                nlist = []

                while len(nlist) < samples_per_system*num_neighbors:
                    Nrep *= 2
                    struc = struc.replicate_upto(Nrep)

                    coord_idx = rng.permutation(len(record.point_density))
                    coord_idx = coord_idx[:samples_per_system]
                    selected_density = record.point_density[coord_idx]/density_scale
                    selected_coords = coords[coord_idx]

                    aabb = freud.locality.AABBQuery(struc.box, struc.positions)
                    nq = aabb.query(selected_coords, query_args)
                    nlist = nq.toNeighborList()

                ri = selected_coords[nlist.query_point_indices]
                rj = struc.positions[nlist.point_indices]
                rij = box.wrap(rj - ri)

                type_names = record.type_names
                tj = [type_name_map.get(type_names[t], 0)
                      for t in struc.types[nlist.point_indices]]
                tj = np.eye(len(type_name_map) + 1)[tj]

                rij = rij.reshape((samples_per_system, num_neighbors, 3))
                tj = tj.reshape((samples_per_system, num_neighbors, -1))
                selected_density = selected_density.reshape(
                    (samples_per_system, 1))

                yield rij, tj, selected_density

    @staticmethod
    def read_gtar_entry(traj, group):
        grid = traj.readPath('{}/grid.u32.uni'.format(group))
        box_matrix = traj.readPath('{}/box_matrix.f32.uni'.format(group)).reshape((3, 3))
        point_density = traj.readPath('{}/point_density.f32.uni'.format(group))
        positions = traj.readPath('{}/position.f32.ind'.format(group))
        types = traj.readPath('{}/type.u32.ind'.format(group))
        type_names = traj.readPath('{}/type_names.json'.format(group))

        return ChargeDensityRecord(
            grid, box_matrix, point_density, positions, types, type_names)

    @staticmethod
    def translate_to_freud(grid, box_matrix, positions):
        (box, quat, fix_transform) = _fix_box(box_matrix)

        grid_x, grid_y, grid_z = [np.linspace(0, 1, N) for N in grid]
        frac_coords = np.concatenate(
            [v[..., None] for v in np.meshgrid(
                grid_x, grid_y, grid_z)], axis=-1)
        frac_coords = np.transpose(frac_coords, (1, 0, 2, 3)).reshape((-1, 3))
        coords = box.wrap(fix_transform(np.dot(frac_coords, box_matrix.T)))

        positions = box.wrap(fix_transform(positions))

        return (box, coords, positions)
