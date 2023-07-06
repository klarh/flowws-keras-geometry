import collections
import json
import pickle

import flowws
from flowws import Argument as Arg
import gtar
import lmdb
import mendeleev
import numpy as np

from .MatProjChargeDensity import _fix_box

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(xs):
        for x in xs:
            yield x

FixedFrame = collections.namedtuple(
    'FixedFrame', ['cell', 'pos', 'atomic_numbers', 'tags', 'force', 'y', 'y_relaxed'])

def fix_frame(frame):
    try:
        _ = frame.pos
        return frame
    except RuntimeError: # mismatched pyg version
        framedict = vars(frame)
        args = [framedict.get(key, None) for key in FixedFrame._fields]
        return FixedFrame(*args)

@flowws.add_stage_arguments
class ImportOCP(flowws.Stage):
    ARGS = [
        Arg('lmdb_file', '-l', str,
            help='Path for data LMDB file to load'),
        Arg('gtar_file', '-g', str, 'ocp.sqlite',
            help='Output getar filename to save'),
        Arg('energy_fit_rows', None, int, 64000,
            help='Maximum number of type histograms to fit energy with'),
        Arg('name_adsorbates', None, bool, True,
            help='If True, give adsorbate atoms the "_ads" suffix'),
    ]

    def run(self, scope, storage):
        gtar_path = self.arguments['gtar_file']
        lmdb_path = self.arguments['lmdb_file']
        env = lmdb.open(lmdb_path, readonly=True, subdir=False)

        fit_rows = self.arguments['energy_fit_rows']
        name_adsorbates = self.arguments['name_adsorbates']

        type_names = ['NONE'] + [e.symbol for e in mendeleev.get_all_elements()]
        type_map = collections.defaultdict(lambda: len(type_map))
        _ = [type_map[t] for t in type_names]

        with env.begin() as txn, gtar.GTAR(gtar_path, 'w') as traj:
            try:
                N = pickle.loads(txn.get('length'.encode()))
            except TypeError: # length not saved
                N = env.stat()['entries']

            rows = {}

            for i in tqdm(range(N)):
                frame = pickle.loads(txn.get(str(i).encode()))
                frame = fix_frame(frame)

                try:
                    energy = frame.y
                    if energy is None:
                        energy = frame.y_relaxed
                except AttributeError:
                    energy = frame.y_relaxed

                (fbox, quat, transform) = _fix_box(frame.cell[0].T)
                boxarr = [fbox.Lx, fbox.Ly, fbox.Lz, fbox.xy, fbox.xz, fbox.yz]

                types = np.asarray(frame.atomic_numbers, dtype=np.int32)
                if name_adsorbates:
                    for j in np.where(frame.tags == 2)[0]:
                        types[j] = type_map['{}_ads'.format(type_names[types[j]])]
                traj.writePath('frames/{}/position.f32.ind'.format(i),
                               transform(np.array(frame.pos)))
                traj.writePath('frames/{}/force.f32.ind'.format(i),
                               transform(np.array(frame.force)))
                traj.writePath('frames/{}/type.u32.ind'.format(i), types)
                traj.writePath('frames/{}/energy.f32.uni'.format(i), energy)
                traj.writePath('frames/{}/box.f32.uni'.format(i), boxarr)
                traj.writePath('frames/{}/num_adsorbate_atoms.u32.uni'.format(i),
                               sum(frame.tags == 2))

                fit_x = np.bincount(types)
                rows[np.random.default_rng(i).integers(fit_rows)] = (
                    fit_x, energy)

            fit_x = np.zeros((len(rows), len(type_map) + 1))
            fit_y = np.zeros(len(fit_x))
            for i, (_, (x, y)) in enumerate(sorted(rows.items())):
                fit_x[i, :len(x)] = x
                fit_y[i] = y
            fit_x[:, -1] = 1
            (x, res, rank, s) = np.linalg.lstsq(fit_x, fit_y)
            x[np.sum(fit_x, axis=0) == 0] = 0
            traj.writePath('linear_energy_fit.f32.uni', x)

            inv_type_name_map = {i: n for (n, i) in type_map.items()}
            type_names = [inv_type_name_map[i] for i in range(len(inv_type_name_map))]
            traj.writeStr('type_names.json', json.dumps(type_names))
