import collections
import os
import subprocess

import flowws
from flowws import Argument as Arg
import numpy as np

PDBRecord = collections.namedtuple('PDBRecord',
    ['box', 'positions', 'types', 'type_names', 'residue_ids', 'residue_types', 'residue_type_names'])
#            (N, 3),       (N,)     (N_types,)     (N,)          (N_res)           (N_res,)

def parse_pdb(filename, ignore_hydrogens=False):
    cmd = ['gunzip', '-ckf', filename]

    positions = []
    types = []
    residue_ids = []
    residue_types = []

    type_name_map = collections.defaultdict(lambda: len(type_name_map))
    residue_type_name_map = collections.defaultdict(lambda: len(residue_type_name_map))
    res_id = -1

    for line in subprocess.check_output(cmd).decode().splitlines():
        if line.startswith('MODEL'):
            last_residue = (None, None)

        if line.startswith('TER'):
            last_residue = (None, None)

        if not line.startswith('ATOM'):
            continue

        # skip non-blank alternate locations
        if line[16] != ' ':
            continue

        coords = tuple(map(float, (line[start:end] for (start, end) in
                                   [(30, 38), (38, 46), (46, 54)])))

        # special atomic type, like CA/CB/etc
        atype = line[12:16].strip()
        # actual element type
        etype = line[76:78].strip()

        res_id_str = line[22:26].strip()
        res_type = line[17:20].strip()
        current_residue = (res_id_str, res_type)

        res_id += current_residue != last_residue
        last_residue = current_residue

        if (etype == 'H' or etype == 'D') and ignore_hydrogens:
            continue

        positions.append(coords)
        types.append(type_name_map['{};{}'.format(atype, etype)])
        residue_ids.append(res_id)
        residue_types.append(residue_type_name_map[res_type])

    box = (2*np.max(positions, axis=0) - 2*np.min(positions, axis=0)).tolist()
    box.extend([0, 0, 0])

    positions = np.array(positions, dtype=np.float32)
    positions -= np.mean(positions, axis=0, keepdims=True)

    types = np.array(types, dtype=np.uint32)
    type_names = [v for (_, v) in sorted((v, i) for (i, v) in type_name_map.items())]

    residue_ids = np.array(residue_ids, dtype=np.int32)
    residue_types = np.array(residue_types, dtype=np.uint32)
    residue_type_names = [v for (_, v) in sorted((v, i) for (i, v) in residue_type_name_map.items())]

    return PDBRecord(box, positions, types, type_names, residue_ids, residue_types, residue_type_names)

@flowws.add_stage_arguments
class PDBCache(flowws.Stage):
    """Load PDB records and cache them in a local directory.

    This module stores and reads atomic coordinates from a set of
    specified PDB records. Files are cached in a local directory to
    avoid repeated downloads.

    """

    ARGS = [
        Arg('cache_directory', None, str, '/tmp',
           help='Cache directory to use for downloaded PDB records'),
        Arg('records', '-r', [str],
           help='Record IDs to use for training (or file://path/to/filename.txt'
            ' to read lines from a file)'),
        Arg('failure_mode', '-f', str, 'default',
           help='Behavior when parsing/fetching a file fails (default/ignore)'),
        Arg('ignore_hydrogens', None, bool, True,
           help='If True, ignore hydrogens when building systems'),
    ]

    def _maybe_get_pdbfile(self, pdb_id):
        fname = '{}.pdb1.gz'.format(pdb_id)
        target = os.path.join(self.arguments['cache_directory'], fname)

        if not os.path.exists(target):
            url = 'https://files.rcsb.org/download/{}'.format(fname)
            cmd = ['wget', url, '-O', target]
            subprocess.check_call(cmd)

        return target

    def run(self, scope, storage):
        record_names = []
        for name in self.arguments['records']:
            if name.startswith('file://'):
                with storage.open(name[len('file://'):], 'r') as f:
                    for line in f.splitlines():
                        record_names.append(line.strip())
            else:
                record_names.append(name)

        parse_kwargs = dict(ignore_hydrogens=self.arguments['ignore_hydrogens'])

        all_records = {}
        for pdb_id in record_names:
            if self.arguments['failure_mode'] in ('ignore', 'print'):
                try:
                    fname = self._maybe_get_pdbfile(pdb_id)
                    all_records[pdb_id] = parse_pdb(fname, **parse_kwargs)
                except Exception as e:
                    if self.arguments['failure_mode'] == 'print':
                        print(pdb_id, e)
                    pass
            else:
                fname = self._maybe_get_pdbfile(pdb_id)
                all_records[pdb_id] = parse_pdb(fname, **parse_kwargs)

        scope['pdb_records'] = all_records
