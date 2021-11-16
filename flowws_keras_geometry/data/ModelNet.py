import itertools
import os
import random
import re
import subprocess

import flowws
from flowws import Argument as Arg
import gtar
import numpy as np

header_pattern = re.compile(r'^OFF\s*(?P<vertex>\d+)\s+(?P<mesh>\d+)\s+\d+\s+')

class GTARDataset:
    def __init__(self, fname, neighborhood_size, validation_fraction=0, seed=13):
        self.handle = gtar.GTAR(fname, 'r')
        self.neighborhood_size = neighborhood_size
        self.validation_fraction = validation_fraction
        self.seed = seed

        group_indices = {}
        group_records = {}
        for rec in self.handle.getRecordTypes():
            if rec.getName() != 'vertex':
                continue
            frames = self.handle.queryFrames(rec)
            group_indices[rec.getGroup()] = frames
            group_records[rec.getGroup()] = rec
        self.group_indices = group_indices
        self.group_records = group_records

        if validation_fraction:
            rng = random.Random(seed)
            for g in list(self.group_indices):
                if 'train' not in g:
                    continue

                index_values = self.group_indices[g]
                indices = list(range(len(index_values)))
                rng.shuffle(indices)
                N = max(1, int((1 - validation_fraction)*len(indices)))

                train_indices = [index_values[i] for i in sorted(indices[:N])]
                val_indices = [index_values[i] for i in sorted(indices[N:])]

                val_g = g.replace('train', 'val')
                if val_indices:
                    self.group_indices[val_g] = val_indices
                self.group_indices[g] = train_indices
                self.group_records[val_g] = self.group_records[g]

        labels = set()
        for group in group_indices:
            label = group.split('/')[0]
            labels.add(label)
        self.labels = list(sorted(labels))
        self.label_map = {l: i for (i, l) in enumerate(self.labels)}
        self.num_labels = len(self.labels)

        self.train_groups = [g for g in sorted(self.group_indices) if '/train' in g]
        self.val_groups = [g for g in sorted(self.group_indices) if '/val' in g]
        self.test_groups = [g for g in sorted(self.group_indices) if '/test' in g]

    def _generator(self, dataset, batch_size, seed=13):
        rng = np.random.default_rng(seed)
        groups = getattr(self, '{}_groups'.format(dataset))

        while True:
            rs = np.zeros((batch_size, self.neighborhood_size, 3))
            vs = np.zeros((batch_size, self.neighborhood_size, 1))
            ys = np.zeros(batch_size, dtype=np.int32)

            for i in range(batch_size):
                y = rng.choice(len(groups))
                group = groups[y]
                indices = self.group_indices[group]
                index = indices[rng.choice(len(indices))]

                vertices = self.handle.getRecord(self.group_records[group], index)
                vertices = vertices.reshape((-1, 3))

                vertices -= np.mean(vertices, axis=0, keepdims=True)
                scale = np.mean(np.linalg.norm(vertices, axis=-1))

                vertex_choice = rng.choice(
                    len(vertices), min(len(vertices), self.neighborhood_size),
                    replace=False)
                rs[i, :len(vertex_choice)] = vertices[vertex_choice]/scale
                vs[i, :len(vertex_choice)] = 1
                ys[i] = y

            yield ((rs, vs), ys)

    def train_generator(self, batch_size, seed=13):
        yield from self._generator('train', batch_size, seed)

    def test_generator(self, batch_size, seed=13):
        yield from self._generator('test', batch_size, seed)

    def val_generator(self, batch_size, seed=13):
        yield from self._generator('val', batch_size, seed)

@flowws.add_stage_arguments
class ModelNet(flowws.Stage):
    """Load the ModelNet40 dataset."""

    ARGS = [
        Arg('cache_dir', '-c', str, '/tmp',
            help='Directory to store dataset'),
        Arg('neighborhood_size', '-n', int, 12,
            help='Number of points to sample for each cloud'),
        Arg('seed', '-s', int, 13,
            help='Random seed for dataset generation'),
        Arg('validation_fraction', '-v', float, 0,
            help='Fraction of training data to use for validation set'),
        Arg('batch_size', '-b', int, 8,
            help='Size of point cloud batches'),
    ]

    def run(self, scope, storage):
        dataset_fname = self._get_dataset(self.arguments['cache_dir'])
        dataset = GTARDataset(
            dataset_fname, self.arguments['neighborhood_size'],
            self.arguments['validation_fraction'], self.arguments['seed'])

        scope['type_names'] = dataset.labels
        scope['type_name_map'] = dataset.label_map
        scope['train_generator'] = dataset.train_generator(
            self.arguments['batch_size'], self.arguments['seed'] + 1)
        scope['test_generator'] = dataset.test_generator(
            self.arguments['batch_size'], self.arguments['seed'] + 2)
        if self.arguments['validation_fraction']:
            scope['validation_generator'] = dataset.val_generator(
                self.arguments['batch_size'], self.arguments['seed'] + 3)
        scope['num_classes'] = len(dataset.labels)

    @classmethod
    def _get_dataset(cls, cache_dir):
        fname = os.path.join(cache_dir, 'modelnet40_gtar.zip')

        if os.path.exists(fname):
            return fname

        url = 'https://modelnet.cs.princeton.edu/ModelNet40.zip'
        cmd = ['wget', '-c', '-P', cache_dir, url]
        subprocess.check_call(cmd)

        source_fname = os.path.join(cache_dir, 'ModelNet40.zip')

        with gtar.GTAR(source_fname, 'r') as src, gtar.GTAR(fname, 'w') as dst:
            for rec in src.getRecordTypes():
                if 'off' not in rec.getName():
                    continue
                for index in src.queryFrames(rec):
                    rec.setIndex(index)
                    cls._translate_record(src, dst, rec)
            pass

        os.remove(source_fname)
        return fname

    @staticmethod
    def _translate_record(src, dst, rec):
        new_group = '/'.join(rec.getGroup().split('/')[1:])
        new_index = rec.getName().split('_')[1].split('.')[0]

        contents = src.getRecord(rec)
        match = header_pattern.match(contents)
        if not match:
            raise ValueError(contents[:32])

        contents = contents[match.end():]
        lines = contents.splitlines()

        # skip "OFF" text
        vertex_count = int(match.group('vertex'))
        mesh_count = int(match.group('mesh'))
        vertices = [list(map(float, line.split())) for line in lines[:vertex_count]]
        mesh_lines = [list(map(int, line.split())) for line in lines[vertex_count:]]
        mesh_stream = list(itertools.chain.from_iterable(mesh_lines))

        assert vertex_count + mesh_count == len(lines)

        new_path = '/'.join([new_group, 'frames', new_index, 'vertex.f32.ind'])
        dst.writePath(new_path, vertices)

        new_path = '/'.join([new_group, 'frames', new_index, 'mesh.u32.uni'])
        dst.writePath(new_path, mesh_stream)
