import collections
import itertools
import functools
import json

import flowws
from flowws import Argument as Arg
import keras_gtar
import numpy as np

LoadResult = collections.namedtuple(
    'LoadResult', ['att_model', 'train_data', 'val_data', 'batch_size', 'type_map'])

@functools.lru_cache(maxsize=1)
def load_model(fname, path_substitutions):
    path_substitutions = dict(path_substitutions)

    with keras_gtar.Trajectory(fname, 'r') as traj:
        weights = traj.get_weights()
        workflow_json = traj.handle.readStr('workflow.json')

    w = flowws.Workflow.from_JSON(json.loads(workflow_json))
    stages = []
    for stage in w.stages:
        stage_json = stage.to_JSON()
        stages.append(stage)

        for (name, val) in stage_json['arguments'].items():
            if isinstance(val, str):
                stage.arguments[name] = path_substitutions.get(
                    stage.arguments[name], stage.arguments[name])

        if stage_json['type'] == 'Train':
            stage.arguments['epochs'] = 0
            break
    new_w = flowws.Workflow(stages)
    scope = new_w.run()
    model = scope['model']
    model.set_weights(weights)

    batch_size = new_w.stages[-1].arguments['batch_size']
    train_xs, train_ts = scope['x_train']
    type_map = scope['type_map']
    attention_model = scope['attention_model']

    return LoadResult(
        attention_model, (train_xs, train_ts), scope['validation_data'],
        batch_size, type_map)

class MoleculeAttentionViewer(flowws.Stage):
    ARGS = [
        Arg('filename', '-f', str,
            help='Saved model to open'),
        Arg('path_substitutions', '-p', [(str, str)],
            help='Paths to replace in stage descriptions'),
        Arg('frame', None, int, 0,
            help='Frame (data index) to visualize'),
        Arg('particle', None, int, 0,
            help='Particle index to focus on'),
    ]

    def run(self, scope, storage):
        loaded = load_model(
            self.arguments['filename'], tuple(self.arguments['path_substitutions']))
        self.attention_model = loaded.att_model
        self.train_data = loaded.train_data
        self.val_data = loaded.val_data
        self.batch_size = loaded.batch_size
        type_map = loaded.type_map

        element_colors = {
            1: (.8, .8, .8, 1),
            6: (.5, .5, .5, 1),
            7: (.5, .7, .8, 1),
            8: (.8, .5, .5, 1),
        }

        type_color_map = [element_colors.get(i, (1., 0, 1, 1.)) for (i, j) in sorted(type_map.items())]
        self.type_color_map = np.array(type_color_map)

        self.arg_specifications['frame'].valid_values = flowws.Range(
            0, len(self.train_data[0]), (True, False))
        self.arg_specifications['particle'].valid_values = flowws.Range(
            0, len(self.train_data[0][1]), (True, False))

        scope.setdefault('visuals', []).append(self)

    def draw_plato(self):
        import plato, plato.draw as draw

        train_xs, train_ts = self.train_data
        i, j = self.arguments['frame'], self.arguments['particle']

        features = dict(additive_rendering=True)
        prim = draw.Spheres()
        edges = draw.Lines(cap_mode=1)
        scene = draw.Scene([prim, edges], zoom=3.5, features=features)

        xs = train_xs[i].copy()
        filt = np.any(xs != 0, axis=-1)
        xs = xs[filt]
        xs -= np.mean(xs, axis=0, keepdims=True)

        attention = self.attention_model.predict(
            (train_xs[i:i+1], train_ts[i:i+1]), batch_size=self.batch_size)[0, j]
        start_points = []
        end_points = []
        colors = []
        for (ii, jj) in itertools.product(range(len(xs)), range(len(xs))):
            start_points.append(xs[ii])
            end_points.append(xs[jj])
            colors.append(attention[ii, jj])
        colors = np.array(colors)[:, 0]
        colors *= np.sqrt(attention.shape[0])
        colors = np.clip(colors, 0, 1)
        colors = plato.cmap.cubehelix(colors)
        edges.start_points = start_points
        edges.end_points = end_points
        edges.widths = .125
        edges.colors = colors

        prim.positions = xs
        colors = self.type_color_map[np.argmax(train_ts[i, :len(xs)], axis=-1)]
        prim.colors = colors
        prim.diameters = 1

        return scene
