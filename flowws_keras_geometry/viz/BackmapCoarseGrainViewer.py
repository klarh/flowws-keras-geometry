import collections
import functools
import json

import flowws
from flowws import Argument as Arg
import keras_gtar
import numpy as np

elt_colors = {
    'H': (.8, .8, .8, 1),
    'C': (.5, .5, .5, 1),
    'N': (.5, .7, .8, 1),
    'O': (.8, .5, .5, 1),
    'P': (.8, .7, 0, 1),
    'S': (.8, .8, 0, 1),
}

elt_radii = {
    'H': .25,
    'C': .7,
    'N': .65,
    'O': .6,
    'P': 1.,
    'S': 1.,
}

LoadResult = collections.namedtuple(
    'LoadResult', ['model', 'scope'])

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

    return LoadResult(model, scope)

@flowws.add_stage_arguments
class BackmapCoarseGrainViewer(flowws.Stage):
    """Visualize the results of backmapped coarse-grainings.

    This module creates a plato scene to visualize a coarse-grained
    system and the predicted fine-scale coordinates of a trained
    model. It loads a gtar file containing a trained model and flowws
    workflow definition (using `PDBCoarseGrained`, for example) and
    enables visualization of the coarse- and fine-grained coordinates.

    Workflows run on different systems will likely require use of the
    `path_substitutions` argument for their data-loading modules.

    """

    ARGS = [
        Arg('filename', '-f', str,
            help='Saved model to open'),
        Arg('path_substitutions', '-p', [(str, str)],
            help='Paths to replace in stage descriptions'),
        Arg('additive_rendering', '-a', bool, True,
            help='If True, use additive rendering'),
        Arg('fast_antialiasing', None, bool, False,
            help='Use Fast Approximate Antialiasing (FXAA)'),
        Arg('ambient_occlusion', None, bool, False,
            help='Use Screen Space Ambient Occlusion (SSAO)'),
        Arg('diameter_scale', '-d', float, 1.,
            help='Size scale for atomic diameters'),
        Arg('true_coords', None, bool, 1,
            help='Show atoms at their true coordinates instead of predicted coordinates'),
        Arg('show_amino_acids', None, bool, 1,
            help='Show amino acid center of mass beads'),
    ]

    def run(self, scope, storage):
        import plato, plato.draw as draw

        loaded = load_model(
            self.arguments['filename'], tuple(self.arguments['path_substitutions']))
        self.model = loaded.model
        self.loaded_scope = loaded.scope

        features = {}
        if self.arguments['additive_rendering']:
            features['additive_rendering'] = True

        prot_prim = draw.Spheres()
        atom_prim = draw.Spheres(diameters=.5*self.arguments['diameter_scale'])
        lines = draw.Lines(widths=.25, cap_mode=1)
        self.scene = draw.Scene([prot_prim, lines, atom_prim], zoom=4, features=features)

        named_type_colors = [
            elt_colors.get(name.split(';')[-1], (.5, .5, .5, 1))
            for name in self.loaded_scope['child_type_names']]
        named_type_colors = np.array(named_type_colors)

        named_type_radii = [
            elt_radii.get(name.split(';')[-1], .125)
            for name in self.loaded_scope['child_type_names']]
        named_type_diameters = np.array(named_type_radii)*2*self.arguments['diameter_scale']

        max_types = len(self.loaded_scope['type_names'])

        sample_batch = next(self.loaded_scope['train_generator'])
        ((sx, st, sct), sy) = sample_batch
        types = np.argmax(st[0, max_types:], axis=-1)
        dy = sy[0]
        diff, sum_ = st[0, :, :max_types], st[0, :, max_types:]
        types = np.argmax(sum_ - diff, axis=-1)
        ti, tj = np.argmax(sum_ - diff, axis=-1), np.argmax(sum_ + diff, axis=-1)

        cfilt = sct[0] != 0
        filt = tj != 0

        pred = self.model.predict((sx, st, sct))[0][cfilt]*self.loaded_scope['y_scale']

        coords = np.concatenate([sx[0][filt]*self.loaded_scope['x_scale'], [(0, 0, 0)]], axis=0)
        prot_prim.positions = coords
        if not self.arguments['show_amino_acids']:
            prot_prim.diameters = 0
        colors = np.ones((len(coords), 4))
        colors[:-1, :3] = plato.cmap.cubeellipse_intensity(tj[filt].astype(np.float32))
        colors[-1:, :3] = plato.cmap.cubeellipse_intensity(ti[:1].astype(np.float32))
        prot_prim.colors = colors
        if self.arguments['true_coords']:
            atom_prim.positions = sy[0][cfilt]*self.loaded_scope['y_scale']
        else:
            atom_prim.positions = pred
        atom_prim.colors = named_type_colors[sct[0][cfilt]]
        atom_prim.diameters = named_type_diameters[sct[0][cfilt]]
        lines.start_points = pred
        lines.end_points = sy[0][cfilt]*self.loaded_scope['y_scale']
        lines.colors = named_type_colors[sct[0][cfilt]]

        scope.setdefault('visuals', []).append(self)

        self.gui_actions = [
            ('Visualize next molecule', self._next_molecule),
        ]

    def _next_molecule(self, scope, storage):
        if scope.get('rerun_callback', None) is not None:
            scope['rerun_callback']()

    def draw_plato(self):
        return self.scene
