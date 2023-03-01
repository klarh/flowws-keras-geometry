import functools

import flowws
from flowws import Argument as Arg
from flowws.Argument import Range
import gtar
import numpy as np

from ..data.MatProjChargeDensity import MatProjChargeDensity

@flowws.add_stage_arguments
class MatProjChargeDensityViewer(flowws.Stage):
    """Visualize charge density from an imported Materials Project file.

    """

    ARGS = [
        Arg('filename', '-f', str, 'materials_project_charge.sqlite',
            help='Filename of imported data to load'),
        Arg('point', '-p', int, -1, valid_values=Range(-1, 16000, True),
            help='Select an individual point by index, or view the whole system if negative'),
        Arg('filter_points', '-n', int, 1024, valid_values=Range(0, 16000, True),
            help='Number of high-density points to view if given'),
        Arg('color_scale', None, float, .25, valid_values=flowws.Range(0, 10, True),
            help='Factor to scale point color RGB intensities by'),
        Arg('diameter', '-d', float, .1,
            help='Point sphere diameter'),
        Arg('group', '-g', str,
            help='Group to visualize'),
    ]

    def run(self, scope, storage):
        if getattr(self, 'gtar_file', None) is None:
            self.gtar_file = gtar.GTAR(self.arguments['filename'], 'r')
            groups = {rec.getGroup() for rec in self.gtar_file.getRecordTypes()
                      if rec.getGroup()}
            groups = list(sorted(groups))
            self.arg_specifications['group'].valid_values = groups
            self.arguments.setdefault('group', groups[0])

        self.record = MatProjChargeDensity.read_gtar_entry(
            self.gtar_file, self.arguments['group'])
        self.freud_system = (box, _, positions) = MatProjChargeDensity.translate_to_freud(
            self.record.grid, self.record.box_matrix, self.record.positions)

        scope['position'] = positions
        scope['type'] = self.record.types
        scope['box'] = [box.Lx, box.Ly, box.Lz, box.xy, box.xz, box.yz]
        scope.setdefault('visuals', []).append(self)

    def draw_plato(self):
        import plato, plato.draw as draw

        record = self.record
        (box, coords, positions) = self.freud_system

        point_thetas = record.point_density.copy()
        point_thetas -= np.min(point_thetas)
        point_thetas /= np.max(point_thetas)
        point_thetas = .8*point_thetas + .2*(1 - point_thetas)

        if self.arguments['filter_points']:
            sortidx = np.argsort(-record.point_density)
            filt = sortidx[:self.arguments['filter_points']]
            coords = coords[filt]
            point_thetas = point_thetas[filt]

        point_colors = plato.cmap.cubehelix(point_thetas)
        point_colors[:, :3] *= self.arguments['color_scale']

        point_prim = draw.Spheres(
            positions=coords, colors=point_colors, diameters=self.arguments['diameter'])

        atom_colors = np.ones((len(positions), 4))
        atom_colors[:, :3] = plato.cmap.cubeellipse_intensity(
            record.types.astype(np.float32))
        atom_prim = draw.Spheres(positions=positions, colors=atom_colors)

        box_prim = draw.Box.from_box(box, color=(.5, .5, .5, 1), width=.1)

        features = dict(additive_rendering=True)
        return draw.Scene([point_prim, atom_prim, box_prim], features=features)
