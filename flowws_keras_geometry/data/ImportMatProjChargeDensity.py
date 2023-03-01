import collections
import contextlib
import json

from emmet.core.summary import HasProps
import flowws
from flowws import Argument as Arg
import gtar
from mp_api.client import MPRester
import numpy as np

@flowws.add_stage_arguments
class ImportMatProjChargeDensity(flowws.Stage):
    """Import charge density data from the Materials Project.

    This module downloads and processes data from the Materials
    Project charge density API
    (https://materialsproject.org/ml/charge_densities). Data are
    imported, adapted into a form suitable for analysis with freud,
    and saved into a getar file.

    """

    ARGS = [
        Arg('api_key', None, str,
            help='API key to access materials project database'),
        Arg('output_file', '-o', str, 'materials_project_charge.sqlite',
            help='Name of output getar file to produce'),
        Arg('chunk_size', '-b', int, 1000,
            help='Chunk size for API requests'),
        Arg('num_chunks', '-n', int,
            help='Number of chunks of data to request'),
        Arg('elements', '-e', [str],
            help='Element symbols to filter search with'),
        Arg('query_kwargs', '-q', [(str, eval)],
            help='Additional query keyword arguments'),
    ]

    def run(self, scope, storage):
        api_key = self.arguments['api_key']
        num_chunks = self.arguments.get('num_chunks', None)
        all_types = set()
        density_stds = []

        with contextlib.ExitStack() as stack:
            mpr = stack.enter_context(MPRester(api_key=api_key))

            kwargs = {}
            if self.arguments.get('elements', None):
                kwargs['elements'] = self.arguments['elements']

            for (key, val) in dict(self.arguments.get('query_kwargs', [])).items():
                kwargs[key] = val

            docs = mpr.summary.search(
                num_chunks=num_chunks,
                chunk_size=self.arguments['chunk_size'],
                has_props=[HasProps.charge_density],
                **kwargs
            )

            traj = stack.enter_context(gtar.GTAR(self.arguments['output_file'], 'w'))

            for doc in docs:
                density = mpr.get_charge_density_from_material_id(doc.material_id)
                data_dict = self.write_gtar_entry(traj, doc, density)
                all_types.update(data_dict['type_names'])
                density_stds.append(data_dict['std'])

            traj.writePath('all_type_names.json', json.dumps(sorted(list(all_types))))
            traj.writePath('density_scale.f32.uni', np.mean(density_stds))

    @staticmethod
    def write_gtar_entry(traj, doc, density):
        group = str(doc.material_id)

        structure = density.structure
        box_matrix = structure.lattice.matrix.T

        Nx = len(density.xpoints)
        Ny = len(density.ypoints)
        Nz = len(density.zpoints)

        assert np.allclose(np.linspace(0, 1, Nx), density.xpoints)
        assert np.allclose(np.linspace(0, 1, Ny), density.ypoints)
        assert np.allclose(np.linspace(0, 1, Nz), density.zpoints)

        grid = (Nx, Ny, Nz)

        positions = structure.cart_coords
        all_type_names = [t.name for t in structure.species]
        type_name_map = collections.defaultdict(lambda: len(type_name_map))
        types = [type_name_map[name] for name in all_type_names]
        type_names = [t for (t, i) in
                      sorted(type_name_map.items(), key=lambda x: x[::-1])]

        density_values = density.data['total']
        std = np.std(density_values)

        traj.writePath('{}/grid.u32.uni'.format(group), grid)
        traj.writePath('{}/box_matrix.f32.uni'.format(group), box_matrix)
        traj.writePath('{}/point_density.f32.uni'.format(group), density_values)
        traj.writePath('{}/position.f32.ind'.format(group), positions)
        traj.writePath('{}/type.u32.ind'.format(group), types)
        traj.writePath('{}/type_names.json'.format(group), json.dumps(type_names))
        return dict(type_names=type_names, std=std)
