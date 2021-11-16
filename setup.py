#!/usr/bin/env python

import os
from setuptools import setup

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

version_fname = os.path.join(THIS_DIR, 'flowws_keras_geometry', 'version.py')
with open(version_fname) as version_file:
    exec(version_file.read())

readme_fname = os.path.join(THIS_DIR, 'README.md')
with open(readme_fname) as readme_file:
    long_description = readme_file.read()

entry_points = set()
flowws_modules = []
package_names = ['flowws_keras_geometry']

def add_subpkg(subpkg, module_names):
    package_names.append('flowws_keras_geometry.{}'.format(subpkg))
    for name in module_names:
        if name not in entry_points:
            flowws_modules.append('{0} = flowws_keras_geometry.{1}.{0}:{0}'.format(name, subpkg))
            entry_points.add(name)
        flowws_modules.append(
            'flowws_keras_geometry.{1}.{0} = flowws_keras_geometry.{1}.{0}:{0}'.format(name, subpkg))

module_names = [
]
for name in module_names:
    if name not in entry_points:
        flowws_modules.append('{0} = flowws_keras_geometry.{0}:{0}'.format(name))
        entry_points.add(name)
    flowws_modules.append(
        'flowws_keras_geometry.{0} = flowws_keras_geometry.{0}:{0}'.format(name))

subpkg = 'data'
module_names = [
    'MD17',
    'ModelNet',
    'PDBCache',
    'PDBCoarseGrained',
    'PyriodicDataset',
    'RMD17',
]
add_subpkg(subpkg, module_names)

subpkg = 'models'
module_names = [
    'CrystalStructureClassification',
    'MoleculeForceRegression',
    'PDBInverseCoarseGrain',
    'PDBInverseCoarseGrainTransformer',
]
add_subpkg(subpkg, module_names)

subpkg = 'viz'
module_names = [
    'BackmapCoarseGrainViewer',
    'MoleculeAttentionViewer',
    'ParticleAttentionViewer',
]
add_subpkg(subpkg, module_names)

setup(name='flowws-keras-geometry',
      author='Matthew Spellings',
      author_email='mspells@vectorinstitute.ai',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
      ],
      description='Stage-based scientific workflows for deep learning experiments with geometric algebra',
      entry_points={
          'flowws_modules': flowws_modules,
      },
      extras_require={},
      install_requires=[
          'flowws',
          'flowws-keras-experimental',
      ],
      license='MIT',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=package_names,
      python_requires='>=3',
      version=__version__
      )
