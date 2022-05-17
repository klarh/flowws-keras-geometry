from flowws import try_to_import

from .CrystalStructureClassification import CrystalStructureClassification
GalaMoleculeForceRegression = try_to_import(
    '.GalaMoleculeForceRegression', 'GalaMoleculeForceRegression', __name__)
from .MoleculeForceRegression import MoleculeForceRegression
from .PDBInverseCoarseGrain import PDBInverseCoarseGrain
