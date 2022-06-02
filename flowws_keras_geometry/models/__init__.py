from flowws import try_to_import

from .CrystalStructureClassification import CrystalStructureClassification
GalaMoleculeForceRegression = try_to_import(
    '.GalaMoleculeForceRegression', 'GalaMoleculeForceRegression', __name__)
GalaPDBInverseCoarseGrain = try_to_import(
    '.GalaPDBInverseCoarseGrain', 'GalaPDBInverseCoarseGrain', __name__)
from .MoleculeForceRegression import MoleculeForceRegression
from .PDBInverseCoarseGrain import PDBInverseCoarseGrain
from .PDBInverseCoarseGrainTransformer import PDBInverseCoarseGrainTransformer
