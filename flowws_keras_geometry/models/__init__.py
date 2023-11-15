from flowws import try_to_import

CrystalStructureClassification = try_to_import(
    '.CrystalStructureClassification', 'CrystalStructureClassification', __name__)
GalaMoleculeForceRegression = try_to_import(
    '.GalaMoleculeForceRegression', 'GalaMoleculeForceRegression', __name__)
GalaPDBInverseCoarseGrain = try_to_import(
    '.GalaPDBInverseCoarseGrain', 'GalaPDBInverseCoarseGrain', __name__)
MoleculeForceRegression = try_to_import(
    '.MoleculeForceRegression', 'MoleculeForceRegression', __name__)
PDBInverseCoarseGrain = try_to_import(
    '.PDBInverseCoarseGrain', 'PDBInverseCoarseGrain', __name__)
PDBInverseCoarseGrainTransformer = try_to_import(
    '.PDBInverseCoarseGrainTransformer', 'PDBInverseCoarseGrainTransformer', __name__)
