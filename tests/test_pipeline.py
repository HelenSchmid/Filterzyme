import pandas as pd

from filtering_pipeline.pipeline import Docking
from filtering_pipeline.pipeline import Superimposition
from filtering_pipeline.pipeline import GeometricFilters

df = pd.read_pickle('examples/DEHP-MEHP.pkl')
df = df.drop_duplicates(subset='Entry', keep='first')
df = df.head(2)

docking = Docking(
    ligand_name = 'TPP', 
    ligand_smiles ='CCCCC(CC)COC(=O)C1=CC=CC=C1C(=O)OCC(CC)CCCC', 
    df = df,
    output_dir = 'docking_test', 
    squidly_dir = '/nvme2/ariane/home/data/models/squidly_final_models/'
)

docking.run()

superimposition = Superimposition(
    ligand_name = 'TPP', 
    maxMatches = 1000, 
    input_dir = 'docking_test', 
    output_dir = 'superimposition_test', 
)
#superimposition.run()

geometricfilters = GeometricFilters(
    substrate_smiles = 'CCCC[C@@H](CC)COC(=O)C1=CC=CC=C1C(=O)OC[C@@H](CC)CCCC',
    smarts_pattern = '[$([CX3](=O)[OX2H0][#6])]', 
    df = pd.read_csv('superimposition_test/ligandRMSD/best_docked_structures.csv'), 
    input_dir='superimposition_test'
)

geometricfilters.run()
