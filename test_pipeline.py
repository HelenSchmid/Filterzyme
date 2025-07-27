import pandas as pd

from filtering_pipeline.pipeline import Docking
from filtering_pipeline.pipeline import Superimposition

df = pd.read_pickle('DEHP-MEHP.pkl')
df = df.drop_duplicates(subset='Entry', keep='first')
df = df.head(2)

docking = Docking(
    ligand_name = 'TPP', 
    ligand_smiles ='CCCCC(CC)COC(=O)C1=CC=CC=C1C(=O)OCC(CC)CCCC', 
    df = df,
    output_dir = 'test_pipeline', 
    squidly_dir = '/nvme2/ariane/home/data/models/squidly_final_models/'
)

docking.run()


superimposition = Superimposition(
    ligand_name = 'TPP',
    maxMatches = 1000, 
    input_dir="test_pipeline", 
    output_dir="test_pipeline"
)

superimposition.run()