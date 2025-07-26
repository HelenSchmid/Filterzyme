import pandas as pd

from filtering_pipeline.pipeline import Docking

df = pd.read_pickle('DEHP-MEHP.pkl')
df = df.drop_duplicates(subset='Entry', keep='first')
df = df.head(1)

docking = Docking(
    ligand_name = 'TPP', 
    ligand_smiles ='CCCCC(CC)COC(=O)C1=CC=CC=C1C(=O)OCC(CC)CCCC', 
    df = df,
    output_dir = 'test_pipeline', 
    squidly_dir = '/home/helen/enzyme-tk/models/squidly_final_models/'
)

docking.run()
