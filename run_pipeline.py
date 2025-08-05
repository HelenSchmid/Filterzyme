import pandas as pd
from filtering_pipeline.pipeline import Pipeline


'''
if __name__ == "__main__":

    # Configure and run
    pipeline = Pipeline(
        df = pd.read_pickle("examples/DEHP-MEHP.pkl").head(2),
        ligand_name="TPP",
        ligand_smiles="CCCCC(CC)COC(=O)C1=CC=CC=C1C(=O)OCC(CC)CCCC",
        smarts_pattern='[$([CX3](=O)[OX2H0][#6])]',
        max_matches=1000,
        esterase=1,
        find_closest_nuc=1,
        num_threads=1,
        metagenomic_enzymes=1,
        squidly_dir='filtering_pipeline/squidly_final_models/',
        base_output_dir="pipeline_output_test", 

    )

    pipeline.run()

'''
from filtering_pipeline.pipeline import Superimposition, GeometricFilters
import pandas as pd
from pathlib import Path

# --- Set parameters ---
ligand_name = "TPP"
ligand_smiles = "CCCCC(CC)COC(=O)C1=CC=CC=C1C(=O)OCC(CC)CCCC"
smarts_pattern = '[$([CX3](=O)[OX2H0][#6])]'
max_matches = 1000
base_output_dir = Path("pipeline_output_test")
num_threads = 1

# --- Run Superimposition ---
superimp = Superimposition(
    ligand_name=ligand_name,
    maxMatches=max_matches,
    input_dir=base_output_dir / "docking",  # assumes docking has already been run
    output_dir=base_output_dir / "superimposition",
    num_threads=num_threads,
)

df_superimposed = superimp.run()  # This runs all: preparation, superimposition, protein & ligand RMSD

# --- Load best structures for filtering ---
df_best = pd.read_pickle(base_output_dir / "superimposition/best_structures.pkl")

# --- Run Geometric Filtering ---
gf = GeometricFilters(
    substrate_smiles=ligand_smiles,
    smarts_pattern=smarts_pattern,
    df=df_best,
    esterase=1,
    find_closest_nuc=1,
    input_dir=base_output_dir / "superimposition",
    output_dir=base_output_dir / "geometricfiltering",
    num_threads=num_threads,
)

df_final = gf.run()

