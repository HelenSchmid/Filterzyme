import pandas as pd
from filtering_pipeline.pipeline import Pipeline

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
        base_output_dir="pipeline_output_test"
    )

    pipeline.run()

