
import pandas as pd
import os
import torch
import sys
import argparse

from filtering_pipeline.pipeline import Pipeline

# Define SMARTS patterns for ligands
SMARTS_MAP = {
    "TPP": "[P](=O)(O)(O)",
    "DEHP": "[C](=O)[O][C]",
    "Monuron": 'Cl', 
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the filtering pipeline for a given ligand"
    )
    parser.add_argument(
        "ligand_name",
        type=str,
        help="Name of the ligand (use underscores instead of spaces, e.g. vinyl_oleate)",
    )
    parser.add_argument(
        "ligand_smiles",
        type=str,
        help="SMILES string of the ligand, e.g. CCCC(=O)OCC",
    )
    return parser.parse_args()


# Run pipeline
def main():
    args = parse_args()

    smarts_pattern = SMARTS_MAP.get(args.ligand_name)

    # Set a unique output directory for this ligand
    output_dir = f"pipeline_output_{args.ligand_name}"

    # Load data and run the pipeline
    pipeline = Pipeline(
        df=pd.read_pickle("examples/DEHP-MEHP.pkl").head(2),
        ligand_name=args.ligand_name,
        ligand_smiles=args.ligand_smiles,
        smarts_pattern=smarts_pattern,
        max_matches=5000,
        find_closest_nuc=1,
        num_threads=1,
        squidly_dir='filtering_pipeline/squidly_final_models/',
        base_output_dir=output_dir,
    )

    pipeline.run()


if __name__ == "__main__":
    main()