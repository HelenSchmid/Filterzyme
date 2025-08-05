# EnzymeStructuralFiltering

Structural filtering pipeline using docking and active site heuristics to prioritze ML-predicted enzyme variants for experimental validation. 
This tool processes superimposed ligand poses and filters them using geometric criteria such as distances, angles, and optionally, esterase-specific filters or nucleophilic proximity.

---

## Features

- Analysis of enzyme-ligand docking using multiple docking tools (ML- and physics-based).
- Optional esterase or nucleophile-focused analysis.
- User-friendly pipeline only using a .pkl file as input and ligand smile strings.
- Different parts of the pipeline can be run independently of each other. 

---

## Installation

### Option 1: Install via pip
```bash
pip install enzyme-filtering-pipline
```
### Option 2: Clone the repository
```bash
git clone https://github.com/HelenSchmid/EnzymeStructuralFiltering.git
cd EnzymeStructuralFiltering
pip install .
```

## Environment Setup
### Using conda
```bash
conda env create -f environment.yml
conda activate filterpipeline
```

## Usage Example
```python
from filtering_pipeline.pipeline import Pipeline
import pandas as pd
from pathlib import Path
df = pd.read_pickle("DEHP-MEHP.pkl").head(5)

pipeline = Pipeline(
        df = df,
        ligand_name="TPP",
        ligand_smiles="CCCCC(CC)COC(=O)C1=CC=CC=C1C(=O)OCC(CC)CCCC", # SMILES string of ligand
        smarts_pattern='[$([CX3](=O)[OX2H0][#6])]',                  # SMARTS pattern of the chemical moiety of interest of ligand
        max_matches=1000,
        esterase=1,
        find_closest_nuc=1,
        num_threads=1,
        squidly_dir='filtering_pipeline/squidly_final_models/',
        base_output_dir="pipeline_output"
    )

pipeline.run()
```
### Running pipline on multiple ligands at the same time
You can run the filtering pipeline for multiple ligands by using a simple Bash script that passes ligand names and their SMILES strings to a Python runner script.

```bash
#!/bin/bash

# Define ligands and their SMILES representations
declare -A LIGANDS
LIGANDS["tri_2_chloroethylPi"]="C(CCl)OP(=O)(OCCCl)OCCCl"
LIGANDS["DEHP"]="CCCCC(CC)COC(=O)C1=CC=CC=C1C(=O)OCC(CC)CCCC"
LIGANDS["TPP"]="C1=CC=C(C=C1)OP(=O)(OC2=CC=CC=C2)OC3=CC=CC=C3"

# Create logs directory
mkdir -p logs

# Loop over each ligand and run the pipeline
for name in "${!LIGANDS[@]}"
do
  echo "Running for $name..."

  python benchmark_filtering_on_exp_tested_variants_run.py "$name" "${LIGANDS[$name]}" \
    2> "logs/${name}.err" \
    1> "logs/${name}.out"

  echo "Finished $name. Logs saved to logs/${name}.out and logs/${name}.err"
done
```
Each run invokes benchmark_filtering_on_exp_tested_variants_run.py, which looks like:
```python
import argparse
import pandas as pd
from filtering_pipeline.pipeline import Pipeline

# SMARTS patterns to define substructures per ligand
SMARTS_MAP = {
    "TPP": "[P](=O)(O)(O)",
    "DEHP": "[C](=O)[O][C]",
    "Monuron": "Cl",
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("ligand_name", type=str, help="Ligand name (e.g. TPP)")
    parser.add_argument("ligand_smiles", type=str, help="SMILES string of the ligand")
    return parser.parse_args()

def main():
    args = parse_args()

    smarts_pattern = SMARTS_MAP.get(args.ligand_name)

    pipeline = Pipeline(
        df=pd.read_pickle("examples/DEHP-MEHP.pkl").head(2),
        ligand_name=args.ligand_name,
        ligand_smiles=args.ligand_smiles,
        smarts_pattern=smarts_pattern,
        max_matches=5000,
        find_closest_nuc=1,
        num_threads=1,
        squidly_dir="filtering_pipeline/squidly_final_models/",
        base_output_dir=f"pipeline_output_{args.ligand_name}",
    )

    pipeline.run()

if __name__ == "__main__":
    main()
```
