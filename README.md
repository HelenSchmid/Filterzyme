# Filterzyme

Structural filtering pipeline using docking and active site heuristics to prioritze ML-predicted enzyme variants for experimental validation. 
This tool processes superimposed ligand poses and filters them using geometric criteria such as distances, angles, and optionally, esterase-specific filters or nucleophilic proximity.

---

## Features

- Analysis of enzyme-ligand docking using multiple docking tools (ML- and physics-based).
- Optional catalytic nucleophile-focused analysis for esterases or other enzymes with nucleophilic catalytic residues. 
- User-friendly pipeline only using a .pkl file as input and ligand smile strings.

---

## Installation

## Environment Setup
### Using conda

Go onto a node with GPU access.

```bash
module load miniforge/25.9.1
eval "$(conda shell.bash hook)"
conda create --name filterzyme python=3.12 pip -y
conda activate filterzyme
```

### Clone the repository
```bash
git clone https://github.com/MoraGroup/Filterzyme.git
cd EnzymeStructuralFiltering
python setup.py sdist bdist_wheel
pip install dist/filterzyme-0.0.6.tar.gz --use-deprecated=legacy-resolver
pip install enzymetk==0.0.8
```

If you have issues with openbabel try running the below code (needed to get working on AITHYRA cluster)
```
conda install -c conda-forge openbabel swig -y
conda install -c conda-forge plip -y
export PYTHONPATH=$CONDA_PREFIX/lib/python3.12/site-packages:$PYTHONPATH
conda install -y "numpy<2.0" "pandas<3.0" "openbabel" "plip"
conda install -y "rdkit<=2023.09.6"
```

## Download cache to the installed directory using

```boltz predict <input file> --cache $CONDA_PREFIX/lib/python3.12/site-packages/boltz/````

Then you need to pass this to the 

## Usage Example

The input pandas **DataFrame** must include:  
- `Entry` – unique identifier for each enzyme and substrate pair
- `Sequence` – amino acid sequence of the enzyme
- `substrate_name` – name of the substrate
- `substrate_smiles` – SMILES string of substrate e.g. MEHP "CCCCC(CC)COC(=O)C1=CC=CC=C1C(=O)O"
- `substrate_moiety` – SMARTS pattern to define chemical moiety of interest within substrate e.g. general ester SMARTS "[C]\(=O)(O)(O)"

If cofactors are included, add:
- `cofactor_name` – name of the cofactor
- `cofactor_smiles` – SMILES string of cofactor e.g. PLP "CC1=NC=C(C(=C1O)C=O)COP(=O)(O)O" 
- `cofactor_moiety` – SMARTS pattern to define chemical moiety of interest within the cofactor 


```python
from filterzyme.pipeline import Pipeline
import pandas as pd

df = pd.read_pickle("example_df.pkl")

pipeline = Pipeline(
        df = df,
        max_matches=1000,                # number of matches during substructure SEARCH
        esterase=0,                      # 1 if interested specifically in esterases
        num_threads=1,                   # number of threads
        skip_catalytic_residue_prediction = False,
        alternative_structure_for_vina = 'Chai', 
        squidly_dir='/nvme2/helen/EnzymeStructuralFiltering/filterzyme/squidly_final_models/',
        base_output_dir="pipeline_output"
    )

pipeline.run()
```
