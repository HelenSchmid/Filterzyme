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
```bash
conda env create -f environment.yml
conda activate filterpipeline
```

### Clone the repository
```bash
git clone https://github.com/HelenSchmid/Filterzyme.git
cd EnzymeStructuralFiltering
pip install .
```

###  Install via pip
```bash
pip install filterzyme
```

## Usage Example

The input pandas **DataFrame** must include:  
- `Entry` – unique identifier for each enzyme  
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
