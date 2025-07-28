# EnzymeStructuralFiltering

Structural filtering pipeline using docking and active site heuristics to prioritze ML-predicted enzyme variants for experimental validation. 
This tool processes superimposed ligand poses and filters them using geometric criteria such as distances, angles, and optionally, esterase-specific filters or nucleophilic proximity.

---

## 🚀 Features

- Parse and apply SMARTS patterns to ligand structures.
- Filter poses based on geometric constraints.
- Optional esterase or nucleophile-focused analysis.
- Supports CSV and pickle-based data pipelines.

---

## 📦 Installation

### Option 1: Install via pip
```bash
pip install XXXX
```
### Option 2: Clone the repository
```bash
git clone https://github.com/HelenSchmid/EnzymeStructuralFiltering.git
cd EnzymeStructuralFiltering
pip install .
```

## :seedling: Environment Setup
### Using conda
```bash
conda env create -f environment.yml
conda activate filterpipeline
```

## 🔧 Usage Example
```python
from filtering_pipeline.pipeline import Pipeline
import pandas as pd
from pathlib import Path
df = pd.read_pickle("DEHP-MEHP.pkl")

pipeline = Pipeline(
        df = df,
        ligand_name="TPP",
        ligand_smiles="CCCCC(CC)COC(=O)C1=CC=CC=C1C(=O)OCC(CC)CCCC", # SMILES string of ligand
        smarts_pattern='[$([CX3](=O)[OX2H0][#6])]',                  # SMARTS pattern of the chemical moiety of interest of ligand
        max_matches=1000,
        esterase=1,
        find_closest_nuc=1,
        num_threads=1,
        squidly_dir='/nvme2/ariane/home/data/models/squidly_final_models/',
        base_output_dir="pipeline_output"
    )

pipeline.run()
