# Filterzyme

Structural filtering pipeline using docking and active site heuristics to prioritize ML-predicted enzyme variants for experimental validation. 
This tool processes superimposed ligand poses and filters them using geometric criteria such as distances, angles, and optionally, esterase-specific filters or nucleophilic proximity.

---

## Features

- Analysis of enzyme-ligand docking using multiple docking tools (ML- and physics-based).
- Optional catalytic nucleophile-focused analysis for esterases or other enzymes with nucleophilic catalytic residues. 
- Optional Vina docking for enzymes with known active sites.
- User-friendly pipeline using a DataFrame as input with ligand SMILES strings.

---

## Quick Start

For full installation instructions, see [docs/getting_started.md](docs/getting_started.md).

```bash
conda create --name filterzyme python=3.11 pip -y
conda activate filterzyme
git clone https://github.com/MoraGroup/Filterzyme.git
cd Filterzyme
python setup.py sdist bdist_wheel
pip install dist/filterzyme-0.0.6.tar.gz --use-deprecated=legacy-resolver
pip install enzymetk==0.0.8
```

### Download Boltz cache

```bash
boltz predict example.yml --cache /path/to/your/boltz/cache/
```

### Run the pipeline

```python
from filterzyme.pipeline_v2 import Pipeline
import pandas as pd

df = pd.DataFrame({
    'Entry': ['enzyme_1'],
    'Sequence': ['MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAATGFHI'],
    'substrate_smiles': ['CCOC(=O)C'],
    'substrate_name': ['ethyl_acetate'],
    'substrate_moiety': ['[C](=O)([O])([O])'],
})

pipeline = Pipeline(
    df=df,
    boltz_cache_dir="/path/to/boltz/cache",
    skip_catalytic_residue_prediction=True,
    base_output_dir="pipeline_output"
)
pipeline.run()
```

### Running with Vina

To enable Vina docking (for non-de-novo enzymes with known active sites):

```python
pipeline = Pipeline(
    df=df,
    boltz_cache_dir="/path/to/boltz/cache",
    run_vina=True,
    alternative_structure_for_vina='Chai',
    base_output_dir="pipeline_output"
)
pipeline.run()
```

Vina requires the `docko` package: `pip install docko`

---

## Input DataFrame Schema

The input pandas **DataFrame** must include:  
- `Entry` -- unique identifier for each enzyme and substrate pair
- `Sequence` -- amino acid sequence of the enzyme
- `substrate_name` -- name of the substrate
- `substrate_smiles` -- SMILES string of substrate
- `substrate_moiety` -- SMARTS pattern to define chemical moiety of interest within substrate

If cofactors are included, add:
- `cofactor_name` -- name of the cofactor
- `cofactor_smiles` -- SMILES string of cofactor (e.g., PLP: `CC1=NC=C(C(=C1O)C=O)COP(=O)(O)O`)
- `cofactor_moiety` -- SMARTS pattern for the cofactor moiety of interest

---

## Documentation

- [Getting Started](docs/getting_started.md) -- Installation and setup
- [Pipeline Overview](docs/pipeline_overview.md) -- Architecture and design
- [API Reference](docs/api_reference.md) -- Full parameter documentation
- [Examples](docs/examples/) -- Working example scripts

---

## Examples

| Script | Description |
|--------|-------------|
| [`00_quickstart.py`](docs/examples/00_quickstart.py) | Minimal 1-sequence example (Chai + Boltz) |
| [`01_docking.py`](docs/examples/01_docking.py) | Docking phase only |
| [`02_superimposition.py`](docs/examples/02_superimposition.py) | Superimposition phase only |
| [`03_geometric_filtering.py`](docs/examples/03_geometric_filtering.py) | Geometric filtering only |
| [`04_full_pipeline.py`](docs/examples/04_full_pipeline.py) | Full pipeline with optional Vina |
| [`05_cofactor_example.py`](docs/examples/05_cofactor_example.py) | Pipeline with cofactor moieties |

All examples use argparse for configurable paths (no hardcoded paths). Run any example with `--help` for options.
