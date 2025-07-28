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
pip install git+https://github.com/yourusername/GeometricFilters.git
```

### Option 2: Clone the repository
```bash
git clone https://github.com/yourusername/GeometricFilters.git
cd GeometricFilters
pip install .
```
