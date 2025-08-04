#!/bin/bash

# Define ligands and their SMILES representations
declare -A LIGANDS
LIGANDS["tri_2_chloroethylPi"]="C(CCl)OP(=O)(OCCCl)OCCCl"
LIGANDS["DEHP"]="CCCCC(CC)COC(=O)C1=CC=CC=C1C(=O)OCC(CC)CCCC"
LIGANDS["TPP"]="C1=CC=C(C=C1)OP(=O)(OC2=CC=CC=C2)OC3=CC=CC=C3"


mkdir -p logs

for name in "${!LIGANDS[@]}"
do
  echo "Running for $name..."

  python benchmark_filtering_on_exp_tested_variants_run.py "$name" "${LIGANDS[$name]}" \
    2> "logs/${name}.err" \
    1> "logs/${name}.out"

  echo "Finished $name. Logs: logs/${name}.out / ${name}.err"
done