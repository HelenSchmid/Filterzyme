import os
import shutil

destination = "/nvme2/helen/EnzymeStructuralFiltering/benchmarking/serine_hydrolases/protein_metrics/target_seqs"
source = "/nvme2/helen/EnzymeStructuralFiltering/benchmarking/serine_hydrolases/filterzyme_output/docking/chai"

os.makedirs(destination, exist_ok=True)

copied = 0

for root, dirs, files in os.walk(source):
    for file in files:
        if file.lower().endswith(".fasta") or file.lower().endswith(".fa"):
            src = os.path.join(root, file)
            dst = os.path.join(destination, file)
            shutil.copy2(src, dst)
            copied += 1
            print(f"Copied: {src} -> {dst}")

print(f"Done. Total FASTA files copied: {copied}")
