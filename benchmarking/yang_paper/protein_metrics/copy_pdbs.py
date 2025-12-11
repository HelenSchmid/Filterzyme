import os
import shutil

destination = "/nvme2/helen/EnzymeStructuralFiltering/benchmarking/serine_hydrolases/protein_metrics/pdbs"
source = "/nvme2/helen/EnzymeStructuralFiltering/benchmarking/serine_hydrolases/filterzyme_output/superimposition/preparedfiles_for_superimposition"

os.makedirs(destination, exist_ok=True)

for file in os.listdir(source):
    if file.lower().endswith(".pdb"):
        src = os.path.join(source, file)
        dst = os.path.join(destination, file)
        shutil.copy2(src, dst)
        print(f"Copied: {file}")

print("Done.")
