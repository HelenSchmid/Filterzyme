import os
from glob import glob

directory = "/nvme2/helen/EnzymeStructuralFiltering/benchmarking/serine_hydrolases/protein_metrics/target_seqs"

# Rule: If files share the same prefix (before "_"), keep only the one containing "probe"
# Otherwise keep the first lexicographically.
to_delete = []

files = sorted(glob(os.path.join(directory, "*.fasta")))
groups = {}

# Group by prefix
for f in files:
    base = os.path.basename(f)
    prefix = base.split("_")[0]     # "soli_probe.fasta" -> "soli"

    groups.setdefault(prefix, []).append(f)

# Decide which to keep and which to delete
for prefix, flist in groups.items():
    # Prefer the one with "probe" in the filename
    keep = None
    for f in flist:
        if "probe" in os.path.basename(f):
            keep = f
            break

    # If none contain "probe", keep the first alphabetically
    if keep is None:
        keep = flist[0]

    # The rest should be deleted:
    for f in flist:
        if f != keep:
            to_delete.append(f)

# Delete the files
for f in to_delete:
    print("Deleting:", f)
    os.remove(f)

print("Done! Kept one fasta per prefix.")
