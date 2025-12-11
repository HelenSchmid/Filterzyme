import os
from glob import glob

in_dir  = "target_seqs"
out_dir = "target_seqs_clean"

os.makedirs(out_dir, exist_ok=True)

n_files = 0
n_kept_records = 0

for fpath in glob(os.path.join(in_dir, "*")):
    if not fpath.lower().endswith((".fa", ".fasta", ".faa")):
        continue

    with open(fpath) as f:
        lines = [l.rstrip("\n") for l in f]

    out_lines = []
    keep = False  # whether we are inside a protein record

    for line in lines:
        if line.startswith(">"):           # new FASTA header
            if line.lower().startswith(">ligand"):
                keep = False              # skip ligand record
            else:
                keep = True               # keep protein or anything not ligand
                out_lines.append(line)
                n_kept_records += 1
        else:
            if keep:
                out_lines.append(line)

    # write cleaned file
    out_path = os.path.join(out_dir, os.path.basename(fpath))
    with open(out_path, "w") as out:
        out.write("\n".join(out_lines) + "\n")
    n_files += 1

print(f"Done. Cleaned {n_files} files. Kept {n_kept_records} protein entries.")
print(f"Clean FASTAs saved to: {os.path.abspath(out_dir)}")
