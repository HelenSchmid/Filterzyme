import os
from glob import glob

bad = []

AA = set("ACDEFGHIKLMNPQRSTVWY")

for f in glob("target_seqs/*"):
    with open(f) as fh:
        header = None
        seq = ""
        for line in fh:
            if line.startswith(">"):
                # finish previous record
                if header is not None and len(seq.strip()) == 0:
                    bad.append((f, header))
                header = line.strip()
                seq = ""
            else:
                seq += line.strip()
        # last record
        if header is not None and len(seq.strip()) == 0:
            bad.append((f, header))

print("Empty sequences found:")
for f, h in bad:
    print(f"  File: {f}, header: {h}")

print(f"\nTotal empty sequences: {len(bad)}")
