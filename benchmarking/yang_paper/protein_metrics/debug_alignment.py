#!/usr/bin/env python3
"""Debug alignment metrics parsing"""

from pathlib import Path

def parse_fasta(filename, return_names=False):
    """Simple FASTA parser"""
    names = []
    seqs = []
    with open(filename, 'r') as f:
        current_name = None
        current_seq = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_name is not None:
                    seqs.append(''.join(current_seq))
                    names.append(current_name)
                current_name = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
        if current_name is not None:
            seqs.append(''.join(current_seq))
            names.append(current_name)
    if return_names:
        return names, seqs
    return seqs

# Parse target and reference sequences
target_seqs_dir = Path('target_seqs_test')
reference_seqs_dir = Path('reference_seqs_test')

print("=== TARGET SEQUENCES ===")
n_train = []
s_train = []
for target_fasta in target_seqs_dir.glob("*.fasta"):
    names, seqs = parse_fasta(str(target_fasta), return_names=True)
    for name, seq in zip(names, seqs):
        n_train.append(name)
        s_train.append(seq)
        print(f"  {name[:50]}")

print("\n=== REFERENCE SEQUENCES ===")
n_query = []
s_query = []
for ref_fasta in reference_seqs_dir.glob("*.fasta"):
    names, seqs = parse_fasta(str(ref_fasta), return_names=True)
    for name, seq in zip(names, seqs):
        n_query.append(name)
        s_query.append(seq)
        print(f"  {name[:50]}")

# Build dicts
train_seqs = {nt: st for st, nt in zip(s_train, n_train)}
query_seqs = {nq: sq for sq, nq in zip(s_query, n_query)}

print(f"\n=== DICTS ===")
print(f"train_seqs keys ({len(train_seqs)}): {list(train_seqs.keys())[:5]}")
print(f"query_seqs keys ({len(query_seqs)}): {list(query_seqs.keys())[:5]}")

# Parse ggsearch results
print("\n=== GGSEARCH PARSING ===")
with open('output_test_clean/ggsearch_results_BLOSUM62.txt') as f:
    lines = f.readlines()

train_coming = False
qns = []  # query names
tns = []  # target names (from ggsearch results)
for i, line in enumerate(lines):
    if '!! No sequences' in line:
        print(f"  Line {i}: No sequences found")
        tns.append(None)
    
    if not train_coming:
        if 'The best scores are:' in line:
            train_coming = True
    else:
        target_name = line.split()[0]
        print(f"  Line {i}: Found target: {target_name}")
        tns.append(target_name)
        train_coming = False
    
    if 'Library: ' in line:
        query_line = lines[i - 1]
        query_name = query_line.split('>')[-1].split()[0]
        print(f"  Line {i}: Query from '{query_line[:100]}' -> '{query_name}'")
        qns.append(query_name)

print(f"\nExtracted query names: {qns}")
print(f"Extracted target names: {tns}")

# Try looking them up
print("\n=== LOOKUP TEST ===")
for qn, tn in zip(qns, tns):
    print(f"\nQuery: {qn}")
    print(f"  In query_seqs? {qn in query_seqs}")
    if qn in query_seqs:
        print(f"    Seq length: {len(query_seqs[qn])}")
    
    if tn is not None:
        print(f"Target: {tn}")
        print(f"  In train_seqs? {tn in train_seqs}")
        if tn in train_seqs:
            print(f"    Seq length: {len(train_seqs[tn])}")
