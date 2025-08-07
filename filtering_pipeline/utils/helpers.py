import os
import sys
import pandas as pd
import logging
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO, Select
import re


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

# Define plotting aesthetics 
def clean_plt(ax):
    ax.tick_params(direction='out', length=2, width=1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['top'].set_linewidth(0)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['right'].set_linewidth(0)
    ax.tick_params(labelsize=10.0)
    ax.tick_params(axis='x', which='major', pad=2.0)
    ax.tick_params(axis='y', which='major', pad=2.0)
    return ax

# Define logging titles
def log_section(title: str):
    border = "#" * 60
    logger.info(f"\n{border}")
    logger.info(f"### {title.upper().center(52)} ###")
    logger.info(f"{border}\n")

# Define logging subtitles
def log_subsection(title: str):
    border = "#" * 60
    logger.info(f"\n{border}")
    logger.info(f"### {title.center(52)} ###")
    logger.info(f"{border}\n")

def log_boxed_note(text):
    border = "-" * (len(text) + 8)
    logger.info(f"\n{border}\n|   {text}   |\n{border}\n")

def generate_boltz_structure_path(input_path):
    """
    Generate the structure file path of Boltz structure based on boltz output directory.
    """
    base_path = Path(input_path)
    base_name = base_path.name  
    new_path = base_path / f"boltz_results_{base_name}" / "predictions" / base_name / f"{base_name}_model_0.cif"
    print(new_path)

    return new_path

def clean_protein_sequence(seq: str) -> str:
    """
    Cleans a protein sequence by:
    - Removing stop codons (*)
    - Removing whitespace or newline characters
    - Ensuring only valid amino acid letters remain (A-Z except B, J, O, U, X, Z)
    """
    if pd.isna(seq):
        return None
    seq = seq.upper()
    seq = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', seq)  # Keep only standard 20 amino acids
    return seq


def delete_empty_subdirs(directory):
    '''Delete empty subdirectories'''
    directory = Path(directory)
    for subdir in directory.iterdir():
        if subdir.is_dir() and not any(subdir.iterdir()):
            subdir.rmdir()
            print(f"Deleted empty directory: {subdir}")


class suppress_stdout_stderr:
    def __enter__(self):
        # Open a null file
        self.devnull = open(os.devnull, 'w')
        self.old_stdout = os.dup(1)
        self.old_stderr = os.dup(2)
        os.dup2(self.devnull.fileno(), 1)
        os.dup2(self.devnull.fileno(), 2)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.dup2(self.old_stdout, 1)
        os.dup2(self.old_stderr, 2)
        os.close(self.old_stdout)
        os.close(self.old_stderr)
        self.devnull.close()


class LigandSelect(Select):
    def __init__(self, ligand_resname):
        self.ligand_resname = ligand_resname

    def accept_residue(self, residue):
        return residue.get_resname() == self.ligand_resname


def extract_entry_name_from_PDB_filename(name):
    '''Extracts the entry name from a PDB filename of docked structures.
    '''
    suffix = name.rsplit('_', 1)[-1]
    parts = name.split('_')

    if suffix in {'boltz'}:
        # Return everything except the last 3 parts
        return '_'.join(parts[:-3])
    elif suffix in {'vina', 'chai'}:
        # Return everything except the last 2 parts
        return '_'.join(parts[:-2])
    else:
        return name  # fallback if unknown suffix


def extract_ligand_from_PDB(input_pdb, output_pdb, ligand_resname):
    """
    Extracts a ligand from a PDB file and writes it to a new PDB.

    Parameters:
    - input_pdb: str, path to the complex PDB file
    - output_pdb: str, path to write the ligand-only PDB file
    - ligand_resname: str, 3-letter residue name of the ligand (e.g., 'LIG')
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("docked", input_pdb)

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_pdb), LigandSelect(ligand_resname))


def add_metrics_to_best_structures(best_strucutures_df, df_dockmetrics):
    """
    Merges docking metrics from df_dockmetrics into best_strucutures_df based on the 'Entry' column.
    Extracts structure IDs and vina indices from the 'best_structure' column.
    """
    dict_columns = [
        "chai_aggregate_score", "chai_ptm", "chai_iptm",
        "chai_per_chain_ptm", "chai_per_chain_pair_iptm", 
        "chai_has_clashes", "chai_chain_chain_clashes", 
        "boltz2_confidence_score", "boltz2_ptm", "boltz2_iptm", 
        "boltz2_ligand_iptm", "boltz2_protein_iptm", 
        "boltz2_complex_plddt", "boltz2_complex_iplddt", 
        "boltz2_complex_pde", "boltz2_complex_ipde", 
        "boltz2_chains_ptm", "boltz2_pair_chains_iptm"
    ]

    def extract_structure_id(full_name):
        parts = full_name.split("_")
        if parts[-1] in {"vina", "chai", "boltz"}:
            return "_".join(parts[:-1])
        return full_name

    def extract_vina_index(structure):
        if structure.endswith("_vina"):
            try:
                return int(structure.split("_")[-2])
            except:
                return None
        return None

    df_dockmetrics_reduced = df_dockmetrics[["Entry"] + dict_columns + ["vina_affinities"]].drop_duplicates(subset="Entry")
    merged_df = pd.merge(best_strucutures_df, df_dockmetrics_reduced, on="Entry", how="left")

    # Extract structure ID and replace dict columns with values
    structure_ids = merged_df["best_structure"].map(extract_structure_id)

    for col in dict_columns:
        merged_df[col] = [
            d.get(structure_id) if isinstance(d, dict) else None
            for d, structure_id in zip(merged_df[col], structure_ids)
        ]

    # Extract vina affinity
    vina_indices = merged_df["best_structure"].map(extract_vina_index)

    merged_df["vina_affinity"] = [
        v.get(idx) if isinstance(v, dict) and idx is not None else None
        for v, idx in zip(merged_df["vina_affinities"], vina_indices)
    ]

    merged_df_final = merged_df.drop(columns=["vina_affinities"])

    return merged_df_final