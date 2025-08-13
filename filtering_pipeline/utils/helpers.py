import os
import sys
import pandas as pd
import logging
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO, Select
import re
from collections import Counter
from io import StringIO
from biotite.structure import AtomArrayStack
from biotite.structure.io.pdb import PDBFile
from rdkit import Chem

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

    return new_path

def generate_chai_structure_path(input_path):
    """
    Generate the structure file path of Chai structure based on chai output directory.
    """
    base_path = Path(input_path)
    base_name = base_path.name  
    new_path = base_path / 'chai' / f"{base_name}_0.cif"

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


class SingleLigandSelect(Select):
    def __init__(self, chain_id, resseq, resname=None):
        self.chain_id = chain_id
        self.resseq = resseq
        self.resname = (resname or "").strip()

    def accept_atom(self, atom):
        res = atom.get_parent()
        ch_match = res.get_parent().id == self.chain_id
        id_match = res.id[1] == self.resseq
        if self.resname:
            rn_match = res.get_resname().strip() == self.resname
        else:
            rn_match = True
        return ch_match and id_match and rn_match

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


def get_hetatm_chain_ids(pdb_path):
    with open(pdb_path, "r") as f:
        pdb_file = PDBFile.read(f)
    structure = pdb_file.get_structure()
    structure = structure[0]

    hetatm_chains = set(structure.chain_id[structure.hetero])
    atom_chains = set(structure.chain_id[~structure.hetero])

    # Exclude chains that also have ATOM records (i.e., protein chains)
    ligand_only_chains = hetatm_chains - atom_chains

    return list(ligand_only_chains)


def extract_chain_as_rdkit_mol(pdb_path, chain_id, sanitize=False):
    '''
    Extract ligand chain as RDKit mol objects given their chain ID. 
    '''
    # Read full structure
    with open(pdb_path, "r") as f:
        pdb_file = PDBFile.read(f)
    structure = pdb_file.get_structure()
    if isinstance(structure, AtomArrayStack):
        structure = structure[0]  # first model only

    # Extract chain
    mask = structure.chain_id == chain_id

    if len(mask) != structure.array_length():
        raise ValueError(f"Mask shape {mask.shape} doesn't match atom array length {structure.array_length()}")

    chain = structure[mask]

    if chain.shape[0] == 0:
        raise ValueError(f"No atoms found for chain {chain_id} in {pdb_path}")

    # Convert to PDB string using Biotite
    temp_pdb = PDBFile()
    temp_pdb.set_structure(chain)
    pdb_str_io = StringIO()
    temp_pdb.write(pdb_str_io)
    pdb_str = pdb_str_io.getvalue()

    # Convert to RDKit mol from PDB string
    mol = Chem.MolFromPDBBlock(pdb_str, sanitize=sanitize)

    return mol


def atom_composition_fingerprint(mol):
    """
    Returns a Counter of atom symbols in the molecule (e.g., {'C': 10, 'N': 2}).
    """
    return Counter([atom.GetSymbol() for atom in mol.GetAtoms()])


def norm_l1_dist(fp_a, fp_b, keys=None):
    """
    Normalized L1 distance on element counts. Used to pick the closest element-count vector
    of all ligands to the reference ligand. 
    """
    if keys is None:
        keys = set(fp_a) | set(fp_b)
    num = 0.0
    den = 0.0
    for k in keys:
        a = fp_a.get(k, 0)
        b = fp_b.get(k, 0)
        num += abs(a - b)
        den += a + b
    return 0.0 if den == 0 else num / den


def closest_ligands_by_element_composition(ligand_mols, reference_smiles, top_k = 2):
    """
    Filters a list of RDKit Mol objects based on atom element composition
    matching a reference SMILES. It returns a mol object that matches the element composition. 
    Because sometimes some atoms especially hydrogens can get lost in conversions, I pick the ligand
    with the closest atom composition to the reference; doesn't have to match perfectly. 
    """
    ref_mol = Chem.MolFromSmiles(reference_smiles)
    if ref_mol is None:
        raise ValueError("Reference SMILES could not be parsed.")

    # calculate atom composition of the reference smile string i.e. the ligand of interest
    ref_fp = atom_composition_fingerprint(ref_mol)

    out = []
    for mol in ligand_mols:
        if mol is None:
            continue
        try:
            fp = atom_composition_fingerprint(mol)
            dist = norm_l1_dist(ref_fp, fp)
            score = 1.0 - dist
            out.append((mol, score))
        except Exception as e:
            print(f"Error processing ligand: {e}")
            continue
    # return closest matching lgiands
    out.sort(key=lambda t: t[1], reverse=True)
    return [mol for mol, _ in out[:top_k]]



def ensure_3d(m: Chem.Mol) -> Chem.Mol:
    """Make sure we have a conformer (PDB usually has one; this is a fallback)."""
    if m is None:
        return None
    if m.GetNumConformers() == 0:
        m = Chem.AddHs(m)
        AllChem.EmbedMolecule(m, randomSeed=0xf00d)
        m = Chem.RemoveHs(m)
    return m

def as_mol(x):
    # In case anything returns (mol, score) or a dict
    if isinstance(x, Mol): return x
    if isinstance(x, tuple) and x and isinstance(x[0], Mol): return x[0]
    if isinstance(x, dict) and isinstance(x.get("mol"), Mol): return x["mol"]
    return None



























