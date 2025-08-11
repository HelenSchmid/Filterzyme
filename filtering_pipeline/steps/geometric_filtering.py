import os
import pandas as pd
from pathlib import Path
import logging
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
import math
import re
from Bio.PDB import PDBIO
from Bio.PDB import PDBParser, Select, PDBIO
from biotite.structure.io.pdb import PDBFile
from biotite.structure import AtomArrayStack
from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS, rdmolops
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.Draw import MolToImage
from rdkit.Chem.Draw import rdMolDraw2D # You'll need this for MolDraw2DCairo/SVG
from rdkit.Chem.Draw.rdMolDraw2D import MolDrawOptions
from rdkit.Geometry import Point3D
from rdkit import RDLogger
from itertools import product
from io import StringIO
import tempfile
from collections import Counter

from filtering_pipeline.steps.step import Step

RDLogger.DisableLog('rdApp.warning')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Dictionary to map residue names to their relevant atoms for catalysis or binding
atom_selection = {
    'CYS': ['SG'],         # Thiol group — nucleophile
    'SER': ['OG'],         # Hydroxyl group — nucleophile
    'THR': ['OG1'],        # Secondary hydroxyl — nucleophile
    'TYR': ['OH'],         # Phenolic hydroxyl — acid/base or H-bond donor/acceptor
    'ASP': ['OD1'],        # Carboxylate — acid/base
    'GLU': ['OE1'],        # Carboxylate — acid/base
    'HIS': ['ND1'],        # Imidazole nitrogens — acid/base catalysis, H-bonding
    'LYS': ['NZ'],         # Terminal amine — nucleophile, acid/base
    'ARG': ['CZ'],         # Guanidinium — often stabilizes charges or binds anions
    'ASN': ['ND2'],        # Amide nitrogen — can form H-bonds
    'GLN': ['NE2'],        # Amide nitrogen — similar to ASN
    'TRP': ['NE1'],        # Indole nitrogen — H-bond donor/acceptor
    'MET': ['SD'],         # Thioether — occasionally involved in redox
    'PRO': ['N'],          # Backbone nitrogen — sometimes key in transition states
    'ALA': [],             # Non-polar, not typically catalytic
    'VAL': [],             # Non-polar
    'LEU': [],             # Non-polar
    'ILE': [],             # Non-polar
    'PHE': [],             # Aromatic, non-polar
    'GLY': []              # No side chain; may participate via backbone flexibility
}


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

def _norm_l1_dist(fp_a, fp_b, keys=None):
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

def closest_ligands_by_element_composition(ligand_mols, reference_smiles, include_h=False, top_k = 2):
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
            dist = _norm_l1_dist(ref_fp, fp)
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

def assign_bond_orders_from_smiles(pdb_mol, ligand_smiles):
    """
    Transfer bond orders from SMILES to a PDB ligand. Ignore hydrogens and
    stereochemistry. Assign aromaticity based on SMILES. Keep 3D coordinates. 
    """
    ref = Chem.MolFromSmiles(ligand_smiles)
    if ref is None:
        return pdb_mol

    # Work on **heavy-atom graphs** only
    ref0 = Chem.RemoveHs(ref)                
    pdb0 = Chem.RemoveHs(Chem.Mol(pdb_mol), sanitize=False)  

    # Kekulize template to transfer explicit single/double bonds
    ref0_kek = Chem.Mol(ref0)
    rdmolops.Kekulize(ref0_kek, clearAromaticFlags=True)

    try:
        # Assign bond orders on the heavy-atom PDB ligand
        new0 = AllChem.AssignBondOrdersFromTemplate(ref0_kek, pdb0)

        # Drop all stereochemistry (you said you don't want it)
        Chem.RemoveStereochemistry(new0)

        # Recompute aromaticity from assigned bonds
        Chem.SanitizeMol(
            new0,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_SYMMRINGS
                      | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
                      | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
        )

        # Restore the original 3D conformer
        if pdb_mol.GetNumConformers():
            conf = pdb0.GetConformer() if pdb0.GetNumConformers() else pdb_mol.GetConformer()
            new0.RemoveAllConformers()
            new0.AddConformer(conf, assignId=True)

        return new0  # heavy-atom ligand with correct bond orders & arom, no stereo

    except Exception as e:
        print("AssignBondOrdersFromTemplate failed:", e)
        return pdb_mol

def find_substructure_matches(mol, sub, is_smarts=False, use_chirality=False):
    """
    sub: SMILES (default) or SMARTS (if is_smarts=True).
    Returns list of tuples of atom indices.
    """
    q = Chem.MolFromSmarts(sub) if is_smarts else Chem.MolFromSmiles(sub)
    if q is None:
        raise ValueError("Could not parse substructure pattern.")
    return list(mol.GetSubstructMatches(q, useChirality=use_chirality, uniquify = True))

def coords_of_atoms(mol, atom_indices):
    conf = mol.GetConformer()
    pts = [conf.GetAtomPosition(i) for i in atom_indices]
    return np.array([[p.x, p.y, p.z] for p in pts])

def centroid_from_match(mol: Chem.Mol, match, confId: int = 0):
    """
    Compute the 3D centroid (x,y,z) of a single substructure match ( = tuple of atom indices).
    """
    if mol is None or mol.GetNumConformers() == 0 or not match:
        return None
    conf = mol.GetConformer(confId)
    pts = np.array([[conf.GetAtomPosition(i).x,
                     conf.GetAtomPosition(i).y,
                     conf.GetAtomPosition(i).z] for i in match], dtype=float)
    return tuple(pts.mean(axis=0))

def nearest_centroid_distance(A, B):
    """
    Smallest pairwise distance between two centroid lists A and B (each list of (x,y,z)).
    """
    if not A or not B:
        return None
    A = np.array(A, float); B = np.array(B, float)
    D = np.sqrt(((A[:, None, :] - B[None, :, :])**2).sum(axis=2))
    return float(D.min())




def get_squidly_residue_atom_coords(pdb_path: str, residue_id_str: str):
    '''    
    Extracts the 3D coordinates of all atoms in specified residues from a PDB file.
    residue_id_str (str): Residue IDs as a pipe-separated string (e.g. '10|25|33'), indexed from 0.
    Returns: Dictionary where keys are residue identifiers (e.g. 'LYS_26') and values are lists of atom info.
    '''
    # Convert residue string IDs from 0-indexed to 1-indexed PDB format
    residue_ids_raw = residue_id_str.split('|')
    residue_ids = []
    for rid in residue_ids_raw:
        rid_stripped = rid.strip()
        if rid_stripped.lower() in ('nan', '', None):
            continue
        try:
            residue_ids.append(int(rid_stripped) + 1)
        except (ValueError, TypeError):
            continue

    matching_residues = {}

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                res_name = line[17:20].strip()
                res_id = int(line[22:26].strip())
                atom_name = line[12:16].strip()

                if res_id in residue_ids:
                    key = f"{res_name}_{res_id}"
                    if key not in matching_residues:
                        matching_residues[key] = []

                    x = float(line[30:38]) # Residue name
                    y = float(line[38:46]) # Residue number 
                    z = float(line[46:54])  # Atom name

                    matching_residues[key].append({
                        'atom': atom_name,
                        'coords': (x, y, z)
                    })

    return matching_residues

def filter_residue_atoms(residue_atom_dict, atom_selection_map = atom_selection):
    """
    Filters the atom coordinates of specific atoms for each residue type.
    Inputs:
        residue_atom_dict (dict): Output from get_squidly_residue_atom_coords().
        atom_selection_map (dict): Mapping from residue name to atom name to extract.
    Returns: Dictionary of filtered atoms per residue.
    """
    filtered = {}

    for residue_key, atoms in residue_atom_dict.items():
        res_name, res_id = residue_key.split('_')

        # Only proceed if this residue type is in our selection map
        if res_name in atom_selection_map:
            ligand_atoms = atom_selection_map[res_name]
            for atom in atoms:
                if atom['atom'] in ligand_atoms:
                    if residue_key not in filtered:
                        filtered[residue_key] = []
                    filtered[residue_key].append(atom)

    return filtered

def find_min_distance(ester_dict, squidly_dict): 
    """
    Find the minimum distance between any ester atom (from multiple ester substructures)
    and any nucleophile atom (e.g. from squidly).
    """
    min_dist = float('inf')
    closest_info = None

    for ester_label, ester_atoms in ester_dict.items():
        for ester_atom in ester_atoms:
            coord1 = np.array(ester_atom['coords'])
            lig_atom = ester_atom['atom']

            for nuc_res, nuc_atoms in squidly_dict.items():
                for nuc_atom in nuc_atoms:
                    coord2 = np.array(nuc_atom['coords'])
                    dist = np.linalg.norm(coord1 - coord2)


                    if dist < min_dist:
                        min_dist = dist
                        closest_info = {
                            'ligand_atom': lig_atom,
                            'ligand_substructure': ester_label,
                            'ligand_coords': coord1,  
                            'nuc_res': nuc_res,
                            'nuc_atom': nuc_atom['atom'],
                            'nuc_coords': coord2,     
                            'distance': dist
                        }

    return closest_info

def find_min_distance_per_squidly(ester_dict, squidly_dict):
    closest_by_residue = {}

    for nuc_res, nuc_atoms in squidly_dict.items():
        min_dist = float('inf')
        closest_info = None

        for nuc_atom in nuc_atoms:
            coord2 = np.array(nuc_atom['coords'])

            for ester_label, ester_atoms in ester_dict.items():
                for ester_atom in ester_atoms:
                    coord1 = np.array(ester_atom['coords'])
                    dist = np.linalg.norm(coord1 - coord2)

                    if dist < min_dist:
                        min_dist = dist
                        closest_info = {
                            'ligand_atom': ester_atom['atom'],
                            'ligand_substructure': ester_label,
                            'ligand_coords': coord1,
                            'nuc_res': nuc_res,
                            'nuc_atom': nuc_atom['atom'],
                            'nuc_coords': coord2,
                            'distance': dist
                        }

        if closest_info:
            closest_by_residue[nuc_res] = closest_info

    return closest_by_residue

def get_all_nucs_atom_coords(pdb_path: str):
    """
    Extracts all nucleophilic residues (Ser, Cys) from a PDB file.
    Returns a dictionary with residue names as keys and lists of their atom coordinates.
    """
    nucs = ["SER", "CYS"]
    matching_residues = {}

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith(('ATOM')):
                res_name = line[17:20].strip()
                res_id = line[22:26].strip()
                atom_name = line[12:16].strip()

                
                if res_name == "SER" and atom_name == "OG":
                    # For SER, we are interested in the OG atom
                    key = f"{res_name}_{res_id}"
                    if key not in matching_residues:
                        matching_residues[key] = []

                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])

                    matching_residues[key].append({
                        'atom': atom_name,
                        'coords': (x, y, z)
                    })
                
                elif res_name == "CYS" and atom_name == "SG":
                    # For CYS, we are interested in the SG atom
                    key = f"{res_name}_{res_id}"
                    if key not in matching_residues:
                        matching_residues[key] = []

                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])

                    matching_residues[key].append({
                        'atom': atom_name,
                        'coords': (x, y, z)
                    })

    return matching_residues

def filter_nucleophilic_residues(residue_atom_dict):
    """
    Filters input residues and returns only those that contain known nucleophilic atoms as a dictionary.

    """
    allowed_resnames = {'SER', 'CYS', 'TYR', 'HIS', 'LYS', 'GLU'}
    filtered = {}

    for residue_id, atoms in residue_atom_dict.items():
        res_name, _ = residue_id.split('_', 1)
        if res_name in allowed_resnames:
            filtered[residue_id] = atoms

    return filtered

def calculate_residue_ligand_distance(ligand_group_dict, residue_dict): 
    """
    Calculates distance between any atom of interest (from the ligand group) and any catalytic residue (e.g. from squidly prediction).
    If the ligand dictionary contains multiple substructures, it will return the closest distance.
    """
    min_dist = float('inf')
    closest_info = None

    for residues, residue_atoms in residue_dict.items():
        for nuc_atom in residue_atoms:
            coord_res = np.array(nuc_atom['coords'])
            residue_atom = nuc_atom['atom']

            for ligand_label, ligand_atoms in ligand_group_dict.items():
                for target_atom in ligand_atoms:
                    coord_target = np.array(target_atom['coords'])
                    lig_atom = target_atom['atom']

                    dist = np.linalg.norm(coord_res - coord_target)

                    if dist < min_dist:
                        min_dist = dist
                        closest_info = {
                            'ligand_atom': lig_atom,
                            'ligand_substructure': ligand_label,
                            'ligand_coords': coord_target,  
                            'nuc_res': residues,
                            'nuc_atom': residue_atom,
                            'nuc_coords': coord_res,     
                            'distance': dist
                        }
    return closest_info

def calculate_dihedral_angle(p1, p2, p3, p4):
    """
    Calculates the dihedral angle between four 3D points.
    Returns the angle in degrees.
    """
    b0 = -1.0 * (np.array(p2) - np.array(p1))
    b1 = np.array(p3) - np.array(p2)
    b2 = np.array(p4) - np.array(p3)

    # Normalize b1 so that it does not influence magnitude of vector
    b1 /= np.linalg.norm(b1)

    # Orthogonal vectors
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)

    return np.degrees(np.arctan2(y, x))

def calculate_burgi_dunitz_angle(atom_nu_coords, atom_c_coords, atom_o_coords):
    """
    Calculates the Bürgi-Dunitz angle. Defined by the nucleophilic atom (Nu),
    the electrophilic carbonyl carbon (C), and one of the carbonyl oxygen atoms (O).
    """
    # Vectors from carbonyl carbon to nucleophile and to carbonyl oxygen
    vec_c_nu = atom_nu_coords - atom_c_coords
    vec_c_o = atom_o_coords - atom_c_coords

    # Calculate the dot product
    dot_product = np.dot(vec_c_nu, vec_c_o)

    # Calculate the magnitudes of the vectors
    magnitude_c_nu = np.linalg.norm(vec_c_nu)
    magnitude_c_o = np.linalg.norm(vec_c_o)

    # Calculate the cosine of the angle
    cos_angle = dot_product / (magnitude_c_nu * magnitude_c_o)

    # Ensure cos_angle is within valid range [-1, 1] to prevent arccos errors due to floating point inaccuracies
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # Calculate the angle in radians and then convert to degrees
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    return angle_deg


class GeneralGeometricFiltering(Step):

    def __init__(self, preparedfiles_dir: str = '', esterase = 0, find_closest_nucleophile = 0, output_dir: str= ''):
        self.esterase = esterase
        self.find_closest_nuc = find_closest_nucleophile
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.preparedfiles_dir = Path(preparedfiles_dir)


    def __execute(self, df: pd.DataFrame, tmp_dir: str) -> list:

        if not self.preparedfiles_dir.exists():
            raise FileNotFoundError(f"Directory does not exist: {self.preparedfiles_dir}")
        
        results = []

        for _, row in df.iterrows():
            entry_name = row['Entry']
            best_structure_name = row['best_structure']
            squidly_residues = str(row['Squidly_CR_Position'])
            substrate_smiles = row['substrate_smiles']
            cofactor_smiles = row['cofactor_smiles']
            substrate_moiety = row['substrate_moiety']
            cofactor_moiety = row['cofactor_moiety']
            row_result = {}

            default_result = {
                'distance_ligand_to_cofactor': None, 
                'distance_ligand_to_squidly_residues': None,
                'distance_ligand_to_closest_nuc': None,
                'Bürgi–Dunitz_angle_to_squidly_residue': None,
                'Bürgi–Dunitz_angle_to_closest_nucleophile': None
            }
            try: 
                # Load full PDB structure
                pdb_file = self.preparedfiles_dir / f"{best_structure_name}.pdb"
                pdb_file = Path(pdb_file)
                print(f"Processing PDB file: {pdb_file.name}")

                # Extract chain IDs of ligands
                chain_ids = get_hetatm_chain_ids(pdb_file)

                # Extract ligands as RDKit mol objects
                ligands = []
                for chain_id in chain_ids:
                    mol  = extract_chain_as_rdkit_mol(pdb_file, chain_id, sanitize=False)
                    ligands.append(mol)

                # Get substrate mol object and assign correct bond order based on smiles
                ligand_candidate = closest_ligands_by_element_composition(ligands, substrate_smiles, top_k=1)
                ligand_mol = as_mol(ligand_candidate[0]) if ligand_candidate else None
                ligand_mol = assign_bond_orders_from_smiles(ligand_mol, substrate_smiles)
                ligand_mol  = ensure_3d(ligand_mol)

                # Find ligand substructure match with moiety of interest
                ligand_match = find_substructure_matches(ligand_mol, substrate_moiety)

                # Calculate ligand-moiety centroid
                ligand_centroid = centroid_from_match(ligand_mol, ligand_match)

                if not ligand_centroid:
                    # optional fallback: whole-ligand centroid
                    logger.warning(f"Ligand-substructure centroid calculation unsuccessfull. Use whole-ligand centroid instead.")
                    ligand_centroid = [centroid_from_match(ligand_mol, tuple(range(ligand_mol.GetNumAtoms())))]


                # --- Distance between ligand and cofactor ---
                # Get cofactor mol object and assign correct bond order based on smiles
                if 'cofactor_smiles' in df.columns and df['cofactor_smiles'] is not None:              
                    cofactor_candidate = closest_ligands_by_element_composition(ligands, cofactor_smiles, top_k=1)
                    cofactor_mol = as_mol(cofactor_candidate[0]) if cofactor_candidate else None
                    cofactor_mol = as_mol(cofactor_candidate[0]) if cofactor_candidate else None
                    cofactor_mol = assign_bond_orders_from_smiles(cofactor_mol, cofactor_smiles)
                    cofactor_mol = ensure_3d(cofactor_mol)

                    # Find cofactor substructure match with moiety of interest
                    cofactor_match = find_substructure_matches(cofactor_mol, cofactor_moiety)

                    # Calculate cofactor-moiety centroid
                    cofactor_centroid = centroid_from_match(cofactor_mol, cofactor_match)
                    if not cofactor_centroid:
                        logger.warning(f"Cofactor-substructure centroid calculation unsuccessfull. Use whole-cofactor centroid instead.")
                        cofactor_centroid = [centroid_from_match(cofactor_mol, tuple(range(cofactor_mol.GetNumAtoms())))]


                    # Minimum distance between centroids
                    ligand_cofactor_distance = nearest_centroid_distance(ligand_centroid, cofactor_centroid)
                    print(ligand_cofactor_distance)
                    if not ligand_cofactor_distance: 
                        logger.warning(f"Ligand-cofactor distance calculation was unsuccessful.")
                        row_result.update(default_result)
                        results.append(row_result)
                        continue
                    row_result['distance_ligand_to_cofactor'] = ligand_cofactor_distance

                # --- Distance between squidly residues and ligand ---

                # Get squidly protein atom coordinates
                squidly_atom_coords = get_squidly_residue_atom_coords(pdb_file, squidly_residues)
                filtered_squidly_atom_coords = filter_residue_atoms(squidly_atom_coords, atom_selection)

                if not squidly_atom_coords:
                    logger.warning(f"No squidly residues found in {entry_name}.")
                    row_result.update(default_result)
                    results.append(row_result)
                    continue

                # Compute distances between squidly predicted residues and ligand
                squidly_distance = find_min_distance_per_squidly(ligand_centroids, filtered_squidly_atom_coords)

                # store distances in a dictionary
                if squidly_distance:
                    squidly_dist_dict = {res_name: match_info['distance'] for res_name, match_info in squidly_distance.items()}
                    row_result['distance_ligand_to_squidly_residues'] = squidly_dist_dict

                # --- Find closest nucleophile overall
                if self.find_closest_nuc == 1: 
                    all_nucleophiles_coords = get_all_nucs_atom_coords(pdb_file) # Get all nucleophilic residues atom coordinates
                    closest_distance = find_min_distance(ligand_centroids, all_nucleophiles_coords) # Compute smallest distances between all nucleophilic residues and ligand

                    if closest_distance:
                        closest_nuc_dict = {closest_distance['nuc_res']: closest_distance['distance']}
                        row_result['distance_ligand_to_closest_nuc'] = closest_nuc_dict


                ### TO DO: TRY SUBSTRUCTURE MATCHING AGAIN NOW THAT ARE ACTUALLY USING SMARTS
                ### TO DO: ADAPT HOW WE CALCULATE THE DISTANCE BETWEEN SQUIDLY AND LIGAND
                ## TO DO: HOW CAN WE HANDLE IF SEVERAL TIMES THE SAME MOIETY IS FOUND IN THE LIGAND FOR MSC?

                """"
                # --- Calculate Bürgi–Dunitz angle between closest nucleophile and ester bond
                if self.esterase == 1: 
                    try: 
                        oxygen_atom_coords = find_substructure_coordinates(extracted_ligand_atoms, self.smarts_pattern, atom_to_get_coords_idx=0) # atom1 from SMARTS match (e.g., double bonded O)
                        
                        # Angle between nucleophilic squidly residues and ester bond
                        nuc_squidly_atom_coords = filter_nucleophilic_residues(filtered_squidly_atom_coords)

                        bd_angles_to_squidly = {}

                        for res_name, atoms in nuc_squidly_atom_coords.items():
                            if not atoms:
                                continue

                            nuc_atom_coords = np.array(atoms[0]['coords'])
                            ligand_coords_list = list(ligand_coords.values())[0]
                            oxygen_coords_list = list(oxygen_atom_coords.values())[0]

                            if not ligand_coords_list or not oxygen_coords_list:
                                continue

                            ligand_c_coords = np.array(ligand_coords_list[0]['coords'])
                            oxygen_coords = np.array(oxygen_coords_list[0]['coords'])

                            angle = calculate_burgi_dunitz_angle(nuc_atom_coords, ligand_c_coords, oxygen_coords)

                            # Store angle in dictionary
                            bd_angles_to_squidly[res_name] = angle

                        if bd_angles_to_squidly:
                            row_result['Bürgi–Dunitz_angles_to_squidly_residues'] = bd_angles_to_squidly

                        # Single angle to closest nucleophile as dictionary
                        if closest_distance:
                            closest_angle_info = {
                                closest_distance['nuc_res']: calculate_burgi_dunitz_angle(
                                    np.array(closest_distance['nuc_coords']),
                                    ligand_c_coords,
                                    oxygen_coords
                                )
                            }
                            row_result['Bürgi–Dunitz_angle_to_closest_nucleophile'] = closest_angle_info
                            
                    except Exception as e:
                        logger.error(f"Error processing {entry_name}: {e}")
                        row_result.update(default_result)
            """                 
            except Exception as e:
                logger.error(f"Error processing {entry_name}: {e}")
                row_result.update(default_result)
            
            results.append(row_result)

        return results
    

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.output_dir:
            print("No output directory provided")
            return df

        results = self.__execute(df, self.output_dir)        
        results_df = pd.DataFrame(results) # Convert list of row-dictionaries to df       
        output_df = pd.concat([df.reset_index(drop=True), results_df], axis=1) # Merge with input df

        return output_df






class EsteraseGeometricFiltering(Step):

    def __init__(self, preparedfiles_dir: str = '', esterase = 0, find_closest_nucleophile = 0, output_dir: str= ''):
        self.esterase = esterase
        self.find_closest_nuc = find_closest_nucleophile
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.preparedfiles_dir = Path(preparedfiles_dir)


    def __execute(self, df: pd.DataFrame, tmp_dir: str) -> list:

        if not self.preparedfiles_dir.exists():
            raise FileNotFoundError(f"Directory does not exist: {self.preparedfiles_dir}")
        
        results = []

        for _, row in df.iterrows():
            entry_name = row['Entry']
            best_structure_name = row['best_structure']
            squidly_residues = str(row['Squidly_CR_Position'])
            substrate_smiles = row['substrate_smiles']
            cofactor_smiles = row['cofactor_smiles']
            substrate_moiety = row['substrate_moiety']
            cofactor_moiety = row['cofactor_moiety']
            row_result = {}

            default_result = {
                'distance_ligand_to_cofactor': None, 
                'distance_ligand_to_squidly_residues': None,
                'distance_ligand_to_closest_nuc': None,
                'Bürgi–Dunitz_angle_to_squidly_residue': None,
                'Bürgi–Dunitz_angle_to_closest_nucleophile': None
            }
            try: 
                # Load full PDB structure
                pdb_file = self.preparedfiles_dir / f"{best_structure_name}.pdb"
                pdb_file = Path(pdb_file)
                print(f"Processing PDB file: {pdb_file.name}")

                # Extract chain IDs of ligands
                chain_ids = get_hetatm_chain_ids(pdb_file)

                # Extract ligands as RDKit mol objects
                ligands = []
                for chain_id in chain_ids:
                    mol  = extract_chain_as_rdkit_mol(pdb_file, chain_id, sanitize=False)
                    ligands.append(mol)

                # Get substrate mol object and assign correct bond order based on smiles
                ligand_candidate = closest_ligands_by_element_composition(ligands, substrate_smiles, top_k=1)
                ligand_mol = as_mol(ligand_candidate[0]) if ligand_candidate else None
                ligand_mol = assign_bond_orders_from_smiles(ligand_mol, substrate_smiles)
                ligand_mol  = ensure_3d(ligand_mol)

                # Find ligand substructure match with moiety of interest
                ligand_centroids  = find_substructure_matches(ligand_mol, substrate_moiety)

                # Calculate ligand-moiety centroid

                # Get cofactor mol objecgt and assign correct bond order based on smiles
                if 'cofactor_smiles' in df.columns and df['cofactor_smiles'] is not None:              
                    cofactor_candidate = closest_ligands_by_element_composition(ligands, cofactor_smiles, top_k=1)
                    cofactor_mol = as_mol(cofactor_candidate[0]) if cofactor_candidate else None
                    cofactor_mol = as_mol(cofactor_candidate[0]) if cofactor_candidate else None
                    cofactor_mol = assign_bond_orders_from_smiles(cofactor_mol, cofactor_smiles)
                    cofactor_mol = ensure_3d(cofactor_mol)

                    # Find cofactor substructure match with moiety of interest
                    cofactor_centroids = find_substructure_matches(cofactor_mol, cofactor_moiety)

                    # Calculate cofactor-moiety centroid




                

                # Distance between centroids
                ligand_cofactor_distance = nearest_centroid_distance(ligand_centroids, cofactor_centroids)
                row_result['distance_ligand_to_cofactor'] = ligand_cofactor_distance

                # Get squidly protein atom coordinates
                squidly_atom_coords = get_squidly_residue_atom_coords(pdb_file, squidly_residues)
                filtered_squidly_atom_coords = filter_residue_atoms(squidly_atom_coords, atom_selection)

                if not squidly_atom_coords:
                    logger.warning(f"No squidly residues found in {entry_name}.")
                    row_result.update(default_result)
                    results.append(row_result)
                    continue

                # Compute distances between squidly predicted residues and ligand
                squidly_distance = find_min_distance_per_squidly(ligand_centroids, filtered_squidly_atom_coords)

                # store distances in a dictionary
                if squidly_distance:
                    squidly_dist_dict = {res_name: match_info['distance'] for res_name, match_info in squidly_distance.items()}
                    row_result['distance_ligand_to_squidly_residues'] = squidly_dist_dict

                # --- Find closest nucleophile overall
                if self.find_closest_nuc == 1: 
                    all_nucleophiles_coords = get_all_nucs_atom_coords(pdb_file) # Get all nucleophilic residues atom coordinates
                    closest_distance = find_min_distance(ligand_centroids, all_nucleophiles_coords) # Compute smallest distances between all nucleophilic residues and ligand

                    if closest_distance:
                        closest_nuc_dict = {closest_distance['nuc_res']: closest_distance['distance']}
                        row_result['distance_ligand_to_closest_nuc'] = closest_nuc_dict


                ### TO DO: TRY SUBSTRUCTURE MATCHING AGAIN NOW THAT ARE ACTUALLY USING SMARTS
                ### TO DO: ADAPT HOW WE CALCULATE THE DISTANCE BETWEEN SQUIDLY AND LIGAND
                ## TO DO: HOW CAN WE HANDLE IF SEVERAL TIMES THE SAME MOIETY IS FOUND IN THE LIGAND FOR MSC?

                """"
                # --- Calculate Bürgi–Dunitz angle between closest nucleophile and ester bond
                if self.esterase == 1: 
                    try: 
                        oxygen_atom_coords = find_substructure_coordinates(extracted_ligand_atoms, self.smarts_pattern, atom_to_get_coords_idx=0) # atom1 from SMARTS match (e.g., double bonded O)
                        
                        # Angle between nucleophilic squidly residues and ester bond
                        nuc_squidly_atom_coords = filter_nucleophilic_residues(filtered_squidly_atom_coords)

                        bd_angles_to_squidly = {}

                        for res_name, atoms in nuc_squidly_atom_coords.items():
                            if not atoms:
                                continue

                            nuc_atom_coords = np.array(atoms[0]['coords'])
                            ligand_coords_list = list(ligand_coords.values())[0]
                            oxygen_coords_list = list(oxygen_atom_coords.values())[0]

                            if not ligand_coords_list or not oxygen_coords_list:
                                continue

                            ligand_c_coords = np.array(ligand_coords_list[0]['coords'])
                            oxygen_coords = np.array(oxygen_coords_list[0]['coords'])

                            angle = calculate_burgi_dunitz_angle(nuc_atom_coords, ligand_c_coords, oxygen_coords)

                            # Store angle in dictionary
                            bd_angles_to_squidly[res_name] = angle

                        if bd_angles_to_squidly:
                            row_result['Bürgi–Dunitz_angles_to_squidly_residues'] = bd_angles_to_squidly

                        # Single angle to closest nucleophile as dictionary
                        if closest_distance:
                            closest_angle_info = {
                                closest_distance['nuc_res']: calculate_burgi_dunitz_angle(
                                    np.array(closest_distance['nuc_coords']),
                                    ligand_c_coords,
                                    oxygen_coords
                                )
                            }
                            row_result['Bürgi–Dunitz_angle_to_closest_nucleophile'] = closest_angle_info
                            
                    except Exception as e:
                        logger.error(f"Error processing {entry_name}: {e}")
                        row_result.update(default_result)
            """                 
            except Exception as e:
                logger.error(f"Error processing {entry_name}: {e}")
                row_result.update(default_result)
            
            results.append(row_result)

        return results
    

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.output_dir:
            print("No output directory provided")
            return df

        results = self.__execute(df, self.output_dir)        
        results_df = pd.DataFrame(results) # Convert list of row-dictionaries to df       
        output_df = pd.concat([df.reset_index(drop=True), results_df], axis=1) # Merge with input df

        return output_df


