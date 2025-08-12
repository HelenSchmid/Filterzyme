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

    # Only heavy atoms
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

        return new0 

    except Exception as e:
        print("AssignBondOrdersFromTemplate failed:", e)
        return pdb_mol

def find_substructure_matches(mol, sub, is_smarts=False, use_chirality=False):
    """
    Input can be SMILES (default) or SMARTS (if is_smarts=True).
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

def centroid_from_indices(mol: Chem.Mol, atom_indices, confId: int = 0):
    """
    Accepts an int (single atom) or an iterable of atom indices.
    Returns (x, y, z) or None.
    """
    if mol is None or mol.GetNumConformers() == 0:
        return None

    # normalize to list of ints
    if isinstance(atom_indices, (int, np.integer)):
        idxs = [int(atom_indices)]
    else:
        try:
            idxs = [int(i) for i in atom_indices]
        except TypeError:
            raise TypeError(f"atom_indices must be int or iterable of ints, got {type(atom_indices)}")

    conf = mol.GetConformer(confId)
    pts = np.array([[conf.GetAtomPosition(i).x,
                     conf.GetAtomPosition(i).y,
                     conf.GetAtomPosition(i).z] for i in idxs], dtype=float)
    return tuple(pts.mean(axis=0))

def centroids_from_matches(mol: Chem.Mol, matches, confId: int = 0):
    return [centroid_from_indices(mol, m, confId=confId) for m in matches]

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
    Returns a dict where keys are residue identifiers (e.g 'LYS_26') and values are lists of atom info.
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

def find_min_distance_per_squidly(ligand_centroids, squidly_dict):
    """
    Calculate minimum distance for each squidly residue and the ligand centroids.
    """
    closest_by_residue = {}

    if not ligand_centroids:
        return closest_by_residue

    # ensure numpy array of ligand centroids
    lig = [np.array(c, dtype=float) for c in ligand_centroids if c is not None]

    for nuc_res, nuc_atoms in squidly_dict.items():
        min_dist = float('inf')
        closest_info = None

        for nuc_atom in nuc_atoms:
            coord2 = np.array(nuc_atom['coords'], dtype=float)

            for i, c in enumerate(lig):
                dist = np.linalg.norm(c - coord2)
                if dist < min_dist:
                    min_dist = dist
                    closest_info = {
                        'ligand_atom': f'centroid_{i}',
                        'ligand_substructure': f'centroid_{i}',
                        'ligand_coords': c,
                        'nuc_res': nuc_res,
                        'nuc_atom': nuc_atom['atom'],
                        'nuc_coords': coord2,
                        'distance': float(dist)
                    }

        if closest_info:
            closest_by_residue[nuc_res] = closest_info

    return closest_by_residue

def find_min_distance(ligand_centroids, squidly_dict):
    """
    Min distance between any ligand centroid and any nucleophile atom.
    Returns a dict or None if inputs are empty.
    """
    if not ligand_centroids or not squidly_dict:
        return None

    min_dist = float('inf')
    closest_info = None

    # preconvert centroids -> ndarray
    lig = [np.asarray(c, dtype=float) for c in ligand_centroids if c is not None]

    for i, c in enumerate(lig):
        lig_label = f"centroid_{i}"
        for nuc_res, nuc_atoms in squidly_dict.items():
            for nuc_atom in nuc_atoms:
                coord2 = np.asarray(nuc_atom['coords'], dtype=float)
                dist = float(np.linalg.norm(c - coord2))
                if dist < min_dist:
                    min_dist = dist
                    closest_info = {
                        'ligand_atom': lig_label,
                        'ligand_substructure': lig_label,
                        'ligand_coords': c,
                        'nuc_res': nuc_res,
                        'nuc_atom': nuc_atom['atom'],
                        'nuc_coords': coord2,
                        'distance': dist
                    }
    return closest_info

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


class GeneralGeometricFiltering(Step):

    def __init__(self, preparedfiles_dir: str = '', esterase = 0, find_closest_nucleophile = 0, output_dir: str= ''):

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
            catalytic_residues = str(row['catalytic_residues'])
            substrate_smiles = row['substrate_smiles']
            cofactor_smiles = row['cofactor_smiles']
            substrate_moiety = row['substrate_moiety']
            cofactor_moiety = row['cofactor_moiety']
            tool = row['tool']
            row_result = {}

            default_result = {
                'distance_ligand_to_cofactor': None, 
                'distance_ligand_to_catalytic_residues': None,
                'distance_cofactor_to_catalytic_residues': None, 
                'distance_ligand_to_closest_nuc': None,
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
                ligand_centroid = centroids_from_matches(ligand_mol, ligand_match)

                if not ligand_centroid:
                    # optional fallback: whole-ligand centroid
                    logger.warning(f"Ligand-substructure centroid calculation unsuccessfull. Use whole-ligand centroid instead.")
                    ligand_centroid = centroids_from_matches(ligand_mol, tuple(range(ligand_mol.GetNumAtoms())))


                # --- Distance between ligand and cofactor ---
                # Get cofactor mol object and assign correct bond order based on smiles
                if 'cofactor_smiles' in df.columns and cofactor_smiles is not None and tool != 'vina':              
                    cofactor_candidate = closest_ligands_by_element_composition(ligands, cofactor_smiles, top_k=1)
                    cofactor_mol = as_mol(cofactor_candidate[0]) if cofactor_candidate else None
                    cofactor_mol = as_mol(cofactor_candidate[0]) if cofactor_candidate else None
                    cofactor_mol = assign_bond_orders_from_smiles(cofactor_mol, cofactor_smiles)
                    cofactor_mol = ensure_3d(cofactor_mol)

                    # Find cofactor substructure match with moiety of interest
                    cofactor_match = find_substructure_matches(cofactor_mol, cofactor_moiety)

                    # Calculate cofactor-moiety centroid
                    cofactor_centroid = centroids_from_matches(cofactor_mol, cofactor_match)
                    if not cofactor_centroid:
                        logger.warning(f"Cofactor-substructure centroid calculation unsuccessfull. Use whole-cofactor centroid instead.")
                        cofactor_centroid = centroids_from_matches(cofactor_mol, tuple(range(cofactor_mol.GetNumAtoms())))

                    # Minimum distance between centroids
                    ligand_cofactor_distance = nearest_centroid_distance(ligand_centroid, cofactor_centroid)
                    if not ligand_cofactor_distance: 
                        logger.warning(f"Ligand-cofactor distance calculation was unsuccessful.")
                        row_result.update(default_result)
                        results.append(row_result)
                        continue
                    row_result['distance_ligand_to_cofactor'] = ligand_cofactor_distance

                # --- Distance between catalytic residues and ligand ---

                # Get squidly protein atom coordinates
                squidly_atom_coords = get_squidly_residue_atom_coords(pdb_file, catalytic_residues)
                filtered_squidly_atom_coords = filter_residue_atoms(squidly_atom_coords, atom_selection)

                if not squidly_atom_coords:
                    logger.warning(f"No squidly residues found in {entry_name}.")
                    row_result.update(default_result)
                    results.append(row_result)
                    continue

                # Compute distances between squidly predicted residues and ligand moiety
                squidly_distance = find_min_distance_per_squidly(ligand_centroid, filtered_squidly_atom_coords)

                # store distances in a dictionary
                if squidly_distance:
                    squidly_dist_dict = {res_name: match_info['distance'] for res_name, match_info in squidly_distance.items()}
                    row_result['distance_ligand_to_catalytic_residues'] = squidly_dist_dict


                # ---Distance between sequidly predicted residues and cofactor moiety
                squidly_distance = find_min_distance_per_squidly(cofactor_centroid, filtered_squidly_atom_coords)
                # store distances in a dictionary
                if squidly_distance:
                    squidly_dist_dict = {res_name: match_info['distance'] for res_name, match_info in squidly_distance.items()}
                    row_result['distance_cofactor_to_catalytic_residues'] = squidly_dist_dict


                # --- Find closest nucleophile overall
                all_nucleophiles_coords = get_all_nucs_atom_coords(pdb_file) # Get all nucleophilic residues atom coordinates
                closest_distance = find_min_distance(ligand_centroid, all_nucleophiles_coords) # Compute smallest distances between all nucleophilic residues and ligand

                if closest_distance:
                    closest_nuc_dict = {closest_distance['nuc_res']: closest_distance['distance']}
                    row_result['distance_ligand_to_closest_nuc'] = closest_nuc_dict

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
