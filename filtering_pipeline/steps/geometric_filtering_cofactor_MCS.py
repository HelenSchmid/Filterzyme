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
from filtering_pipeline.utils.helpers import get_hetatm_chain_ids, norm_l1_dist,atom_composition_fingerprint
from filtering_pipeline.utils.helpers import closest_ligands_by_element_composition, atom_composition_fingerprint, extract_chain_as_rdkit_mol
from filtering_pipeline.utils.helpers import as_mol, ensure_3d

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

def tolerant_query_from_smiles(moiety_smiles: str):
    """
    Build a relaxed query smiles (from ligand or cofactor moiety column) to survive kekulization/aromatic/protonation quirks.
    """
    q = Chem.MolFromSmiles(moiety_smiles)
    if q is None:
        return None
    p = Chem.AdjustQueryParameters()
    # Relax common failure points:
    p.makeBondsGeneric = True
    p.matchAromaticToAliphatic = True
    p.aromatizeIfPossible = False
    p.adjustDegree = True
    p.adjustHeavyDegree = True
    p.adjustRingCount = True
    p.adjustRingConnectivity = True
    p.maxImplicitHs = 8
    return Chem.AdjustQueryMol(q, p)

def find_substructure_matches(
    mol: Chem.Mol,
    sub: str,
    is_smarts: bool = False,
    use_chirality: bool = False,
    try_tolerant_on_fail: bool = True
):
    """
    Find substructure atom-index matches. Can use smiles or smarts string as input. 
    First try strict query and if enable try tolerant query. 
    """
    # 1) strict query
    q = Chem.MolFromSmarts(sub) if is_smarts else Chem.MolFromSmiles(sub)
    if q is None:
        raise ValueError("Could not parse substructure pattern.")

    matches = list(mol.GetSubstructMatches(q, useChirality=use_chirality, uniquify=True))
    if matches or is_smarts or not try_tolerant_on_fail:
        return matches

    # 2) tolerant retry (SMILES-only path)
    tq = tolerant_query_from_smiles(sub)
    if tq is None:
        return matches  # keep as empty if tolerant build failed
    logger.warning(f'Strict ligand-substructure matching unsucessfuly, used tolerant query.')
    return list(mol.GetSubstructMatches(tq, useChirality=False, uniquify=True))

def mcs_match_indices(ligand: Chem.Mol, moiety_smiles: str, timeout=5):
    """
    Use MCS between the ligand and the standalone moiety.
    Returns list of tuples of atom indices in the ligand.
    """
    q = Chem.MolFromSmiles(moiety_smiles)
    if q is None:
        return []
    res = rdFMCS.FindMCS(
        [ligand, q],
        completeRingsOnly=False,
        ringMatchesRingOnly=False,
        bondCompare=rdFMCS.BondCompare.CompareAny,
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        matchValences=False,
        timeout=timeout,
    )
    if not res or not res.smartsString:
        return []
    mcs = Chem.MolFromSmarts(res.smartsString)
    if mcs is None:
        return []
    return list(ligand.GetSubstructMatches(mcs, uniquify=True))

def moiety_centroid_with_fallbacks(
    mol: Chem.Mol,
    moiety_smiles: str,
    grow_mcs_by_one_bond: bool = True,
    use_chirality: bool = False
):
    """
    1) strict substructure → 2) tolerant substructure → 3) MCS → 4) whole-ligand.
    Returns (centroids_list, method_label, used_indices_list).
    """
    # 1–2) strict then tolerant (your updated function handles both)
    try:
        matches = find_substructure_matches(
            mol, moiety_smiles, is_smarts=False, use_chirality=use_chirality, try_tolerant_on_fail=True
        )
    except Exception:
        matches = []
    if matches:
        return centroids_from_matches(mol, matches), "substructure_or_tolerant", list(matches[0])

    # 3) use MSC
    mcs_matches = mcs_match_indices(mol, moiety_smiles, timeout=5)
    if mcs_matches:
        core = set(mcs_matches[0])
        if grow_mcs_by_one_bond:
            for idx in list(core):
                a = mol.GetAtomWithIdx(idx)
                for nb in a.GetNeighbors():
                    core.add(nb.GetIdx())
        cent = centroid_from_indices(mol, list(core))
        logger.warning(f"Ligand-substructure matching using MSC")
        return [cent], "mcs" if not grow_mcs_by_one_bond else "mcs", list(core)

    # 4) fallback: whole-ligand centroid
    all_idx = tuple(range(mol.GetNumAtoms()))
    logger.warning(f"Ligand-substructure centroid calculation unsuccessfull. Use whole-ligand centroid instead.")
    return centroids_from_matches(mol, all_idx), "whole_ligand", list(all_idx)

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
                'ligand_moiety_method': None, 
                'cofactor_moiety_method': None
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

                # Find ligand substructure match with moiety of interest and calculate ligand-centroid
                ligand_centroid, lig_method, lig_used = moiety_centroid_with_fallbacks(
                    ligand_mol, substrate_moiety, grow_mcs_by_one_bond=True,
                    use_chirality=False)
                
                row_result["ligand_moiety_method"] = lig_method


                # --- Distance between ligand and cofactor ---

                # Get cofactor mol object and assign correct bond order based on smiles
                if 'cofactor_smiles' in df.columns and cofactor_smiles is not None and tool != 'vina':              
                    cofactor_candidate = closest_ligands_by_element_composition(ligands, cofactor_smiles, top_k=1)
                    cofactor_mol = as_mol(cofactor_candidate[0]) if cofactor_candidate else None
                    cofactor_mol = as_mol(cofactor_candidate[0]) if cofactor_candidate else None
                    cofactor_mol = assign_bond_orders_from_smiles(cofactor_mol, cofactor_smiles)
                    cofactor_mol = ensure_3d(cofactor_mol)

                    # Find cofactor substructure match with moiety of interest and cofactor centroid
                    cofactor_centroid, cof_method, cof_used = moiety_centroid_with_fallbacks(
                        cofactor_mol,
                        cofactor_moiety,
                        grow_mcs_by_one_bond=True,
                        use_chirality=False
                    )
                    row_result["cofactor_moiety_method"] = cof_method

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
