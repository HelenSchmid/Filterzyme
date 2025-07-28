import os
import pandas as pd
from pathlib import Path
import logging
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
import re
from Bio.PDB import PDBIO
from Bio.PDB import PDBParser, Select, PDBIO
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import MolToImage
from rdkit.Chem.Draw import rdMolDraw2D # You'll need this for MolDraw2DCairo/SVG
from rdkit.Chem.Draw.rdMolDraw2D import MolDrawOptions
from rdkit import RDLogger
import tempfile

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


def load_pdb_structure(pdb_filepath):
    """
    Loads a PDB structure from a given file path.
    """
    pdb_parser = PDBParser(QUIET=True)
    try:
        structure = pdb_parser.get_structure("prot", pdb_filepath)
        return structure
    except Exception as e:
        raise IOError(f"Error loading PDB structure from {pdb_filepath}: {e}")


def extract_ligand_from_pdb(structure, ligand_smiles, ligand_resname = 'LIG'):
    """
    Extracts a ligand (by residue name) from a Biopython structure and saves it to a temporary PDB file.
    Loads ligand from a PDB file into RDKit, then assigns bond orders from a provided SMILES string 
    template to ensure correct chemical perception.
    """
    class LigandSelect(Select):
        def accept_residue(self, residue):
            return residue.get_resname().strip() == ligand_resname

    io = PDBIO()
    io.set_structure(structure)

    temp_pdb = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb")
    io.save(temp_pdb.name, LigandSelect())

    # Load the PDB ligand into RDKit
    pdb_mol = Chem.MolFromPDBFile(temp_pdb.name, removeHs=False)
    if pdb_mol is None:
        raise ValueError("Failed to parse ligand PDB with RDKit.")

    # Create template molecule from SMILES
    template_mol = Chem.MolFromSmiles(ligand_smiles)
    if template_mol is None:
        raise ValueError(f"Could not parse SMILES for template: {ligand_smiles}")

    # Assign bond orders from the template to the PDB-derived molecule
    try:
        ligand_mol = AllChem.AssignBondOrdersFromTemplate(template_mol, pdb_mol)
    except Exception as e:
        print(f"WARNING: Error assigning bond orders from template: {e}")
        print("Proceeding with PDB-parsed molecule (may have incorrect bond orders/valency).")
        ligand_mol = pdb_mol # Fallback if assignment fails

    return ligand_mol


def find_substructure_coordinates(mol, smarts_pattern, atom_to_get_coords_idx=0):
    """
    Finds substructure matches for a given SMARTS pattern in an RDKit molecule
    and returns the 3D coordinates of a specified atom within each match.
    The atom_idx for the carbonyl C and the phosphate atom are 0. 
    """

    coords_dict = {}

    if mol.GetNumConformers() == 0:
        logger.warning("Ligand molecule has no 3D conformers. Cannot get coordinates.")
        return {}

    # Compile SMARTS pattern
    pattern = Chem.MolFromSmarts(smarts_pattern)
    if pattern is None:
        raise ValueError(f"Invalid SMARTS pattern: {smarts_pattern}")
    
    matches = mol.GetSubstructMatches(pattern)
    label = smarts_pattern
    coords_dict[label] = []

    if not matches: 
        logger.warning(f"There was no match found for the SMARTS {smarts_pattern}")
        return coords_dict 

    for match in matches: 
        if atom_to_get_coords_idx >= len(match):
            logger.warning(f"Index {atom_to_get_coords_idx} out of range in match {match}")
            continue

        atom_idx = match[atom_to_get_coords_idx]
        atom = mol.GetAtomWithIdx(atom_idx)
        conf = mol.GetConformer()
        pos = conf.GetAtomPosition(atom_idx)

        coords_dict[label].append({
            'atom': atom.GetSymbol(),
            'coords': (pos.x, pos.y, pos.z)
        })

    return coords_dict


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


class GeometricFiltering(Step):

    def __init__(self, ligand_smiles: str= '', smarts_pattern: str = '', preparedfiles_dir: str = '', esterase = 0, find_closest_nucleophile = 0, output_dir: str= ''):
        
        self.ligand_smiles = ligand_smiles
        self.smarts_pattern = smarts_pattern
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
            row_result = {}

            default_result = {
                'distance_ligand_to_squidly_residues': None,
                'distance_ligand_to_closest_nuc': None,
                'Bürgi–Dunitz_angle_to_squidly_residue': None,
                'Bürgi–Dunitz_angle_to_closest_nucleophile': None
            }

            try:
                # Load full PDB structure
                pdb_file = self.preparedfiles_dir / f"{best_structure_name}.pdb"
                print(f"Processing PDB file: {pdb_file.name}")
                protein_structure = load_pdb_structure(pdb_file)

                # Extract ligand atoms from PDB
                extracted_ligand_atoms = extract_ligand_from_pdb(protein_structure, self.ligand_smiles)

                # Find coordinates of chemical moiety of interest of the ligand
                ligand_coords = find_substructure_coordinates(extracted_ligand_atoms, self.smarts_pattern, atom_to_get_coords_idx=0) # carbonyl C and phosphate atom are both at index 0

                # Get squidly protein atom coordinates
                squidly_atom_coords = get_squidly_residue_atom_coords(pdb_file, squidly_residues)
                filtered_squidly_atom_coords = filter_residue_atoms(squidly_atom_coords, atom_selection)

                # Compute distances between squidly predicted residues and target chemical moiety
                if not ligand_coords or not isinstance(ligand_coords, dict):
                    logger.warning(f"No coordinates found that match the SMARTS pattern in {entry_name}.")
                    row_result.update(default_result)
                    results.append(row_result)
                    continue

                if not squidly_atom_coords:
                    logger.warning(f"No squidly residues found in {entry_name}.")
                    row_result.update(default_result)
                    results.append(row_result)
                    continue

                # --- Find distance between squidly residues and ligand
                squidly_distance = find_min_distance_per_squidly(ligand_coords, filtered_squidly_atom_coords)

                # store distances in a dictionary
                if squidly_distance:
                    squidly_dist_dict = {res_name: match_info['distance'] for res_name, match_info in squidly_distance.items()}
                    row_result['distance_ligand_to_squidly_residues'] = squidly_dist_dict

                # --- Find closest nucleophile overall
                if self.find_closest_nuc == 1: 
                    all_nucleophiles_coords = get_all_nucs_atom_coords(pdb_file) # Get all nucleophilic residues atom coordinates
                    closest_distance = find_min_distance(ligand_coords, all_nucleophiles_coords) # Compute smallest distances between all nucleophilic residues and ligand

                    if closest_distance:
                        closest_nuc_dict = {closest_distance['nuc_res']: closest_distance['distance']}
                        row_result['distance_ligand_to_closest_nuc'] = closest_nuc_dict

                
                # --- Calculate Bürgi–Dunitz angle between closest nucleophile and ester bond
                if self.esterase == 1: 
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




