"""Calculate distances between ligand and protein termini from Chai CIF output files."""

import gemmi
import numpy as np
from scipy.spatial.distance import euclidean
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import pandas as pd


def load_cif_structure(cif_path: str) -> gemmi.Structure:
    """
    Load a CIF structure file (e.g., from Chai).
    
    Parameters
    ----------
    cif_path : str
        Path to the CIF file
        
    Returns
    -------
    gemmi.Structure
        Parsed structure object
        
    Raises
    ------
    FileNotFoundError
        If the CIF file does not exist
    """
    path = Path(cif_path)
    if not path.exists():
        raise FileNotFoundError(f"CIF file not found: {cif_path}")
    
    doc = gemmi.cif.read(str(cif_path))

    structure = gemmi.make_structure_from_block(doc[0])
    return structure


def extract_ligand_atoms(structure: gemmi.Structure, ligand_name: str = "LIG2") -> np.ndarray:
    """
    Extract ligand atom coordinates from structure.
    
    Parameters
    ----------
    structure : gemmi.Structure
        Parsed structure
    ligand_name : str
        Residue name of the ligand (default: "LIG")
        
    Returns
    -------
    np.ndarray
        Shape (n_atoms, 3) array of ligand atom coordinates
        
    Raises
    ------
    ValueError
        If ligand is not found in structure
    """
    ligand_coords = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.name == ligand_name:
                    for atom in residue:
                        pos = atom.pos
                        ligand_coords.append([pos.x, pos.y, pos.z])
    
    if not ligand_coords:
        raise ValueError(f"Ligand '{ligand_name}' not found in structure")
    
    return np.array(ligand_coords)


def is_protein(residue) -> bool:
    """
    Check if a residue is part of the protein (standard amino acid).
    
    Parameters
    ----------
    residue : gemmi.Residue
        Residue object to check
        
    Returns
    -------
    bool
        True if residue is a standard amino acid, False otherwise
    """
    standard_amino_acids = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
        'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
        'THR', 'TRP', 'TYR', 'VAL'
    }
    return residue.name in standard_amino_acids


def extract_protein_termini(structure: gemmi.Structure):
    """
    Extract N- and C-terminus CA atom coordinates from the longest protein chain.
    """

    def get_ca_atom(residue):
        """
        Get the CA atom of a residue
        """
        # Collect all CA atoms 
        ca_atoms = [atom for atom in residue if atom.name == "CA"]
        if not ca_atoms:
            return None
        # Pick highest occupancy CA atom
        return max(ca_atoms, key=lambda a: a.occ)
    

    chains = []

    for model in structure:
        for chain in model:
            residues = [r for r in chain if is_protein(r)]
            if len(residues) > 1:
                chains.append(residues)

    if not chains:
        raise ValueError("No protein chains found.")

    # choose longest chain
    residues = max(chains, key=len)

    n_term_atom = get_ca_atom(residues[0])
    c_term_atom = get_ca_atom(residues[-1])

    if n_term_atom is None or c_term_atom is None:
        raise ValueError("Could not find CA atoms at protein termini")

    n_term = np.array([n_term_atom.pos.x, n_term_atom.pos.y, n_term_atom.pos.z])
    c_term = np.array([c_term_atom.pos.x, c_term_atom.pos.y, c_term_atom.pos.z])

    return n_term, c_term


def calculate_ligand_terminus_distances(
    cif_path: str,
    ligand_name: str = "LIG2") -> Dict[str, float]:
    """
    Calculate distances from ligand *centroid* to N- and C-terminus CA atoms.
    Parameters
    ----------
    cif_path : str
        Path to the CIF file
    ligand_name : str, optional
        Residue name of the ligand (default is "LIG2")
    Returns
    -------
    Dict[str, float]
        Dictionary with distances and coordinates
    Raises
    """

    structure = load_cif_structure(cif_path)
    ligand_coords = extract_ligand_atoms(structure, ligand_name=ligand_name)
    n_term, c_term = extract_protein_termini(structure)

    # --- NEW: ligand centroid ---
    ligand_centroid = np.mean(ligand_coords, axis=0)

    # --- NEW: distance from centroid to termini ---
    dist_n = float(np.linalg.norm(ligand_centroid - n_term))
    dist_c = float(np.linalg.norm(ligand_centroid - c_term))

    # return {
    #     'cif_path': str(cif_path),
    #     'n_terminus_distance': dist_n,
    #     'c_terminus_distance': dist_c,
    #     'ligand_centroid': ligand_centroid.tolist(),
    #     'n_terminus_coord': n_term.tolist(),
    #     'c_terminus_coord': c_term.tolist(),
    # }

    return dist_n, dist_c, ligand_centroid, n_term, c_term


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python chai_ligand_distance.py <path_to_cif>")
        sys.exit(1)
    
    cif_path = sys.argv[1]
    result = calculate_ligand_terminus_distances(cif_path)
    
    print(f"\nResults for {cif_path}:")
    print(f"  N-terminus distance: {result['n_terminus_distance']:.2f} Å")
    print(f"  C-terminus distance: {result['c_terminus_distance']:.2f} Å")
