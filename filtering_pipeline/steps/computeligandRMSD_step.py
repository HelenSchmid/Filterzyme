import pandas as pd
from pathlib import Path
import logging
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os 

from filtering_pipeline.steps.step import Step
from filtering_pipeline.utils.helpers import clean_plt

from rdkit import Chem
from rdkit.Chem import rdMolAlign
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from Bio import PDB
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
from biotite.structure.io.pdb import PDBFile
from scipy.spatial.distance import cdist  
from biotite.structure import AtomArrayStack
from openbabel import openbabel as ob
from openbabel import pybel
from io import StringIO
import tempfile

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Global plot style
plt.rcParams['svg.fonttype'] = 'none'  # Ensure text is saved as text
plt.rcParams['figure.figsize'] = (3,3)
sns.set(rc={'figure.figsize': (3,3), 'font.family': 'sans-serif', 'font.sans-serif': 'DejaVu Sans', 'font.size': 12}, 
        style='ticks')


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

def visualize_rmsd_by_entry(rmsd_df, output_dir="ligandRMSD_heatmaps"):
    '''
    Visualizes RMSD values as heatmaps for each entry in the resulting dataframe.
    '''   
    os.makedirs(output_dir, exist_ok=True)

    for entry, group in rmsd_df.groupby('Entry'):
        # Get all docked structures for the entry
        docked_proteins = list(set(group['docked_structure1']) | set(group['docked_structure2']))
        docked_proteins = sorted(docked_proteins, key=lambda x: (0 if "chai" in x.lower() else 1, x))
    
        rmsd_matrix = pd.DataFrame(np.nan, index=docked_proteins, columns=docked_proteins)

        for _, row in group.iterrows():
            l1, l2, rmsd = row['docked_structure1'], row['docked_structure2'], row['ligand_rmsd']
            rmsd_matrix.loc[l1, l2] = rmsd
            rmsd_matrix.loc[l2, l1] = rmsd

        plt.figure(figsize=(6, 5))
        ax = sns.heatmap(rmsd_matrix,annot=False, cmap='viridis', square=True, cbar=True)
        ax = clean_plt(ax)
        ax.set_title(f"Ligand RMSD Heatmap: {entry}", fontsize=14)
        ax.set_xlabel("Docked Structures")
        ax.set_ylabel("Docked Structures")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

        plt.tight_layout()
        filename = f"{entry.replace('/', '_')}_heatmap.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close() 

def get_tool_from_structure_name(structure_name: str) -> str:
    """
    Extracts the docking tool name from a structure string (e.g., 'Q97WW0_1_vina' -> 'vina').
    Assumes the tool is the last segment after the last underscore.
    """
    if '_' in structure_name:
        return structure_name.split('_')[-1]
    return "UNKNOWN_tool" # Fallback if format doesn't match

def select_best_docked_structures(rmsd_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each Entry, selects the best docked structure for each tool. Defined as structure with the lowest 
    average RMSD to all structures from same docking tool. 
    """
    best_structures = []

    # Filter the DataFrame to only include intra-tool RMSD comparisons
    intra_tool_rmsd_df = rmsd_df[rmsd_df['tool1'] == rmsd_df['tool2']].copy()
    intra_tool_rmsd_df['tool'] = intra_tool_rmsd_df['tool1']

    # Group by both 'Entry' and 'tool' on the filtered df
    for (entry, tool), group in intra_tool_rmsd_df.groupby(["Entry", "tool"]):
        # Extract unique structure names within this specific Entry-tool group
        structures = list(set(group['docked_structure1']).union(group['docked_structure2']))
        structures.sort() # Ensure consistent order for matrix indexing

        rmsd_matrix = pd.DataFrame(np.nan, index=structures, columns=structures)

        # Populate the RMSD matrix
        for _, row in group.iterrows():
            s1, s2, r = row['docked_structure1'], row['docked_structure2'], row['ligand_rmsd']
            rmsd_matrix.loc[s1, s2] = r
            rmsd_matrix.loc[s2, s1] = r # Fill symmetric entry

        # Fill diagonal with 0 (RMSD of a structure to itself is 0)
        np.fill_diagonal(rmsd_matrix.values, 0)
        
        # Calculate mean RMSD 
        avg_rmsd = rmsd_matrix.mean(axis=1)
        
        # Drop rows (structures) from avg_rmsd that are all NaNs (meaning no valid comparisons)
        avg_rmsd = avg_rmsd.dropna()

        if avg_rmsd.empty:
            logger.warning(f"No valid RMSD averages for Entry: {entry}, tool: {tool}. Skipping.")
            continue # Skip if no valid averages could be calculated

        # Select the structure with the lowest average RMSD
        best_structure_name = avg_rmsd.idxmin()

        squidly_residues = rmsd_df.loc[rmsd_df['Entry'] == entry, 'Squidly_CR_Position']

        best_structures.append({
            'Entry': entry,
            'tool': tool, 
            'best_structure': best_structure_name,
            'avg_rmsd': avg_rmsd[best_structure_name],
            'Squidly_CR_Position': squidly_residues.iloc[0] if not squidly_residues.empty else None
        })

    best_df = pd.DataFrame(best_structures)

    return best_df

class LigandRMSD(Step):
    def __init__(self, entry_col = 'Entry', input_dir: str = '', output_dir: str = '', visualize_heatmaps = False, maxMatches = 1000): 
        self.entry_col = entry_col
        self.input_dir = Path(input_dir)   
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.visualize_heatmaps = visualize_heatmaps
        self.maxMatches = maxMatches

    def __execute(self, df) -> list:

        rmsd_values = []

        # Iterate through all subdirectories in the input directory
        for sub_dir in self.input_dir.iterdir():
            print(f"Processing entry: {sub_dir.name}")

            # Process all PDB files in subdirectories
            for pdb_file_path in sub_dir.glob("*.pdb"):

                # Extract chain IDs of ligands
                chain_ids = get_hetatm_chain_ids(pdb_file_path)

                # Extract ligands as RDKit mol objects
                ligand1 = extract_chain_as_rdkit_mol(pdb_file_path, chain_id=chain_ids[0])
                ligand2 = extract_chain_as_rdkit_mol(pdb_file_path, chain_id=chain_ids[1]) 

                if ligand1 is None or ligand2 is None:
                    print(f"Could not extract both ligands, skipping {pdb_file_path}")
                    continue

                try:
                    Chem.SanitizeMol(ligand1)
                    Chem.SanitizeMol(ligand2)
                    ligand1 = Chem.RemoveHs(ligand1)
                    ligand2 = Chem.RemoveHs(ligand2)
                except Chem.rdchem.AtomValenceException as e:
                    print(f"Valence error in {pdb_file_path.name}: {e}")
                    print(Chem.MolToSmiles(ligand1))  # Just to check
                    print(Chem.MolToSmiles(ligand2))  # Just to check
                    continue  # skip this ligand pair
                except Exception as e:
                    print(f"Unexpected RDKit error in {pdb_file_path.name}: {e}")
                    continue

                # Calculate ligandRMSD
                rmsd = rdMolAlign.CalcRMS(ligand1, ligand2, maxMatches = self.maxMatches )

                # Store the RMSD value in a dictionary
                pdb_file_name = pdb_file_path.name
                structure_names = pdb_file_name.replace(".pdb", "").split("__")
                entry_name = sub_dir.name 
                
                docked_structure1_name = structure_names[0] if len(structure_names) > 0 else None
                docked_structure2_name = structure_names[1] if len(structure_names) > 1 else None

                if 'Squidly_CR_Position' in df.columns:
                    squidly_residues = df.loc[df[self.entry_col] == entry_name.strip(), 'Squidly_CR_Position']
                else:
                    squidly_residues = pd.Series(dtype=object)


                tool1_name = get_tool_from_structure_name(docked_structure1_name)
                tool2_name  = get_tool_from_structure_name(docked_structure2_name)

                rmsd_values.append({
                    'Entry': entry_name, 
                    'pdb_file': pdb_file_path.name,  # Store the name of the PDB file
                    'docked_structure1' : docked_structure1_name, 
                    'docked_structure2' : docked_structure2_name, 
                    'tool1' : tool1_name, 
                    'tool2': tool2_name,
                    'ligand_rmsd': rmsd,   # Store the calculated RMSD value
                    'Squidly_CR_Position': squidly_residues.iloc[0] if not squidly_residues.empty else None
                })

        # Convert the list of dictionaries into a df
        rmsd_df = pd.DataFrame(rmsd_values)

        # If heatmaps are to be visualized, call the visualization function
        if self.visualize_heatmaps:
            heatmap_output_dir = Path(self.output_dir) / 'ligandRMSD_heatmaps'
            os.makedirs(heatmap_output_dir, exist_ok=True)
            visualize_rmsd_by_entry(rmsd_df, output_dir=heatmap_output_dir)

        # Select the best docked structures based on RMSD
        best_docked_structure_df = select_best_docked_structures(rmsd_df)
        output_path = Path(self.output_dir) / "best_docked_structures.csv"
        best_docked_structure_df.to_csv(output_path, index=False)
        logger.info(f"Best docked structures (per tool) saved to: {output_path}")

        # Save the DataFrame as a csv file
        csv_file = Path(self.output_dir) / "ligand_rmsd.csv"
        rmsd_df.to_csv(csv_file, index=False)
        logger.info(f"Ligand RMSD results saved to: {csv_file}")
        return rmsd_df             


    def execute(self, df) -> pd.DataFrame:
        self.input_dir = Path(self.input_dir)
        return self.__execute(df)
