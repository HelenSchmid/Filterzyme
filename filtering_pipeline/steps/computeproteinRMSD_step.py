from filtering_pipeline.steps.step import Step

import pandas as pd
from pathlib import Path
import logging
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 

import biotite.structure as struc
import biotite.structure.io.pdb as pdb
from biotite.structure.io.pdb import PDBFile
from scipy.spatial.distance import cdist  
from openbabel import openbabel as ob
from openbabel import pybel
import tempfile
from Bio import PDB

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Global plot style
plt.rcParams['svg.fonttype'] = 'none'  # Ensure text is saved as text
sns.set(rc={'figure.figsize': (3,3), 'font.family': 'sans-serif', 'font.sans-serif': 'DejaVu Sans', 'font.size': 12}, 
        style='ticks')

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


def compute_proteinrmsd(pdb_file): 
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('complex', pdb_file)

    # Extract chains A and B 
    chain_A = structure[0]['A']  # Chain A from model 0
    chain_B = structure[0]['B']  

    # Get C-alpha atoms from both chains
    atoms_A = [atom for atom in chain_A.get_atoms() if atom.get_name() == 'CA']
    atoms_B = [atom for atom in chain_B.get_atoms() if atom.get_name() == 'CA']

    # Ensure that the number of C-alpha atoms in both chains match
    if len(atoms_A) != len(atoms_B):
        print("The number of C-alpha atoms in the two chains do not match!")
    else:
        # Compute RMSD
        super_imposer = PDB.Superimposer()
        super_imposer.set_atoms(atoms_A, atoms_B)  # Superimpose the chains based on C-alpha atoms
        super_imposer.apply(chain_B.get_atoms())  # Apply the transformation to chain B
        rmsd = super_imposer.rms
        return rmsd

def get_tool_from_structure_name(structure_name: str) -> str:
    """
    Extracts the docking tool name from a structure string (e.g., 'Q97WW0_1_vina' -> 'vina').
    Assumes the tool is the last segment after the last underscore.
    """
    if '_' in structure_name:
        return structure_name.split('_')[-1]
    return "UNKNOWN_tool" # Fallback if format doesn't match

def visualize_rmsd_by_entry(rmsd_df, output_dir="proteinRMSD_heatmaps"):
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
            l1, l2, rmsd = row['docked_structure1'], row['docked_structure2'], row['protein_rmsd']
            rmsd_matrix.loc[l1, l2] = rmsd
            rmsd_matrix.loc[l2, l1] = rmsd

        plt.figure(figsize=(6, 5))
        sns.heatmap(rmsd_matrix, annot=False, cmap="viridis", square=True, cbar=True)
        plt.title(f"Heatmap of protein RMSD: {entry}", fontsize=14)
        plt.xlabel("Docked Structures")
        plt.ylabel("Docked Structures")
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()

        filename = f"{entry.replace('/', '_')}_heatmap.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()


class ProteinRMSD(Step):
    def __init__(self, entry_col = 'Entry', input_dir: str = '', output_dir: str = '', visualize_heatmaps = False,  num_threads=1): 
        self.entry_col = entry_col
        self.input_dir = Path(input_dir)   
        self.output_dir = Path(output_dir)
        self.visualize_heatmaps = visualize_heatmaps
        self.num_threads = num_threads or 1

    def __execute(self, df) -> list:

        rmsd_values = []

        # Iterate through all subdirectories in the input directory
        for sub_dir in self.input_dir.iterdir():
            print(f"Processing entry: {sub_dir.name}")

            # Process all PDB files in subdirectories
            for pdb_file_path in sub_dir.glob("*.pdb"):

                rmsd = compute_proteinrmsd(pdb_file_path)  # Compute protein RMSD for the PDB file

                # Store the RMSD value in a dictionary to append later
                pdb_file_name = pdb_file_path.name
                structure_names = pdb_file_name.replace(".pdb", "").split("__")
                
                docked_structure1_name = structure_names[0] if len(structure_names) > 0 else None
                docked_structure2_name = structure_names[1] if len(structure_names) > 1 else None

                entry_name = docked_structure1_name.split('_')[0]

                mask = df[self.entry_col].str.strip() == entry_name.strip()
                if 'Squidly_CR_Position' in df.columns and mask.any():
                    squidly_residues = df.loc[mask, 'Squidly_CR_Position'].iat[0]
                else:
                    squidly_residues = ""

                tool1_name = get_tool_from_structure_name(docked_structure1_name)
                tool2_name  = get_tool_from_structure_name(docked_structure2_name)

                rmsd_values.append({
                    'Entry': entry_name, 
                    'pdb_file': pdb_file_path.name,  # Store the name of the PDB file
                    'docked_structure1' : docked_structure1_name, 
                    'docked_structure2' : docked_structure2_name, 
                    'tool1' : tool1_name, 
                    'tool2': tool2_name,
                    'Squidly_CR_Position': squidly_residues,
                    'protein_rmsd': rmsd,   # Store the calculated RMSD value
                })
        
        # Convert the list of dictionaries into a df
        rmsd_df = pd.DataFrame(rmsd_values)

        # If heatmaps are to be visualized, call the visualization function
        if self.visualize_heatmaps:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            visualize_rmsd_by_entry(rmsd_df, output_dir=self.output_dir)

        return rmsd_df


    def execute(self, df) -> pd.DataFrame:
        self.input_dir = Path(self.input_dir)
        return self.__execute(df)
