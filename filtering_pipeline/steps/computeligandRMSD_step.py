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
from collections import defaultdict

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




'''
1. Weighted Inter-Tool Average RMSD (already implemented)
- For each pose, compute the average RMSD to all poses from each other tool
- Weight each tool by the number of poses it contributes
- Select the pose with the lowest weighted average RMSD

2. Closest Pose Per Other Tool (Tool Min-RMSD)
- For each pose, find the single most similar pose (lowest RMSD) from each other tool
- Take the average of these lowest RMSDs
- Select the pose with the lowest average across other tools
'''


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
            'avg_ligandRMSD': avg_rmsd[best_structure_name],
            'Squidly_CR_Position': squidly_residues.iloc[0] if not squidly_residues.empty else None
        })

    best_df = pd.DataFrame(best_structures)

    return best_df


def compute_normalized_ligand_rmsd_stats(rmsd_df: pd.DataFrame):
    """
    Computes per-entry normalized ligand RMSD statistics:
    - Mean and std of within-tool RMSDs (e.g. vina-vina, dchai-chai)
    - Mean and std of between-tool RMSDs (e.g. vina-chai, vina-boltz)
    - Overall mean and std RMSD for each entry
    Adds these as new columns to rmsd_df.
    """
    from itertools import combinations_with_replacement

    # Ensure lowercase and clean tool names
    rmsd_df["tool1"] = rmsd_df["tool1"].str.strip().str.lower()
    rmsd_df["tool2"] = rmsd_df["tool2"].str.strip().str.lower()

    enriched_df = rmsd_df.copy()

    # Collect per-entry statistics
    entry_stats = []

    for entry, group in rmsd_df.groupby("Entry"):
        stats = {"Entry": entry}
        overall_rmsds = []

        tools = sorted(set(group["tool1"]).union(group["tool2"]))

        for t1, t2 in combinations_with_replacement(tools, 2):
            pair_label = f"{min(t1, t2)}-{max(t1, t2)}"
            mask = group.apply(
                lambda row: set([row["tool1"], row["tool2"]]) == set([t1, t2]),
                axis=1
            )
            subset = group[mask]
            rmsds = subset["ligand_rmsd"].dropna().tolist()
            overall_rmsds.extend(rmsds)

            if t1 == t2:
                stats[f"{t1}_{t1}_mean_ligandRMSD"] = np.mean(rmsds) if rmsds else np.nan
                stats[f"{t1}_{t1}_std_ligandRMSD"] = np.std(rmsds) if rmsds else np.nan
            else:
                stats[f"{t1}_{t2}_mean_ligandRMSD"] = np.mean(rmsds) if rmsds else np.nan
                stats[f"{t1}_{t2}_std_ligandRMSD"] = np.std(rmsds) if rmsds else np.nan

        stats["overall_ligandRMSD_mean"] = np.mean(overall_rmsds) if overall_rmsds else np.nan
        stats["overall_ligandRMSD_std"] = np.std(overall_rmsds) if overall_rmsds else np.nan

        entry_stats.append(stats)

    stats_df = pd.DataFrame(entry_stats)
    enriched_df = enriched_df.merge(stats_df, on="Entry", how="left")

    return enriched_df



def select_best_docked_structures(rmsd_df: pd.DataFrame) -> pd.DataFrame:
    """
    Selects the best overall docked structure per Entry using two methods:
    
    1. inter_tool_weighted_avg: Weighted average RMSD to all structures from other tools,
       weighted by number of structures each tool contributes.
       
    2. inter_tool_min_per_tool: Average of *minimum* RMSDs to each other tool
       (one closest structure per tool).

    3. vina_avg_intra_tool: Best structure among all vina generated structures based on
       lowest average RMSD to other Vina poses.
       
    Returns a DataFrame with one row per method per Entry.
    """
    best_structures = []

    for entry, entry_df in rmsd_df.groupby("Entry"):
        # Build structure -> tool mapping
        structure_to_tool = {}
        all_structures = pd.unique(entry_df[['docked_structure1', 'docked_structure2']].values.ravel('K'))

        for _, row in entry_df.iterrows():
            structure_to_tool[row['docked_structure1']] = row['tool1']
            structure_to_tool[row['docked_structure2']] = row['tool2']

        tools = set(structure_to_tool.values())

        # Build RMSD matrix
        rmsd_matrix = pd.DataFrame(np.nan, index=all_structures, columns=all_structures)
        for _, row in entry_df.iterrows():
            s1, s2 = row['docked_structure1'], row['docked_structure2']
            rmsd_matrix.loc[s1, s2] = row['ligand_rmsd']
            rmsd_matrix.loc[s2, s1] = row['ligand_rmsd']
        np.fill_diagonal(rmsd_matrix.values, 0)

        # Group structures by tool
        tool_to_structures = defaultdict(list)
        for s, t in structure_to_tool.items():
            tool_to_structures[t].append(s)

        squidly_pos = entry_df['Squidly_CR_Position'].iloc[0] if 'Squidly_CR_Position' in entry_df.columns else None

        # ----- Method 1: Weighted average RMSD to all poses per other tool -----
        weighted_rmsd_scores = {}
        for s in rmsd_matrix.index:
            tool_s = structure_to_tool[s]
            weighted_sum = 0.0
            total_weight = 0

            for other_tool, other_structures in tool_to_structures.items():
                if other_tool == tool_s:
                    continue

                values = rmsd_matrix.loc[s, other_structures].dropna()
                if values.empty:
                    continue

                weight = len(other_structures)
                avg_rmsd = values.mean()

                weighted_sum += weight * avg_rmsd
                total_weight += weight

            if total_weight > 0:
                weighted_rmsd_scores[s] = weighted_sum / total_weight

        if weighted_rmsd_scores:
            best_s = min(weighted_rmsd_scores, key=weighted_rmsd_scores.get)
            best_structures.append({
                'Entry': entry,
                'tool': structure_to_tool[best_s],
                'best_structure': best_s,
                'avg_ligandRMSD': weighted_rmsd_scores[best_s],
                'method': 'inter_tool_weighted_avg',
                'Squidly_CR_Position': squidly_pos
            })

        # ----- Method 2: Average of closest pose per tool -----
        closest_rmsd_scores = {}
        for s in rmsd_matrix.index:
            tool_s = structure_to_tool[s]
            closest_sum = 0.0
            tool_count = 0

            for other_tool, other_structures in tool_to_structures.items():
                if other_tool == tool_s:
                    continue

                values = rmsd_matrix.loc[s, other_structures].dropna()
                if values.empty:
                    continue

                closest_sum += values.min()
                tool_count += 1

            if tool_count > 0:
                closest_rmsd_scores[s] = closest_sum / tool_count

        if closest_rmsd_scores:
            best_s = min(closest_rmsd_scores, key=closest_rmsd_scores.get)
            best_structures.append({
                'Entry': entry,
                'tool': structure_to_tool[best_s],
                'best_structure': best_s,
                'avg_ligandRMSD': closest_rmsd_scores[best_s],
                'method': 'inter_tool_min_per_tool',
                'Squidly_CR_Position': squidly_pos
            })

        # ----- Method 3: Intra-tool average RMSD within Vina structures -----
        vina_structures = tool_to_structures.get('vina', [])
        if len(vina_structures) >= 2:
            vina_matrix = rmsd_matrix.loc[vina_structures, vina_structures]
            avg_rmsd_vina = vina_matrix.mean(axis=1).dropna()

            if not avg_rmsd_vina.empty:
                best_vina = avg_rmsd_vina.idxmin()
                best_structures.append({
                    'Entry': entry,
                    'tool': 'vina',
                    'best_structure': best_vina,
                    'avg_ligandRMSD': avg_rmsd_vina[best_vina],
                    'method': 'vina_avg_intra_tool',
                    'Squidly_CR_Position': squidly_pos
                })

    return pd.DataFrame(best_structures)

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

                    if ligand1.GetNumConformers() == 0:
                        AllChem.EmbedMolecule(ligand1)

                    if ligand2.GetNumConformers() == 0:
                        AllChem.EmbedMolecule(ligand2)

                except Chem.rdchem.AtomValenceException as e:
                    print(f"Valence error in {pdb_file_path.name}: {e}")
                    print(Chem.MolToSmiles(ligand1))  # Just to check
                    print(Chem.MolToSmiles(ligand2))  # Just to check
                    continue  # skip this ligand pair
                except Exception as e:
                    print(f"Unexpected RDKit error in {pdb_file_path.name}: {e}")
                    continue

                # Calculate ligandRMSD
                try:
                    rmsd = rdMolAlign.CalcRMS(ligand1, ligand2, maxMatches=self.maxMatches)
                except RuntimeError as e:
                    print(f"LigandRMSD calculation failed for {pdb_file_path.name}: {e}")
                    continue 

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

        # Add tool-wise and overall normalized stats per entry
        rmsd_df = compute_normalized_ligand_rmsd_stats(rmsd_df)

        return rmsd_df             


    def execute(self, df) -> pd.DataFrame:
        self.input_dir = Path(self.input_dir)
        return self.__execute(df)
