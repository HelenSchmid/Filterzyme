import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import re
from tempfile import TemporaryDirectory
from multiprocessing.dummy import Pool as ThreadPool

from .step import Step

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def clean_vina_ligand_file(input_path, output_path=None):
    """
    Cleans a Vina ligand PDBQT file by:
    - Removing the first line and the line immediately following any 'ENDMDL'.
    - Relabeling ATOM records to HETATM.
    - Appending a number to the atom names (e.g., C1, C2, O1, O2) per atom type.
    - Saves the cleaned content to a new file.
    
    Parameters:
    input_path (str or Path): The path to the original file.
    output_path (str or Path, optional): Where to save the cleaned file. 
                                         If not provided, '_cleaned' is appended to the filename.
    
    Returns:
    Path: Path to the cleaned output file.
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_cleaned{input_path.suffix}")

    with open(input_path, 'r') as file:
        lines = file.readlines()

    filtered_lines = []
    skip_next_line = False
    atom_counter = {'C': 1, 'O': 1, 'P': 1, 'N': 1}  # Separate counters for each atom type

    for i, line in enumerate(lines[1:], start=1):  # Skip the first line
        if skip_next_line:
            skip_next_line = False
            continue
        if line.startswith('ENDMDL'):
            filtered_lines.append(line)
            skip_next_line = True
        else:
            if line.startswith(('ATOM', 'HETATM')):
                parts = list(line.rstrip('\n'))
                atom_name = ''.join(parts[12:16]).strip()
                residue_name = ''.join(parts[17:20]).strip()
                element = line[76:78].strip()

                # Fallback if element field is blank
                if not element:
                    element = atom_name[0]

                # Atom counters per true element
                if element in atom_counter:
                    counter = atom_counter[element]
                    atom_counter[element] += 1
                else:
                    counter = 1
                    atom_counter[element] = 2

                # Change ATOM to HETATM
                parts[12:16] = f'{element}{counter}'.ljust(4)
                parts[0:6] = list('HETATM')
                parts[21] = 'B'

                # Reassemble the line with the new atom name
                new_line = ''.join(parts)
                filtered_lines.append(new_line + '\n')
            else:
                filtered_lines.append(line)

    with open(output_path, 'w') as file:
        file.writelines(filtered_lines)

    return output_path


def split_ligands_and_combine(protein_path, ligands_path, entry_name, output_dir, renumber_atoms=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = []
    # Load protein atoms
    protein_lines = Path(protein_path).read_text().splitlines()
    protein_atoms = [line for line in protein_lines if line.startswith(('ATOM', 'HETATM'))]

    # Load and split ligand blocks
    ligand_blocks = []
    current_block = []

    for line in Path(ligands_path).read_text().splitlines():
        if line.startswith(('ATOM', 'HETATM')):
            current_block.append(line)
        elif line.strip() == 'ENDMDL' and current_block:
            ligand_blocks.append(current_block)
            current_block = []

    # Catch any ligand block without a trailing END
    if current_block:
        ligand_blocks.append(current_block)

    # Combine and write each protein + ligand combo
    for i, ligand_atoms in enumerate(ligand_blocks, start=1):

        combined_atoms = []

        # Add protein atoms
        combined_atoms.extend(protein_atoms)
        if protein_atoms:
            last = protein_atoms[-1].ljust(80)
            ter_line = (
                f"TER   {len(protein_atoms)+1:5d}      "
                f"{last[17:20]} {last[21]}{last[22:26]}"
            )
            combined_atoms.append(ter_line)

        # Add ligand atoms
        combined_atoms.extend(ligand_atoms)
        if ligand_atoms:
            last = ligand_atoms[-1].ljust(80)
            ter_line = (
                f"TER   {len(combined_atoms)+1:5d}      "
                f"{last[17:20]} {last[21]}{last[22:26]}"
            )
            combined_atoms.append(ter_line)


        if renumber_atoms:
            serial = 1
            new_combined = []
            for line in combined_atoms:
                line = line.ljust(80)  # Ensure line is long enough
                new_line = f"{line[:6]}{serial:5d}{line[11:]}"
                new_combined.append(new_line)
                serial += 1
            combined_atoms = new_combined
        
        # Save the combined structure to a PDB file
        output_path = output_dir / f"{entry_name}_{i}_vina.pdb"
        Path(output_path).write_text('\n'.join(combined_atoms) + '\nEND\n')
        output_paths.append(str(output_path))

    return output_paths


class PrepareVina(Step):
    def __init__(self, vina_dir = None, ligand_name: str = '',  output_dir: str = '' , num_threads=1):
        self.vina_dir = vina_dir
        self.output_dir = Path(output_dir)
        self.ligand_name = ligand_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_threads = num_threads or 1
        

    def __execute(self, df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
        results = []

        for vina_path in df[self.vina_dir].apply(str):

            vina_path = Path(vina_path)
            if not vina_path.exists():
                logger.warning(f"Vina path not found: {vina_path}")
                results.append(None)
                continue

            entry_name = vina_path.stem
            ligand_file = vina_path.parent / f'{entry_name}-{self.ligand_name}.pdb'

            try:
                # Read and clean ligand
                cleaned_ligand_file = clean_vina_ligand_file(ligand_file)

                # Combine protein and ligands in same file
                output_files = split_ligands_and_combine(
                    protein_path=vina_path,
                    ligands_path=cleaned_ligand_file,
                    entry_name=entry_name,
                    output_dir=self.output_dir, 
                    renumber_atoms=True
                )
                results.append(output_files)

            except Exception as e:
                logger.error(f"Error processing {vina_path}: {e}")
                results.append(None)

        return results


    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.output_dir:
            if self.num_threads > 1:
                output_filenames = []
                df_list = np.array_split(df, self.num_threads)
                for df_chunk in df_list:
                    output_filenames += self.__execute(df_chunk, self.output_dir)
                    
                df['vina_files_for_superimposition'] = output_filenames
                return df
            
            else:
                output_filenames = self.__execute(df, self.output_dir)
                df['vina_files_for_superimposition'] = output_filenames
                return df
        else:
            print('No output directory provided')