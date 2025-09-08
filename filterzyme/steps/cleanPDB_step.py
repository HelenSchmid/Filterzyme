import os
import shutil
import pandas as pd
import numpy as np
from steps.step import Step
from pathlib import Path
import subprocess
import logging
import tempfile
from collections import defaultdict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

STANDARD_RESIDUES = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
    'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
    'THR', 'TRP', 'TYR', 'VAL', 'DA', 'DC', 'DG', 'DT', 'A', 'C', 'G', 'U'
}

class CleanPDB(Step):
    def __init__(self, input_col: str, output_dir: str, num_threads: int = 1):
        self.input_col = input_col
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_threads = num_threads or 1

    def __execute(self, df: pd.DataFrame) -> list:
        output_paths = []

        for cif_dir in df[self.input_col]:
            cif_dir = Path(cif_dir)

            if not cif_dir.exists():
                logger.warning(f"Directory not found: {cif_dir}")
                output_paths.append(None)
                continue

            with tempfile.TemporaryDirectory() as pdb_temp_dir:
                pdb_temp_dir = Path(pdb_temp_dir)

                cif_files = list(cif_dir.rglob("*.cif"))

                final_dir = self.output_dir / cif_dir.name
                final_dir.mkdir(parents=True, exist_ok=True)

                for cif_file in cif_files:
                    pdb_file = self.convert_cif_to_pdb(cif_file, pdb_temp_dir)
                    cleaned_file = self.clean_pdb(pdb_file)
                    shutil.copy(cleaned_file, final_dir)

                output_paths.append(str(final_dir))

        return output_paths

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.num_threads > 1:
            df_chunks = np.array_split(df, self.num_threads)
            all_outputs = []
            for chunk in df_chunks:
                all_outputs += self.__execute(chunk)
        else:
            all_outputs = self.__execute(df)

        df['clean_pdb_dir'] = all_outputs
        return df

    def convert_cif_to_pdb(self, cif_file: Path, output_dir: Path) -> Path:
        pdb_path = output_dir / (cif_file.stem + ".pdb")
        modified_pdb_path = output_dir / (cif_file.stem + "_modified.pdb")

        subprocess.run(
            ["obabel", str(cif_file), "-O", str(pdb_path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        ligand_resnames = set()
        filtered_lines = []
        with open(pdb_path, "r") as infile, open(modified_pdb_path, "w") as outfile:
            for line in infile:
                if line.startswith(("ATOM", "HETATM", "TER")):
                    filtered_lines.append(line)
                    if line.startswith("ATOM"):
                        resname = line[17:20].strip()
                        if resname not in STANDARD_RESIDUES:
                            ligand_resnames.add(resname)

            self.relabel_hetatm_and_rename_atoms(filtered_lines, outfile, ligand_resnames)

        return modified_pdb_path

    def relabel_hetatm_and_rename_atoms(self, lines: list[str], outfile, ligand_resnames: set[str]):
        chain_counters = defaultdict(int)
        residue_map = {}
        for line in lines:
            if line.startswith('ATOM'):
                resname = line[17:20].strip()
                chain_id = line[21]
                resseq = line[22:26].strip()

                if resname in ligand_resnames:
                    line = 'HETATM' + line[6:]
                    key = (chain_id, resname, resseq)
                    if key not in residue_map:
                        chain_counters[chain_id] += 1
                        residue_map[key] = chain_counters[chain_id]
                    new_resseq = residue_map[key]
                    line = line[:22] + f"{new_resseq:>4}" + line[26:]

            # Rename atom name
            if line.startswith(("ATOM", "HETATM")):
                atom_name = line[12:16].strip()
                if '_' in atom_name:
                    atom_name = atom_name.split('_')[0]
                    line = line[:12] + f"{atom_name:>4}" + line[16:]

            outfile.write(line)

    def clean_pdb(self, pdb_file_path: Path) -> Path:
        with open(pdb_file_path, 'r') as f:
            lines = f.readlines()

        filtered_chains = self.remove_duplicate_chains(lines)
        cleaned_lines = self.remove_duplicate_ligands(filtered_chains)

        cleaned_path = pdb_file_path.with_name(pdb_file_path.stem + "_cleaned.pdb")

        with open(cleaned_path, 'w') as f:
            f.writelines(cleaned_lines)

        return cleaned_path

    def relabel_hetatm(self, pdb_input, pdb_output, ligand_resnames):
        from collections import defaultdict

        chain_counters = defaultdict(int)
        residue_map = {}

        with open(pdb_input, 'r') as infile, open(pdb_output, 'w') as outfile:
            for line in infile:
                if line.startswith('ATOM'):
                    resname = line[17:20].strip()
                    chain_id = line[21]
                    resseq = line[22:26].strip()

                    if resname in ligand_resnames:
                        # Convert to HETATM
                        line = 'HETATM' + line[6:]

                        # Remap residue number
                        key = (chain_id, resname, resseq)
                        if key not in residue_map:
                            chain_counters[chain_id] += 1
                            residue_map[key] = chain_counters[chain_id]
                        new_resseq = residue_map[key]
                        line = line[:22] + f"{new_resseq:>4}" + line[26:]

                        # Rename atom name like 'C1_1' → 'C1'
                        atom_name = line[12:16].strip()
                        if '_' in atom_name:
                            atom_name = atom_name.split('_')[0]
                            # Right-align to 4 characters, as per PDB format
                            line = line[:12] + f"{atom_name:>4}" + line[16:]
                            print('test')

                outfile.write(line)

    def remove_duplicate_chains(self, pdb_lines):
        chain_sequences = {}
        chains_to_keep = set()
        sequence_to_chain = {}

        for line in pdb_lines:
            if line.startswith('ATOM'):
                resname = line[17:20].strip()
                chain_id = line[21]
                chain_sequences.setdefault(chain_id, []).append(resname)

        for chain, residues in chain_sequences.items():
            seq_str = ''.join(residues)
            if seq_str not in sequence_to_chain:
                sequence_to_chain[seq_str] = chain
                chains_to_keep.add(chain)

        return [line for line in pdb_lines if line[21] in chains_to_keep or not line.startswith("ATOM")]

    def remove_duplicate_ligands(self, pdb_lines):
        ligand_chain_map = {}
        result_lines = []

        for line in pdb_lines:
            if line.startswith("HETATM"):
                lig = line[17:20].strip()
                chain = line[21].strip()
                if lig not in ligand_chain_map:
                    ligand_chain_map[lig] = chain
                    result_lines.append(line)
                elif ligand_chain_map[lig] == chain:
                    result_lines.append(line)
            else:
                result_lines.append(line)

        return result_lines
