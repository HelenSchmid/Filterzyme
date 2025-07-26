from steps.step import Step
import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PLACER(Step):

    def __init__(self, input_col: str, output_dir: str, ligand_name: str, predict_ligand: str, num_threads: int = 1, nsamples: int = 10, rerank: str = "prmsd"):
        self.input_col = input_col
        self.output_dir = Path(output_dir)
        self.ligand_name = ligand_name 
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.predict_ligand = predict_ligand
        self.num_threads = num_threads or 1
        self.nsamples = nsamples
        self.rerank = rerank

    def extract_model1_atoms(self, file_path: Path) -> list[str]:
        """Extract ATOM lines from MODEL 1, rename atoms, and assign chain ID."""
        atoms = []
        in_model1 = False
        atom_counts = {}  # Dictionary to track atom name counts

        with open(file_path) as f:
            for line in f:
                line = line.rstrip('\n')
                if line.startswith("MODEL") and "1" in line:
                    in_model1 = True
                    atom_counts = {}  # Reset counts for each model (though we only process model 1)
                elif line.startswith("ENDMDL") and in_model1:
                    break
                elif in_model1 and line.startswith("ATOM"):
                    # Change 'ATOM' to 'HETATM'
                    modified_line = "HETATM" + line[6:]

                    # Get the atom name (e.g., "C", "O", "N")
                    atom_name = line[12:16].strip()

                    # Increment the count for this atom name
                    atom_counts[atom_name] = atom_counts.get(atom_name, 0) + 1
                    atom_number = atom_counts[atom_name]

                    # Append the number to the atom name
                    modified_atom_name = f"{atom_name}{atom_number}"
                    modified_line = f"{modified_line[:12]:<12}{modified_atom_name:<4}{modified_line[16:]}"

                    # Insert chain identifier 'L' at the correct position (column 21)
                    modified_line = modified_line[:21] + "L" + modified_line[22:]
                    atoms.append(modified_line + '\n')
        return atoms

    def __execute(self, df: pd.DataFrame) -> list:
        placer_dirs = []

        for input_dir in df[self.input_col]:
            input_dir = Path(input_dir).parent
            if not input_dir.exists() or not input_dir.is_dir():
                logger.warning(f"Input directory not found or not a directory: {input_dir}")
                placer_dirs.append(None)
                continue

            result_dirs = []

            # Locate the ligand file ending with ligand_name
            ligand_files = list(input_dir.glob(f"*{self.ligand_name}.pdb"))
            if not ligand_files:
                logger.warning(f"No ligand file found in {input_dir} with pattern *{self.ligand_name}.pdb")
                placer_dirs.append(None)
                continue
            ligand_file = ligand_files[0]  # Assuming one per dir
            
            
            for input_path in input_dir.glob("*_AF2.pdb"):
                if not input_path.is_file():
                    continue
                
                with open(input_path, 'r') as f:
                    lines = f.readlines()

                # Find the index of the 'END' line
                try:
                    end_index = next(i for i, line in enumerate(lines) if line.strip() == 'END')
                except StopIteration:
                    end_index = len(lines)  # Append at end if 'END' is missing

                # Find index of start of real sequence
                first_atom_index = -1  # Initialize to -1 to indicate not found

                for i, line in enumerate(lines):
                    if line.strip().startswith("ATOM"):
                        first_atom_index = i
                        break

                # Extract ATOM lines from MODEL 1
                model1_atoms = self.extract_model1_atoms(ligand_file)

                # Insert ligand ATOMs before 'END'
                ligand_block = [line + '\n' if not line.endswith('\n') else line for line in model1_atoms]
                lines = lines[first_atom_index:end_index-2] + ligand_block #+ ['END\n']


                # Renumber all ATOM and HETATM lines sequentially
                final_lines = []
                atom_counter = 1
                for line in lines:
                    if line.startswith("ATOM") or line.startswith("HETATM"):
                        final_lines.append(f"{line[:6]:<6}{atom_counter:>5d}{line[11:]}")
                        atom_counter += 1
                    else:
                        final_lines.append(line)

                new_file_path = input_path.with_name(input_path.stem + "_withligand.pdb")

                # Write updated content back to file
                with open(new_file_path, 'w') as f:
                    f.writelines(final_lines)

            # Make new directory for results
            output_subdir = self.output_dir / f"{input_path.stem.replace('_AF2', '')}"
            
            output_subdir.mkdir(parents=True, exist_ok=True)

            command = [
                "python", "PLACER/run_PLACER.py",
                "--ifile", str(new_file_path),
                "--odir", str(output_subdir),
                "--rerank", self.rerank,
                "-n", str(self.nsamples),
                "--predict_ligand", self.predict_ligand
            ]

            logger.info(f"Running PLACER on {input_path.name}")
            result = subprocess.run(command, capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"PLACER failed on {input_path.name}:\n{result.stderr}")
            else:
                result_dirs.append(str(self.output_dir))

            placer_dirs.append(result_dirs if result_dirs else None)

        return placer_dirs

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.output_dir:
            if self.num_threads > 1:
                all_dirs = []
                df_chunks = np.array_split(df, self.num_threads)
                for chunk in df_chunks:
                    all_dirs += self.__execute(chunk)
                df["placer_dir"] = all_dirs
            else:
                df["placer_dir"] = self.__execute(df)
            return df
        else:
            print("No output directory provided")
            return df

