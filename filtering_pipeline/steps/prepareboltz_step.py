import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import re
from tempfile import TemporaryDirectory
from multiprocessing.dummy import Pool as ThreadPool

from filtering_pipeline.steps.step import Step


from Bio.PDB import MMCIFParser
from Bio.PDB import PDBIO
from Bio.PDB.Polypeptide import is_aa

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def convert_cif_to_pdb(cif_filepath, pdb_filepath=None):
    """
    Converts a mmCIF file to a PDB file. Labels non-amino acid residues as LIG.
    """
    cif_filepath = Path(cif_filepath)

    if pdb_filepath is None:
        pdb_filepath = cif_filepath.with_suffix('.pdb')
    else:
        pdb_filepath = Path(pdb_filepath)

    parser = MMCIFParser(QUIET=True)
    try:
        structure_id = cif_filepath.stem
        structure = parser.get_structure(structure_id, str(cif_filepath))

        for model in structure:
            for boltzn in model:
                for residue in boltzn:
                    hetflag, resseq, icode = residue.id
                    # Rename ligand residues to 'LIG' (optional)
                    if hetflag != " " and not is_aa(residue, standard=True):
                        residue.resname = "LIG"

                    # Clean up atom names
                    for atom in residue:
                        if "_" in atom.fullname.strip():
                            # Only keep part before underscore, and pad to 4 characters
                            base_name = atom.fullname.strip().split("_")[0]
                            # Right-align and pad to length 4 
                            atom.fullname = f"{base_name:>4}"

        io = PDBIO()
        io.set_structure(structure)
        io.save(str(pdb_filepath))
        return pdb_filepath

    except Exception as e:
        print(f"Error converting {cif_filepath}: {e}")
        return None


class PrepareBoltz(Step):
    def __init__(self, boltz_dir = None,  output_dir: str = '' , num_threads=1):
        self.boltz_dir = boltz_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_threads = num_threads or 1

    def __execute(self, df: pd.DataFrame, output_dir: str) -> list:
        results = []

        for boltz_path in df[self.boltz_dir]:

            boltz_path = Path(boltz_path)
            if not boltz_path.exists():
                logger.warning(f"Boltz path not found: {boltz_path}")
                results.append(None)
                continue
            
            entry_name = boltz_path.name
            boltz_files = Path(boltz_path) / f"boltz_results_{entry_name}" / 'predictions' / entry_name
            pdb_file_paths = []

            for boltz_file_path in boltz_files.glob('*.cif'):
                entry_name = boltz_file_path.stem
                pdb_filepath = self.output_dir / f"{entry_name}_boltz.pdb"

                try:
                    # Convert mmCIF to PDB files
                    convert_cif_to_pdb(boltz_file_path, pdb_filepath)
                    pdb_file_paths.append(str(pdb_filepath))

                except Exception as e:
                    logger.error(f"Error processing {boltz_path}: {e}")
                    pdb_file_paths.append(None)
            results.append(pdb_file_paths)

        return results


    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.output_dir:
            if self.num_threads > 1:
                output_filenames = []
                df_list = np.array_split(df, self.num_threads)
                for df_chunk in df_list:
                    output_filenames += self.__execute(df_chunk, self.output_dir)
                    
                df['boltz_files_for_superimposition'] = output_filenames
                return df
            
            else:
                output_filenames = self.__execute(df, self.output_dir)
                df['boltz_files_for_superimposition'] = output_filenames
                return df
        else:
            print('No output directory provided')