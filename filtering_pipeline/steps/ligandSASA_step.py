import pandas as pd
from pathlib import Path
import logging
from tempfile import TemporaryDirectory
from Bio.PDB import PDBParser, Select, PDBIO
import freesasa
freesasa.setVerbosity(1)

from filtering_pipeline.steps.step import Step
from filtering_pipeline.utils.helpers import extract_ligand_from_PDB, LigandSelect


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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



class LigandSASA(Step):
    def __init__(self, input_dir = None,  output_dir: str = ''):

        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)        

    def __execute(self, df: pd.DataFrame, tmp_dir: str) -> list:

        if not self.input_dir.exists():
            raise FileNotFoundError(f"Directory does not exist: {self.input_dir}")
        
        results = []

        for _, row in df.iterrows():
            entry_name = row['Entry']
            best_structure_name = row['best_structure']
            pdb_file = Path(self.input_dir / f"{best_structure_name}.pdb")
            row_result = {}

            print(f"Processing PDB file: {pdb_file.name}")
            
            try:

                # Default result structure
                default_result = {
                    'sasa_ligand_in_complex': None,
                    'sasa_ligand_alone': None,
                    'buried_sasa': None,
                    'percent_buried_sasa': None
                }

                # Extract ligand from PDB file containing docked protein-ligand structure and save in temporary directory
                
                with TemporaryDirectory() as tmpdir: 
                    ligand_path = Path(tmpdir) / "ligand.pdb"
                    extract_ligand_from_PDB(pdb_file, ligand_path, 'LIG')

                    # Load structures from PDB files
                    structure_complex = freesasa.Structure(str(pdb_file), options={'hetatm': True})
                    structure_ligand = freesasa.Structure(str(ligand_path), options={'hetatm': True})

                # Run SASA calculation
                result_ligand = freesasa.calc(structure_ligand)
                result_complex = freesasa.calc(structure_complex)

                # Get SASA values
                selection = ["ligand, resn LIG"] #selection = ["ligand, chain B"]
                sasa_ligand_in_complex = freesasa.selectArea(selection, structure_complex, result_complex)
                sasa_ligand_alone = result_ligand.totalArea()

                # Buried SASA = exposed alone - exposed in complex
                buried_sasa = sasa_ligand_alone - sasa_ligand_in_complex["ligand"]
                if sasa_ligand_alone > 0:
                    percent_buried = (buried_sasa / sasa_ligand_alone) * 100
                else:
                    percent_buried = 0.0

                row_result['sasa_ligand_in_complex'] = sasa_ligand_in_complex["ligand"]
                row_result['sasa_ligand_alone'] = sasa_ligand_alone
                row_result['buried_sasa'] = buried_sasa
                row_result['percentage_buried_sasa'] = percent_buried

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
