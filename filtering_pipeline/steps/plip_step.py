import pandas as pd
from pathlib import Path
import logging
from tempfile import TemporaryDirectory
from plip.structure.preparation import PDBComplex
from filtering_pipeline.steps.step import Step
from filtering_pipeline.utils.helpers import suppress_stdout_stderr


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class PLIP(Step):
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
            pdb_file_as_path = Path(self.input_dir / f"{best_structure_name}.pdb")
            pdb_file_as_str = str(self.input_dir / f"{best_structure_name}.pdb")
            row_result = {}

            print(f"Processing PDB file: {pdb_file_as_path.name}")
            
            try:

                # Default result structure
                default_result = {
                    'plip_hydrogen_nbonds': None,
                    'plip_hydrophobic_contacts': None,
                    'plip_salt_bridges': None,
                    'plip_pi_stacking': None, 
                    'plip_pi_cation': None,
                    'plip_halogen_bonds': None, 
                    'plip_water_bridges': None,
                    'plip_metal_complexes': None,
                }

                # Load and analyze the docked structure
                with suppress_stdout_stderr():
                    prot = PDBComplex()
                    prot.load_pdb(pdb_file_as_str)
                    prot.analyze()


                # Always use first interaction set i.e. first ligand
                lig_id = list(prot.interaction_sets.keys())[0]
                interactions = prot.interaction_sets[lig_id]

                # Count interactions
                num_hbonds = len(interactions.hbonds_ldon) + len(interactions.hbonds_pdon)
                num_hydrophobics = len(interactions.hydrophobic_contacts)
                num_saltbridges = len(interactions.saltbridge_pneg) + len(interactions.saltbridge_lneg)
                num_pistacking = len(interactions.pistacking)
                num_pication = len(interactions.pication_laro) + len(interactions.pication_paro)
                num_halogen = len(interactions.halogen_bonds)
                num_waterbridges = len(interactions.water_bridges)
                num_metal = len(interactions.metal_complexes) 

                # Update row_result with interaction counts
                row_result['plip_hydrogen_nbonds'] = num_hbonds
                row_result['plip_hydrophobic_contacts'] = num_hydrophobics
                row_result['plip_salt_bridges'] = num_saltbridges
                row_result['plip_pi_stacking'] = num_pistacking
                row_result['plip_pi_cation'] = num_pication
                row_result['plip_halogen_bonds'] = num_halogen
                row_result['plip_water_bridges'] = num_waterbridges
                row_result['plip_metal_complexes'] = num_metal  
                
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
