import os
import sys
import pandas as pd

from steps.step import Step


def extract_docking_metrics_vina(input_dir):
    '''
    Function to extract binding affinities from the vina docking prediction. 
    '''
    for vina_path in df[self.vina_dir].apply(str):

            vina_path = Path(vina_path)
            if not vina_path.exists():
                logger.warning(f"Vina path not found: {vina_path}")
                results.append(None)
                continue

            entry_name = vina_path.stem
            ligand_file = vina_path.parent / f'{entry_name}-{self.ligand_name}.pdb'

            try:





def extract_docking_metrics_chai():



def extract_docking_metrics_boltz():

