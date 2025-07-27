import os
import sys
import pandas as pd
import logging
from pathlib import Path

from filtering_pipeline.steps.step import Step
from filtering_pipeline.steps.save_step import Save
from filtering_pipeline.steps.preparevina_step import PrepareVina
from filtering_pipeline.steps.preparechai_step import PrepareChai
from filtering_pipeline.steps.prepareboltz_step import PrepareBoltz


def log_section(title: str):
    border = "#" * 60
    logging.info(f"\n{border}")
    logging.info(f"### {title.upper().center(52)} ###")
    logging.info(f"{border}\n")

def log_subsection(title: str):
    border = "#" * 60
    logging.info(f"\n{border}")
    logging.info(f"### {title.center(52)} ###")
    logging.info(f"{border}\n")


def prepare_files_for_superimposition(df_vina = 'vina.pkl', df_chai = 'chai.pkl', ligand_name: str = '', output_dir = 'preparedfiles_for_superimposition/'):
    '''
    Format output files of the various docking tools into coherent format to use as input for superimposing them onto each other. 
    Output: Dataframe combining all the paths to the prepared PDB files. 
    '''
    # Prepare vina files
    df_vina = pd.read_pickle(df_vina)
    prepared_vina = df_vina << (PrepareVina('output_dir', ligand_name,  output_dir))

    # Prepare chai files
    df_chai = pd.read_pickle(df_chai)
    prepared_chai = df_chai << (PrepareChai('output_dir', output_dir, 1))

    # Prepare boltz files
    df_boltz = df_vina.copy()
    df_boltz['boltz_dir'] = df_vina['output_dir'].apply(
        lambda x: str(Path(x).parent).replace('vina', 'boltz') if isinstance(x, (str, Path)) else None)
    
    # Filter out invalid boltz_dir entries
    df_boltz = df_boltz[df_boltz['boltz_dir'].notna()]
    df_boltz = df_boltz[df_boltz['boltz_dir'].apply(lambda x: isinstance(x, (str, Path)))]

    prepared_boltz = df_boltz << (PrepareBoltz('boltz_dir' , output_dir, 1))

    # Combine prepared dataframes
    df_combined = prepared_vina.merge(
        prepared_chai[['Entry', 'chai_files_for_superimposition']],
        on='Entry',
        how='left'  
    ).merge(
        prepared_boltz[['Entry', 'boltz_files_for_superimposition']],
        on='Entry',
        how='left'  
    )

    df = df_combined[
        df_combined['vina_files_for_superimposition'].notna() &
        df_combined['chai_files_for_superimposition'].notna() &
        df_combined['vina_files_for_superimposition'].apply(lambda x: isinstance(x, list)) &
        df_combined['chai_files_for_superimposition'].apply(lambda x: isinstance(x, list))
    ]

    current_script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    output_dir = Path(current_script_dir)
    df.to_pickle(f'{output_dir}/df_for_superimposition')


