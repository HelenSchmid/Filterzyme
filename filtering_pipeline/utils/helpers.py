import os
import sys
import pandas as pd
import logging

from ..steps.step import Step
from ..steps.save_step import Save
from ..steps.preparevina_step import PrepareVina
from ..steps.preparechai_step import PrepareChai
from ..steps.prepareboltz_step import PrepareBoltz


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)


def log_section(title: str):
    border = "#" * 60
    logger.info(f"\n{border}")
    logger.info(f"### {title.center(54)} ###")
    logger.info(f"{border}\n")


def prepare_files_for_superimposition(df_vina = 'vina.pkl', df_chai = 'chai.pkl', ligand_name: str = '', output_dir = 'preparedfiles_for_superimposition/'):
    '''
    Format output files of the various docking tools into coherent format to use as input for superimposing them onto each other. 
    Output: Dataframe combining all the paths to the prepared PDB files. 
    '''
    # Prepare vina files
    df_vina = pd.read_pickle(df_vina)
    df_vina << (PrepareVina('output_dir', ligand_name,  output_dir) >> Save('preparedfiles_vina.pkl'))

    # Prepare chai files
    df_chai = pd.read_pickle(df_chai)
    df_chai << (PrepareChai('output_dir', output_dir, 1) >> Save('preparedfiles_chai.pkl'))

    # Prepare boltz files
    df_boltz = df_vina.copy()
    df_boltz['boltz_dir'] = df_vina['output_dir'].apply(
        lambda x: str(Path(x).parent).replace('vina', 'boltz') if isinstance(x, (str, Path)) else None)
    
    # Filter out invalid boltz_dir entries
    df_boltz = df_boltz[df_boltz['boltz_dir'].notna()]
    df_boltz = df_boltz[df_boltz['boltz_dir'].apply(lambda x: isinstance(x, (str, Path)))]

    df_boltz << (PrepareBoltz('boltz_dir' , output_dir, 1) >> Save('preparedfiles_boltz.pkl'))

    # Combine prepared dataframes
    df_chai = pd.read_pickle('preparedfiles_chai.pkl') 
    df_vina = pd.read_pickle('preparedfiles_vina.pkl')
    df_boltz = pd.read_pickle('preparedfiles_boltz.pkl')

    df_combined = df_vina.merge(
        df_chai[['Entry', 'chai_files_for_superimposition']],
        on='Entry',
        how='left'  
    ).merge(
        df_boltz[['Entry', 'boltz_files_for_superimposition']],
        on='Entry',
        how='left'  
    )

    df = df_combined[
        df_combined['vina_files_for_superimposition'].notna() &
        df_combined['chai_files_for_superimposition'].notna() &
        df_combined['vina_files_for_superimposition'].apply(lambda x: isinstance(x, list)) &
        df_combined['chai_files_for_superimposition'].apply(lambda x: isinstance(x, list))
    ]

    df.to_pickle('df_for_superimposition')

