import pandas as pd
import os
import numpy as np
import torch
import sys
import argparse
from pathlib import Path 

sys.path.insert(0, '/nvme2/helen/EnzymeStructuralFiltering/')
import filterzyme
print(filterzyme.__file__)  

from filterzyme.pipeline_v2 import Pipeline
from filterzyme.pipeline_v2 import Docking
from filterzyme.pipeline_v2 import Superimposition
from filterzyme.pipeline_v2 import GeometricFilters


df = pd.read_pickle('MSR_input_data.pkl')

base_output_dir = "filterzyme_output"

if __name__ == "__main__":

    # Configure and run
    pipeline = Pipeline(
        df = df,
        max_matches=1000,
        esterase = 1,
        num_threads=1,
        metagenomic_enzymes=0,
        skip_catalytic_residue_prediction = True, 
        alternative_structure_for_vina = 'Chai', 
        #squidly_dir='/nvme2/helen/EnzymeStructuralFiltering/filtering_pipeline/squidly_final_models/',
        base_output_dir=base_output_dir, 
    )

pipeline.run()

"""
    docking = Docking(
        df=df,
        output_dir= Path(base_output_dir) / "docking",
        squidly_dir='/nvme2/helen/EnzymeStructuralFiltering/filtering_pipeline/squidly_final_models/',
        metagenomic_enzymes= 0,
        skip_catalytic_residue_prediction = True,
        alternative_structure_for_vina = "Chai", 
        num_threads=1,
    )

    superimposition = Superimposition(
    maxMatches = 1000,
    num_threads = 1,
    input_dir = Path(base_output_dir) / 'docking',
    output_dir = Path(base_output_dir) / 'superimposition')
    
    #ligandrmsd = superimposition._ligandRMSD(df)

    filtering = GeometricFilters(
        esterase = 1, 
        df = df, 
        input_dir = Path(base_output_dir) / 'superimposition',
        output_dir = Path(base_output_dir) / 'geometricfiltering')    

    #df = pd.read_pickle('/nvme2/helen/masterthesis/3_manuscript/benchmark_martinez/geometricfiltering/ligandSASA.pkl')
    #filtering._plip_interactions(df)
"""

