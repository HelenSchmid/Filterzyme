""" Test run placer"""


import pandas as pd
import os
import numpy as np
import torch
import sys
import argparse
from pathlib import Path 

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s |%(message)s",
)

sys.path.insert(0, '/nvme2/helen/EnzymeStructuralFiltering/')
from filterzyme.steps.PLACER_forChai_step import PLACER   
import filterzyme

df = pd.read_pickle('/nvme2/helen/EnzymeStructuralFiltering/benchmarking/serine_hydrolases/filterzyme_output/geometricfiltering/structural_features_final.pkl')

# Instantiate the step
placer = PLACER(
    input_col="chai_files_for_superimposition",
    predict_ligand="LIG",     # example ligand
    num_threads=2,            # parallel PLACER runs
    rerank="prmsd"
)

# Run
placer.execute(df)

