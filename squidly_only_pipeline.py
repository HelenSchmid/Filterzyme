import pandas as pd
import os
import torch
import sys
import argparse
from pathlib import Path 

sys.path.insert(0, '/nvme2/helen/EnzymeStructuralFiltering/')
import filtering_pipeline
print(filtering_pipeline.__file__)  

from filtering_pipeline.steps.save_step import Save
from filtering_pipeline.steps.predict_catalyticsite_step import ActiveSitePred


vina_df = pd.read_pickle('/nvme2/helen/masterthesis/3_manuscript/DB_pipeline_short_enzymes_vina_only.pkl')

df_unique_sequences = vina_df.drop_duplicates(subset='Sequence', keep='first')
df_cat_res = df_unique_sequences << ActiveSitePred('Entry', 'Sequence', 'filtering_pipeline/squidly_final_models/', 1)
df_squidly = pd.merge(vina_df, df_cat_res, left_on='Entry', right_on='label', how='inner')

df_squidly