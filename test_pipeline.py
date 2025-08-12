import pandas as pd
from pathlib import Path


from filtering_pipeline.pipeline import Pipeline
from filtering_pipeline.pipeline import Docking
from filtering_pipeline.pipeline import Superimposition
from filtering_pipeline.pipeline import GeometricFilters



df = pd.read_pickle('/nvme2/helen/masterthesis/3_manuscript/DB_pipeline_short_enzymes.pkl').head(1)

df_sup = pd.read_pickle('pipeline_output_test/superimposition/best_structures.pkl')
df_sup['cofactor_moiety'] = 'C#N'
df_sup['substrate_moiety'] = 'CCC'

if __name__ == "__main__":

    # Configure and run
    pipeline = Pipeline(
        df = df,
        max_matches=1000,
        num_threads=1,
        skip_catalytic_residue_prediction = False,
        metagenomic_enzymes=0,
        squidly_dir='filtering_pipeline/squidly_final_models/',
        base_output_dir="pipeline_output_test", 

    )


    superimposition = Superimposition(
    maxMatches = 1000,
    num_threads = 1,
    input_dir = Path('pipeline_output_test/docking'),
    output_dir = Path('pipeline_output_test/superimposition')
    )

    #super_df= superimposition._ligandRMSD(df_sup)

    filtering = GeometricFilters(
        esterase = 0, 
        find_closest_nuc=1, 
        num_threads=1, 
        df = df_sup,     
        input_dir = Path('pipeline_output_test/superimposition'),
        output_dir = Path('pipeline_output_test/superimposition'))


    pipeline.run()

