import pandas as pd
from pathlib import Path


from filtering_pipeline.pipeline import Pipeline
from filtering_pipeline.pipeline import Docking
from filtering_pipeline.pipeline import Superimposition
from filtering_pipeline.pipeline import GeometricFilters


df = pd.read_pickle('/nvme2/helen/masterthesis/3_manuscript/DB_pipeline_short_enzymes_vina_only.pkl')

if __name__ == "__main__":

    # Configure and run
    pipeline = Pipeline(
        df = df,
        smarts_pattern='[$([CX3](=O)[OX2H0][#6])]',
        max_matches=1000,
        num_threads=1,
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

    pipeline.run()

