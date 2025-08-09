import pandas as pd
from pathlib import Path

from filtering_pipeline.pipeline import Pipeline
from filtering_pipeline.pipeline import Docking
from filtering_pipeline.pipeline import Superimposition
from filtering_pipeline.pipeline import GeometricFilters

df = pd.read_pickle('/nvme2/helen/masterthesis/3_manuscript/DB_pipeline_short_enzymes_vina_only.pkl')

if __name__ == "__main__":

    base_out = Path("pipeline_output_vina_only")
    (base_out / "docking").mkdir(parents=True, exist_ok=True)

    # --- Create Docking with your settings (we will call only the parts we need) ---
    docking = Docking(
        df=df,
        output_dir=base_out / "docking",
        squidly_dir='filtering_pipeline/squidly_final_models/',
        metagenomic_enzymes=0,   # set to 1 if you want to use Boltz structures as receptor input
        num_threads=1
    )


    df_squidly = docking._catalytic_residue_prediction()

    df_vina = docking._run_vina(df_squidly)

    print("Vina-only docking finished. Results saved to:", base_out / "docking" / "vina.pkl")
