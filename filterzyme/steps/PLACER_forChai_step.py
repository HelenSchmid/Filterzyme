import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import logging

from filterzyme.steps.step import Step

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class PLACER(Step):
    """
    PLACER step to predict ligand binding sites.
    - Inputs: *_0_chai.pdb structures.
    - Runs PLACER once per unique entry.
    - Outputs: Flat directory of CSV results.
    - Aggregates all individual CSVs into 'placer_summary.csv'.
    """

    def __init__(
        self,
        input_col: str,
        predict_ligand: str,
        output_dir: str = "placer_output",
        entry_col: str = "docked_structure",
        num_threads: int = 1,
        nsamples: int = 50,
        rerank: str = "prmsd",
    ):
        self.entry_col = entry_col
        self.input_col = input_col
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.predict_ligand = predict_ligand
        self.num_threads = num_threads or 1
        self.nsamples = nsamples
        self.rerank = rerank

    def _select_0_chai(self, pdb_list):
        """Return Path to *_0_chai.pdb or None."""
        if not isinstance(pdb_list, (list, tuple)):
            return None
        for p in pdb_list:
            p = Path(p)
            if p.name.endswith("0_chai.pdb"):
                return p
        return None

    def __execute(self, df: pd.DataFrame) -> dict:
        """
        Run PLACER once per unique Entry.
        Returns dict: {Entry -> input_pdb_stem} 
        (We return the PDB stem to help locate the CSV later)
        """
        entry_to_pdb = {}

        # 1. Identify valid inputs
        for _, row in df.iterrows():
            entry = row[self.entry_col]
            if pd.isna(entry) or "0_chai" not in str(entry):
                continue
            pdb = self._select_0_chai(row[self.input_col])
            if pdb:
                entry_to_pdb[entry] = pdb

        if not entry_to_pdb:
            logger.warning("No *_0_chai.pdb files found")
            return {}

        # 2. Setup script path
        placer_script = (
            Path(__file__).resolve().parent.parent / "PLACER" / "run_PLACER.py"
        )
        
        if not placer_script.exists():
            logger.error(f"PLACER script not found at {placer_script}")
            return {}

        processed_stems = {}

        # 3. Run Subprocess
        for entry, input_path in sorted(entry_to_pdb.items()):
            if not input_path.exists():
                logger.warning(f"Input PDB not found for {entry}: {input_path}")
                continue

            command = [
                "python", str(placer_script),
                "--ifile", str(input_path),
                "--odir", str(self.output_dir), 
                "--rerank", self.rerank,
                "-n", str(self.nsamples),
                "--predict_multi",
                "--predict_ligand", self.predict_ligand,
            ]

            logger.info(f"Running PLACER on {entry}")

            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                )

                if result.returncode != 0:
                    logger.error(f"PLACER failed for {entry}\n{result.stderr}")
                else:
                    # Store the stem (filename without extension) to find the CSV later
                    processed_stems[entry] = input_path.stem
            except Exception as e:
                logger.error(f"Error running subprocess for {entry}: {e}")

        return processed_stems

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes PLACER, looks for CSVs in the output dir matching input filenames,
        aggregates them, and updates the DataFrame.
        """
        # Run PLACER
        # entry_to_stem: {'entry_id': 'protein_structure_0_chai'}
        entry_to_stem = self.__execute(df)

        # --- Aggregation Logic ---
        all_dfs = []
        
        # Invert map to find entry_id by filename stem: {'protein_structure_0_chai': 'entry_id'}
        stem_to_entry = {v: k for k, v in entry_to_stem.items()}
        
        # Search for CSVs in the output directory
        for csv_file in self.output_dir.glob("*.csv"):
            
            # Skip the summary file itself if it already exists
            if csv_file.name == "placer_results.csv":
                continue

            # Try to match CSV filename to an entry
            matched_entry = None
            for stem, entry in stem_to_entry.items():
                if csv_file.name.startswith(stem):
                    matched_entry = entry
                    break
            
            if matched_entry:
                try:
                    sub_df = pd.read_csv(csv_file)
                    # Add metadata columns
                    sub_df.insert(0, "entry_id", matched_entry)
                    sub_df["source_csv"] = csv_file.name
                    all_dfs.append(sub_df)
                except Exception as e:
                    logger.warning(f"Failed to read CSV {csv_file}: {e}")
        
        # Concatenate and Save
        if all_dfs:
            summary_df = pd.concat(all_dfs, ignore_index=True)
            summary_path = self.output_dir / "placer_results.csv"
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"Combined summary CSV saved to: {summary_path}")
        else:
            logger.warning("No matching CSVs were found to aggregate.")

        # Update df with the directory path (since they are all in the same dir)
        df["placer_dir"] = df[self.entry_col].apply(
            lambda x: str(self.output_dir) if x in entry_to_stem else None
        )
        
        return df