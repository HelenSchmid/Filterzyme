import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import logging
import time

from filterzyme.steps.step import Step

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class PLACER(Step):
    """
    PLACER step to predict ligand binding sites.
    - Robust handling for Single vs Multi ligand detection.
    - Fallback logic: If Multi-mode fails, retries as Single-mode.
    - Aggregates results into 'placer_results.csv'.
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
        self.num_threads = num_threads
        self.nsamples = nsamples
        self.rerank = rerank
        
        # Path to the script
        self.placer_script = (
            Path(__file__).resolve().parent.parent / "PLACER" / "run_PLACER.py"
        )

    def _select_0_chai(self, pdb_list):
        """Return Path to *_0_chai.pdb or None."""
        if not isinstance(pdb_list, (list, tuple)):
            return None
        for p in pdb_list:
            p = Path(p)
            if p.name.endswith("0_chai.pdb"):
                return p
        return None

    def _count_ligands(self, pdb_path: Path, ligand_resname: str) -> int:
        """Counts distinct instances (Chain + ResSeq) of the ligand."""
        unique_ligands = set()
        target_resname = ligand_resname.strip().upper()
        try:
            with open(pdb_path, 'r') as f:
                for line in f:
                    if line.startswith(("HETATM", "ATOM")):
                        res_name = line[17:20].strip()
                        if res_name == target_resname:
                            # Unique ID is Chain (21) + ResSeq (22-26)
                            unique_id = (line[21], line[22:26])
                            unique_ligands.add(unique_id)
            return len(unique_ligands)
        except Exception:
            return 0

    def _run_placer_subprocess(self, cmd, entry):
        """Helper to run the subprocess and handle basic logging."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )
            return result
        except Exception as e:
            logger.error(f"Subprocess exception for {entry}: {e}")
            return None

    def __execute(self, df: pd.DataFrame) -> dict:
        """
        Run PLACER once per unique Entry with Fallback Logic.
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

        if not self.placer_script.exists():
            logger.error(f"PLACER script not found at {self.placer_script}")
            return {}

        processed_stems = {}

        # 2. Iterate and Run
        for entry, input_path in sorted(entry_to_pdb.items()):
            if not input_path.exists():
                continue

            # Check if file already exists 
            existing_files = list(self.output_dir.glob(f"{input_path.stem}*.csv"))
            existing_files = [f for f in existing_files if "placer_summary.csv" not in f.name]

            if existing_files:
                logger.info(f"Skipping {entry}, output already exists: {existing_files[0].name}")
                processed_stems[entry] = input_path.stem
                continue

            # --- Base Command ---
            base_cmd = [
                "python", str(self.placer_script),
                "--ifile", str(input_path),
                "--odir", str(self.output_dir), 
                "--rerank", self.rerank,
                "-n", str(self.nsamples),
                "--predict_ligand", self.predict_ligand,
            ]

            # --- Logic Decision ---
            n_ligands = self._count_ligands(input_path, self.predict_ligand)
            success = False
            
            # CASE A: We detected multiple ligands -> Try Multi Mode
            if n_ligands > 1:
                logger.info(f"Entry {entry}: Detected {n_ligands} ligands. Attempting multi-ligand mode...")
                
                multi_cmd = base_cmd + ["--predict_multi"]
                result = self._run_placer_subprocess(multi_cmd, entry)

                if result and result.returncode == 0:
                    success = True
                elif result and "CUDA out of memory" in result.stderr:
                     logger.error(f"CUDA OOM on {entry}. Skipping to prevent stall.")
                     success = False # Do not retry OOM, it will just fail again
                elif result and "AssertionError" in result.stderr:
                    logger.warning(f"Entry {entry}: Multi-mode rejected by PLACER (likely recognized only 1). Retrying single-mode...")
                    # Fallback to single mode immediately
                    single_cmd = base_cmd
                    result = self._run_placer_subprocess(single_cmd, entry)
                    if result and result.returncode == 0:
                        success = True
                    else:
                        logger.error(f"Retry failed for {entry}: {result.stderr if result else 'Unknown error'}")

            # CASE B: Single ligand detected -> Run Standard Mode
            else:
                logger.info(f"Entry {entry}: Single ligand detected. Running standard mode...")
                result = self._run_placer_subprocess(base_cmd, entry)
                
                if result and result.returncode == 0:
                    success = True
                elif result and "CUDA out of memory" in result.stderr:
                     logger.error(f"CUDA OOM on {entry}.")
                elif result:
                    logger.error(f"PLACER failed for {entry}: {result.stderr}")

            if success:
                processed_stems[entry] = input_path.stem

        return processed_stems

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        entry_to_stem = self.__execute(df)

        # --- Aggregation Logic ---
        all_dfs = []
        stem_to_entry = {v: k for k, v in entry_to_stem.items()}
        
        # Scan output directory for results matching our successful runs
        for csv_file in self.output_dir.glob("*.csv"):
            if csv_file.name == "placer_results.csv":
                continue

            matched_entry = None
            for stem, entry in stem_to_entry.items():
                if csv_file.name.startswith(stem):
                    matched_entry = entry
                    break
            
            if matched_entry:
                try:
                    sub_df = pd.read_csv(csv_file)
                    sub_df.insert(0, "entry_id", matched_entry)
                    sub_df["source_csv"] = csv_file.name
                    all_dfs.append(sub_df)
                except Exception as e:
                    logger.warning(f"Failed to read CSV {csv_file}: {e}")
        
        if all_dfs:
            summary_df = pd.concat(all_dfs, ignore_index=True)
            summary_path = self.output_dir / "placer_results.csv"
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"Combined summary CSV saved to: {summary_path}")

        # Map results back
        df["placer_dir"] = df[self.entry_col].apply(
            lambda x: str(self.output_dir) if x in entry_to_stem else None
        )
        
        return df