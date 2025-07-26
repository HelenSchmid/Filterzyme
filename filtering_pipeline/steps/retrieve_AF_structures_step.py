import os
import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import logging
from multiprocessing.dummy import Pool as ThreadPool
from step import Step

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# WTFFFF?????

class PLACER(Step):
    def __init__(self, input_col: str, output_dir: str, predict_ligand: str, nsamples: int = 10, num_threads: int = 1, rerank: str = "prmsd"):
        self.input_col = input_col
        self.output_dir = Path(output_dir)
        self.predict_ligand = predict_ligand
        self.nsamples = nsamples
        self.num_threads = num_threads
        self.rerank = rerank
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def __execute(self, df: pd.DataFrame) -> list:
        placer_paths = []

        for i, row in df.iterrows():
            input_dir = Path(row[self.input_col])

            if not input_dir.exists() or not input_dir.is_dir():
                logger.warning(f"Input directory not found or not a directory: {input_dir}")
                placer_paths.append(None)
                continue

            placer_dirs = []

            for input_file in input_dir.glob("*"):
                if not input_file.is_file():
                    continue

                out_dir = self.output_dir / f"{input_file.stem}_placer_out"
                out_dir.mkdir(parents=True, exist_ok=True)

                cmd = [
                    "python", "PLACER/run_PLACER.py",
                    "--ifile", str(input_file),
                    "--odir", str(out_dir),
                    "--rerank", self.rerank,
                    "-n", str(self.nsamples),
                    "--predict_ligand", self.predict_ligand
                ]

                logger.info(f"Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode != 0:
                    logger.error(f"PLACER failed on {input_file.name}:\n{result.stderr}")
                else:
                    placer_dirs.append(str(out_dir))

            placer_paths.append(placer_dirs if placer_dirs else None)

        return placer_paths

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.reset_index(drop=True)

        if self.num_threads > 1:
            chunks = np.array_split(df, self.num_threads)
            with ThreadPool(self.num_threads) as pool:
                results = pool.map(self.__execute, chunks)
            placer_paths = [path for sublist in results for path in sublist]
        else:
            placer_paths = self.__execute(df)

        df["placer_dir"] = placer_paths
        return df
