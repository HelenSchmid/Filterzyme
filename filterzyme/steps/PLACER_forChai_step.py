from step import Step
import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PLACER(Step):

    def __init__(self, input_col: str, output_dir: str, predict_ligand: str, num_threads: int = 1, nsamples: int = 10, rerank: str = "prmsd"):
        self.input_col = input_col
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.predict_ligand = predict_ligand
        self.num_threads = num_threads or 1
        self.nsamples = nsamples
        self.rerank = rerank

    def __execute(self, df: pd.DataFrame) -> list:
        placer_dirs = []

        for input_dir in df[self.input_col]:
            input_dir = Path(input_dir)
            if not input_dir.exists() or not input_dir.is_dir():
                logger.warning(f"Input directory not found or not a directory: {input_dir}")
                placer_dirs.append(None)
                continue

            result_dirs = []

            for input_path in input_dir.glob("*"):  # Filter with "*.pdb" if needed
                if not input_path.is_file():
                    print("error")
                    continue

                #output_subdir = self.output_dir / f"{input_path.stem}_placer_out"
                #output_subdir.mkdir(parents=True, exist_ok=True)


                command = [
                    "python", "PLACER/run_PLACER.py",
                    "--ifile", str(input_path),
                    "--odir", str(self.output_dir),
                    "--rerank", self.rerank,
                    "-n", str(self.nsamples),
                    "--predict_ligand", self.predict_ligand
                ]

                print(command)

                logger.info(f"Running PLACER on {input_path.name}")
                result = subprocess.run(command, capture_output=True, text=True)

                if result.returncode != 0:
                    logger.error(f"PLACER failed on {input_path.name}:\n{result.stderr}")
                else:
                    result_dirs.append(str(self.output_dir))

            placer_dirs.append(result_dirs if result_dirs else None)

        return placer_dirs

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.output_dir:
            if self.num_threads > 1:
                all_dirs = []
                df_chunks = np.array_split(df, self.num_threads)
                for chunk in df_chunks:
                    all_dirs += self.__execute(chunk)
                df["placer_dir"] = all_dirs
            else:
                df["placer_dir"] = self.__execute(df)
            return df
        else:
            print("No output directory provided")
            return df

