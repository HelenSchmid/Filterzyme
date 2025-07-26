from pathlib import Path
from typing import Union
import pandas as pd
import logging
import os

import pandas as pd

from .utils import helpers  
from .steps.predict_catalyticsite_step import ActiveSitePred
from .steps.save_step import Save
from enzymetk.dock_chai_step import Chai
from enzymetk.dock_boltz_step import Boltz
from enzymetk.dock_vina_step import Vina



logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

class Docking:
    def __init__(
        self,
        ligand_name: str,
        ligand_smiles: str,
        df: pd.DataFrame,
        output_dir: Union[str, Path] = "pipeline_output",
        squidly_dir: Union[str, Path] = '',
        num_threads = 1,
        run_docking: bool = True,
        run_superimposition: bool = True,
        run_filtering: bool = True
    ):
        self.ligand_name = ligand_name
        self.ligand_smiles = ligand_smiles
        self.df = df.copy()
        self.squidly_dir = squidly_dir
        self.num_threads = num_threads
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.run_docking = run_docking
        self.run_superimposition = run_superimposition
        self.run_filtering = run_filtering

        # store intermediate results
        self.state = {
            "df": self.df,
            "ligand_smiles": self.ligand_smiles,
            "output_dir": self.output_dir
        }

    def run(self):
        log_section("Start docking")
        
        if self.run_docking: 
            df_squidly = self._catalytic_residue_prediction()
            df_chai = self._run_chai(df_squidly)
            df_boltz= self._run_boltz(df_chai)
            df_vina = self._run_vina(df_boltz)
        log_section("Finished docking")

        #save_dataframe(self.state["df"], self.output_dir / "final_output.pkl")

    def _catalytic_residue_prediction(self):
        logger.info("Predicting active site residues")
        df_cat_res = self.df << ActiveSitePred('Entry', 'Sequence', self.squidly_dir, self.num_threads)
        df_squidly = pd.merge(self.df, df_cat_res, left_on='Entry', right_on='label', how='inner')
        output_path = os.path.join(self.output_dir, 'squidly.pkl')
        df_squidly.to_pickle(output_path)
        logger.info("Finished predicting active site residues")
        return df_squidly

    def _run_chai(self, df_squidly):
        logger.info("Docking using Chai")
        chai_dir = os.path.join(self.output_dir, 'chai/')
        chai_dir.mkdir(exist_ok=True, parents=True)
        df_squidly.loc[:, 'substrate'] = self.ligand_smiles
        df_chai = df_squidly << (Chai('Entry', 'Sequence', 'substrate', chai_dir, self.num_threads) >> Save('chai.pkl'))
        logger.info("Finished docking using Chai")
        return df_chai

    def _run_boltz(self, df_chai):
        logger.info("Docking using Boltz")
        boltz_dir = os.path.join(self.output_dir, 'boltz/')
        boltz_dir.mkdir(exist_ok=True, parents=True)
        df_chai['Intermediate'] = None
        df_boltz = df_chai << (Boltz('Entry', 'Sequence', 'substrate', 'Intermediate', boltz_dir, self.num_threads) >> Save('boltz.pkl'))
        logger.info("Finished docking using Boltz2")

    def _run_vina(self, df_boltz):
        logger.info("Docking using Vina")
        vina_dir = os.path.join(self.output_dir, 'vina/')
        vina_dir.mkdir(exist_ok=True, parents=True)
        df_boltz['structure'] = None # or path to AF structure
        df_boltz['substrate_name'] = self.ligand_name
        df_vina = df_boltz << (Vina('Entry', 'structure', 'Sequence', 'substrate', 'substrate_name', 'Squidly_CR_Position', vina_dir, self.num_threads) >> Save('vina.pkl'))
        logger.info("Finished docking using Vina")

        return df_vina
   

class Superimposition:()





class GeometricFilters:()



'''
    def _run_folding(self):
        logger.info("Running Folding step")
        #folding = Folding()
        self.state = folding.run(self.state)

    def _run_docking(self):
        logger.info("Running Docking step")
        #docking = Docking()
        self.state = docking.run(self.state)

    def _run_superimposition(self):
        logger.info("Running Superimposition step")
        #sup = Superimposition()
        self.state = sup.run(self.state)

    def _run_filtering(self):
        logger.info("Running Geometric Filtering step")
        #filter_step = GeometricFiltering()
        self.state = filter_step.run(self.state)
'''