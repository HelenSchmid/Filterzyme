from pathlib import Path
from typing import Union
import pandas as pd
import logging
import os

import pandas as pd

from filtering_pipeline.utils.helpers import log_section, log_subsection
from filtering_pipeline.utils.helpers import prepare_files_for_superimposition
from filtering_pipeline.steps.predict_catalyticsite_step import ActiveSitePred
from filtering_pipeline.steps.save_step import Save
from filtering_pipeline.steps.superimposestructures_step import SuperimposeStructures
from filtering_pipeline.steps.computeproteinRMSD_step import ProteinRMSD
from filtering_pipeline.steps.computeligandRMSD_step import LigandRMSD
from filtering_pipeline.steps.geometric_filtering import GeometricFiltering


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
        num_threads = 1
    ):
        self.ligand_name = ligand_name
        self.ligand_smiles = ligand_smiles
        self.df = df.copy()
        self.squidly_dir = squidly_dir
        self.num_threads = num_threads
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # store intermediate results
        self.state = {
            "df": self.df,
            "ligand_smiles": self.ligand_smiles,
            "output_dir": self.output_dir
        }

    def run(self):
        
        log_section("Predicting active site residues")
        df_squidly = self._catalytic_residue_prediction()
        log_section("Start docking")
        df_chai = self._run_chai(df_squidly)
        df_boltz= self._run_boltz(df_chai)
        df_vina = self._run_vina(df_boltz)
        log_section("Finished docking")
        print(df_vina)

        #save_dataframe(self.state["df"], self.output_dir / "final_output.pkl")

    def _catalytic_residue_prediction(self):
        df_cat_res = self.df << ActiveSitePred('Entry', 'Sequence', self.squidly_dir, self.num_threads)
        df_squidly = pd.merge(self.df, df_cat_res, left_on='Entry', right_on='label', how='inner')
        output_path = os.path.join(self.output_dir, 'squidly.pkl')
        df_squidly.to_pickle(output_path)
        logger.info("Finished predicting active site residues")
        return df_squidly

    def _run_chai(self, df_squidly):
        log_subsection("Docking using Chai")
        chai_dir = Path(self.output_dir) / 'chai'
        chai_dir.mkdir(exist_ok=True, parents=True)
        df_squidly.loc[:, 'substrate'] = self.ligand_smiles
        df_chai = df_squidly << (Chai('Entry', 'Sequence', 'substrate', chai_dir, self.num_threads) >> Save(Path(self.output_dir)/'chai.pkl'))
        df_chai.rename(columns = {'output_dir':'chai_dir'}, inplace=True)
        return df_chai

    def _run_boltz(self, df_chai):
        log_subsection("Docking using Boltz")
        boltz_dir = Path(self.output_dir) / 'boltz/'
        boltz_dir.mkdir(exist_ok=True, parents=True)
        df_chai['Intermediate'] = None
        df_boltz = df_chai << (Boltz('Entry', 'Sequence', 'substrate', 'Intermediate', boltz_dir, self.num_threads) >> Save(Path(self.output_dir)/'boltz.pkl'))
        df_boltz.rename(columns = {'output_dir':'boltz_dir'}, inplace=True)
        return df_boltz

    def _run_vina(self, df_boltz):
        log_subsection("Docking using Vina")
        vina_dir = Path(self.output_dir) /'vina/'
        vina_dir.mkdir(exist_ok=True, parents=True)
        df_boltz['structure'] = None # or path to AF structure
        df_boltz['substrate_name'] = self.ligand_name
        df_vina = df_boltz << (Vina('Entry', 'structure', 'Sequence', 'substrate', 'substrate_name', 'Squidly_CR_Position', vina_dir, self.num_threads) >> Save(Path(self.output_dir)/'vina.pkl'))
        df_vina.rename(columns = {'output_dir':'vina_dir'}, inplace=True)
        return df_vina
   

class Superimposition:
    def __init__(self, ligand_name: str, maxMatches, input_dir="pipeline_output", output_dir="pipeline_output", num_threads=1):
        self.ligand_name = ligand_name
        self.maxMatches = maxMatches
        self.num_threads = num_threads
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)


    def run(self):
        
        log_subsection('Extracting docking quality metrics')
        self._extract_docking_metrics()
        df_super = log_subsection('Superimposing docked structures')
        self._superimposition()
        log_subsection('Calculating protein and ligand RMSDs')
        df_proteinRMSD = self._proteinRMSD(df_super)
        self._ligandRMSD(df_proteinRMSD)


    def _extract_docking_metrics(self):
        # TODO: process your vina/chai pickle files
        pass

    def _superimposition(self):              
        prepare_files_for_superimposition(
        df_vina=Path(self.input_dir) / 'vina.pkl',
        df_chai=Path(self.input_dir) / 'chai.pkl',
        ligand_name=self.ligand_name,
        output_dir=Path(self.output_dir) / 'preparedfiles_for_superimposition'
        )
        
        output_sup_dir = Path(self.output_dir) / 'superimposed_structures'
        df = pd.read_pickle('/nvme2/helen/EnzymeStructuralFiltering/superimposition_test/df_for_superimposition')
        df_sup = df << (SuperimposeStructures('vina_files_for_superimposition',  'chai_files_for_superimposition',  output_dir = output_sup_dir, name1='vina', name2='chai', num_threads = self.num_threads) 
                >> SuperimposeStructures('vina_files_for_superimposition',  'boltz_files_for_superimposition',  output_dir = output_sup_dir, name1='vina', name2='boltz', num_threads = self.num_threads) 
                >> SuperimposeStructures('chai_files_for_superimposition',  'boltz_files_for_superimposition',  output_dir = output_sup_dir, name1='chai', name2='boltz', num_threads = self.num_threads)
                >> Save(Path(self.output_dir) / 'superimposedstructures.pkl'))
        return df_sup
    
    def _proteinRMSD(self, df):  
        proteinRMSD_dir = Path(self.output_dir) / 'proteinRMSD'
        proteinRMSD_dir.mkdir(exist_ok=True, parents=True) 
        input_dir = output_dir=Path(self.output_dir) / 'preparedfiles_for_superimposition'
        df_proteinRMSD = df << (ProteinRMSD('Entry', input_dir = input_dir, output_dir = proteinRMSD_dir, visualize_heatmaps = True)  >> Save(Path(self.output_dir)/'proteinRMSD.pkl'))
        return df_proteinRMSD

    def _ligandRMSD(self, df): 
        ligandRMSD_dir = Path(self.output_dir) / 'ligandRMSD'
        ligandRMSD_dir.mkdir(exist_ok=True, parents=True) 
        df << (LigandRMSD('Entry', input_dir = self.output_dir, output_dir = ligandRMSD_dir, visualize_heatmaps= True, maxMatches = self.maxMatches)  >> Save(Path(self.output_dir)/'ligandrmsd.pkl'))





class GeometricFilters:
    def __init__(self, substrate_smiles: str, smarts_pattern: str, df, input_dir="preparedfiles_for_superimposition", output_dir="geometricfiltering", num_threads=1):
        self.smarts_pattern = smarts_pattern
        self.substrate_smiles = substrate_smiles
        self.num_threads = num_threads
        self.df = df.copy()
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)


    def run(self):
        
        log_section('Running geometric filtering')
        log_subsection('Calculate catalytic residue - ligand distances')
        self._run_geometric_filtering()
        log_subsection('Calculate active site volume')


    def _run_geometric_filtering(self):
        self.df << (GeometricFiltering(self.substrate_smiles, self.smarts_pattern, self.input_dir, self.output_dir)  >> Save(Path(self.output_dir) / 'geometricfiltering.pkl'))

    #def _active_site_volume(self):






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