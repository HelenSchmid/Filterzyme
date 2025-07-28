from pathlib import Path
from typing import Union
import pandas as pd
import logging
import os

from filtering_pipeline.utils.helpers import log_section, log_subsection
from filtering_pipeline.steps.predict_catalyticsite_step import ActiveSitePred
from filtering_pipeline.steps.save_step import Save
from filtering_pipeline.steps.preparevina_step import PrepareVina
from filtering_pipeline.steps.preparechai_step import PrepareChai
from filtering_pipeline.steps.prepareboltz_step import PrepareBoltz
from filtering_pipeline.steps.superimposestructures_step import SuperimposeStructures
from filtering_pipeline.steps.computeproteinRMSD_step import ProteinRMSD
from filtering_pipeline.steps.computeligandRMSD_step import LigandRMSD
from filtering_pipeline.steps.geometric_filtering import GeometricFiltering
from filtering_pipeline.steps.fpocket_step import Fpocket

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
        self.squidly_dir = Path(squidly_dir) 
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
        log_section("Protein-Ligand docking")
        df_chai = self._run_chai(df_squidly)
        df_boltz= self._run_boltz(df_chai)
        df_vina = self._run_vina(df_boltz)
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
        df_boltz = df_chai << (Boltz('Entry', 'Sequence', 'substrate', 'Intermediate', boltz_dir, self.num_threads) 
                            >> Save(Path(self.output_dir)/'boltz.pkl'))
        df_boltz.rename(columns = {'output_dir':'boltz_dir'}, inplace=True)
        return df_boltz

    def _run_vina(self, df_boltz):
        log_subsection("Docking using Vina")
        vina_dir = Path(self.output_dir) / 'vina/' 
        vina_dir.mkdir(exist_ok=True, parents=True)
        df_boltz['structure'] = None # or path to AF structure
        df_boltz['substrate_name'] = self.ligand_name
        df_vina = df_boltz << (Vina('Entry', 'structure', 'Sequence', 'substrate', 'substrate_name', 'Squidly_CR_Position', vina_dir, self.num_threads) 
                            >> Save(Path(self.output_dir)/'vina.pkl'))
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
        
        log_section('Superimposition')
        log_subsection('Extracting docking quality metrics')
        #self._extract_docking_metrics()
        log_subsection('Superimposing docked structures')
        df_sup = self._prepare_files_for_superimposition()
        self._superimposition(df_sup)
        log_subsection('Calculating protein RMSDs')
        df_proteinRMSD = self._proteinRMSD(df_sup)
        log_subsection('Calculating ligand RMSDs')
        df_ligandRMSD = self._ligandRMSD(df_proteinRMSD)
        return df_ligandRMSD


    def _extract_docking_metrics(self):
        # TODO: process your vina/chai pickle files
        pass


    def _prepare_files_for_superimposition(self):
        df_vina = pd.read_pickle(Path(self.input_dir) / 'vina.pkl')
        preparedfiles_dir = Path(self.output_dir) / 'preparedfiles_for_superimposition/'

        df_vina << (PrepareVina('output_dir', self.ligand_name,  preparedfiles_dir) 
                >> PrepareChai('chai_dir', preparedfiles_dir, 1) 
                >> PrepareBoltz('boltz_dir' , preparedfiles_dir, 1))
        return df_vina

    def _superimposition(self,  df):                   
        output_sup_dir = Path(self.output_dir) / 'superimposed_structures'
        #f = pd.read_pickle('/nvme2/helen/EnzymeStructuralFiltering/superimposition_test/df_for_superimposition')
        df_sup = df << (SuperimposeStructures('vina_files_for_superimposition',  'chai_files_for_superimposition',  output_dir = output_sup_dir, name1='vina', name2='chai', num_threads = self.num_threads) 
                >> SuperimposeStructures('vina_files_for_superimposition',  'boltz_files_for_superimposition',  output_dir = output_sup_dir, name1='vina', name2='boltz', num_threads = self.num_threads) 
                >> SuperimposeStructures('chai_files_for_superimposition',  'boltz_files_for_superimposition',  output_dir = output_sup_dir, name1='chai', name2='boltz', num_threads = self.num_threads)
                >> Save(Path(self.output_dir) / 'superimposedstructures.pkl'))
        return df_sup
    
    def _proteinRMSD(self, df):  
        proteinRMSD_dir = Path(self.output_dir) / 'proteinRMSD'
        proteinRMSD_dir.mkdir(exist_ok=True, parents=True) 
        input_dir = Path(self.output_dir) / 'superimposed_structures'
        df_proteinRMSD = df << (ProteinRMSD('Entry', input_dir = input_dir, output_dir = proteinRMSD_dir, visualize_heatmaps = True)  
                            >> Save(Path(self.output_dir)/'proteinRMSD.pkl'))
        return df_proteinRMSD

    def _ligandRMSD(self, df): 
        ligandRMSD_dir = Path(self.output_dir) / 'ligandRMSD'
        ligandRMSD_dir.mkdir(exist_ok=True, parents=True) 
        input_dir = Path(self.output_dir) / 'superimposed_structures'
        df_ligandRMSD = df << (LigandRMSD('Entry', input_dir = input_dir, output_dir = ligandRMSD_dir, visualize_heatmaps= True, maxMatches = self.maxMatches)  
                            >> Save(Path(self.output_dir)/'ligandRMSD.pkl'))
        return df_ligandRMSD

class GeometricFilters:
    def __init__(self, substrate_smiles: str, smarts_pattern: str, df, esterase = 0, find_closest_nuc = 0, input_dir="superimposition", output_dir="geometricfiltering", num_threads=1):
        self.smarts_pattern = smarts_pattern
        self.substrate_smiles = substrate_smiles
        self.esterase = esterase
        self.find_closest_nuc = find_closest_nuc
        self.num_threads = num_threads
        self.df = df.copy()
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)


    def run(self):
        
        log_section('Running geometric filtering')
        log_subsection('Calculate catalytic residue - ligand distances')
        df_filter = self._run_geometric_filtering()
        log_subsection('Calculate active site volume')
        self._active_site_volume(df_filter)

    def _run_geometric_filtering(self):
        df_geo_filter = self.df << (GeometricFiltering(
                                        ligand_smiles=self.substrate_smiles,
                                        smarts_pattern=self.smarts_pattern,
                                        preparedfiles_dir=Path(self.input_dir) / 'preparedfiles_for_superimposition',
                                        output_dir=self.output_dir,
                                        esterase=self.esterase,
                                        find_closest_nucleophile=self.find_closest_nuc
                                    )
                                >> Save(Path(self.output_dir) / 'geometricfiltering.pkl'))
        return df_geo_filter

    def _active_site_volume(self, df):
        fpocket_dir = Path(self.output_dir) / 'ASVolume'
        fpocket_dir.mkdir(exist_ok=True, parents=True)
        df << (Fpocket(preparedfiles_dir=Path(self.input_dir) / 'preparedfiles_for_superimposition', output_dir = fpocket_dir)  
            >> Save(Path(self.output_dir) / 'ASVolume.pkl'))

    def _ligand_surface_exposure(self):
        # TODO: Implement step
        pass


class Pipeline:
    def __init__(self,
                 df, 
                 ligand_name: str,
                 ligand_smiles: str,
                 smarts_pattern: str,
                 max_matches: int = 1000,
                 esterase: int = 0,
                 find_closest_nuc: int = 0,
                 num_threads: int = 1,
                 squidly_dir: Union[str, Path] = '',
                 base_output_dir: Union[str, Path] = "pipeline_output"
                ):
                 
        self.df = df.copy()
        self.ligand_name = ligand_name
        self.ligand_smiles = ligand_smiles
        self.smarts_pattern = smarts_pattern
        self.max_matches = max_matches
        self.esterase = esterase
        self.find_closest_nuc = find_closest_nuc
        self.num_threads = num_threads
        self.squidly_dir = squidly_dir
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True, parents=True)

    def run(self):
        # Docking
        docking = Docking(
            ligand_name=self.ligand_name,
            ligand_smiles=self.ligand_smiles,
            df=self.df,
            output_dir= Path(self.base_output_dir) / "docking",
            squidly_dir=Path(self.squidly_dir),
            num_threads=self.num_threads,
        )
        docking.run()

        # Superimposition
        superimp = Superimposition(
            ligand_name=self.ligand_name,
            maxMatches=self.max_matches,
            input_dir=Path(self.base_output_dir) / "docking",
            output_dir=Path(self.base_output_dir) / "superimposition",
            num_threads=self.num_threads,
        )
        superimp.run()  

        # Geometric filtering
        gf = GeometricFilters(
            substrate_smiles=self.ligand_smiles,
            smarts_pattern=self.smarts_pattern,
            df = pd.read_csv(Path(self.base_output_dir) / 'superimposition/ligandRMSD/best_docked_structures.csv'),
            esterase=self.esterase,
            find_closest_nuc=self.find_closest_nuc,
            input_dir=Path(self.base_output_dir) / "superimposition",
            output_dir=Path(self.base_output_dir) / "geometricfiltering",
            num_threads=self.num_threads,
        )
        gf.run()

