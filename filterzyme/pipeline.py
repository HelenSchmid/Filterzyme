from pathlib import Path
from typing import Union
import pandas as pd
import logging
import os

from filterzyme.utils.helpers import log_section, log_subsection, log_boxed_note, generate_boltz_structure_path, generate_chai_structure_path
from filterzyme.utils.helpers import clean_protein_sequence, delete_empty_subdirs, add_metrics_to_best_structures, valid_file_list
from filterzyme.steps.predict_catalyticsite_step import ActiveSitePred
from filterzyme.steps.save_step import Save
from filterzyme.steps.dock_vina_step import Vina
from filterzyme.steps.extract_docking_metrics_step import DockingMetrics
from filterzyme.steps.preparevina_step import PrepareVina
from filterzyme.steps.preparechai_step import PrepareChai
from filterzyme.steps.prepareboltz_step import PrepareBoltz
from filterzyme.steps.superimposestructures_step import SuperimposeStructures
from filterzyme.steps.computeproteinRMSD_step import ProteinRMSD
from filterzyme.steps.computeligandRMSD_step import LigandRMSD
from filterzyme.steps.geometric_filtering_cofactor import GeneralGeometricFiltering
from filterzyme.steps.geometric_filtering_esterase import EsteraseGeometricFiltering
from filterzyme.steps.fpocket_step import Fpocket
from filterzyme.steps.ligandSASA_step import LigandSASA
from filterzyme.steps.plip_step import PLIP

from enzymetk.dock_chai_step import Chai
from enzymetk.dock_boltz_step import Boltz
#from enzymetk.dock_vina_step import Vina

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

class Docking:
    def __init__(
        self,
        df: pd.DataFrame,
        output_dir: Union[str, Path] = "pipeline_output",
        squidly_dir: Union[str, Path] = '',
        metagenomic_enzymes = 0,
        skip_catalytic_residue_prediction = False,
        alternative_structure_for_vina = 'Chai',
        num_threads = 1
    ):
        self.df = df.copy()
        self.squidly_dir = Path(squidly_dir) 
        self.skip_catalytic_residue_prediction = skip_catalytic_residue_prediction
        self.alternative_structure_for_vina = alternative_structure_for_vina
        self.num_threads = num_threads
        self.output_dir = Path(output_dir)
        self.metagenomic_enzymes = metagenomic_enzymes
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def run(self):
        
        if self.skip_catalytic_residue_prediction:
            log_section("Skipping catalytic residue prediction")
            df_squidly = self.df
        else:
            log_section("Predicting active site residues")
            df_squidly = self._catalytic_residue_prediction()

        log_section("Protein-Ligand docking")
        df_chai = self._run_chai(df_squidly)
        df_boltz= self._run_boltz(df_chai)
        df_vina = self._run_vina(df_boltz)
        df_metrics = self._extract_docking_quality_metrics(df_vina)

    def _catalytic_residue_prediction(self):
        self.df['Sequence'] = self.df['Sequence'].apply(clean_protein_sequence)

        reps  = (self.df[['Entry','Sequence']]
            .drop_duplicates(subset='Sequence', keep='first')
            .rename(columns={'Entry': 'rep_entry'}))
        pred_in = reps.rename(columns={'rep_entry': 'Entry'})
        
        df_cat_res = pred_in << ActiveSitePred('Entry', 'Sequence', self.squidly_dir, self.num_threads)
        df_cat_res = df_cat_res.merge(reps, left_on='label', right_on='rep_entry', how='left')

        df_squidly = self.df.merge(
        df_cat_res[['Sequence', 'Squidly_CR_Position']].drop_duplicates('Sequence'),
        on='Sequence', how='left' )

        # Remove entries without catalytic residues for proteins without user-specified residues for vina-docking
        if 'vina_residues' not in df_squidly.columns:
            df_squidly['vina_residues'] = None

        # Ensure both columns exist and normalize to clean strings
        for col in ['Squidly_CR_Position', 'vina_residues']:
            if col not in df_squidly.columns:
                df_squidly[col] = ''
            # Convert list-like to pipe-delimited; keep strings as-is; clean up None/NaN/whitespace
            def _norm(v):
                if v is None:
                    return ''
                if isinstance(v, (list, tuple)):
                    return '|'.join(str(x).strip() for x in v if str(x).strip() != '')
                s = str(v).strip()
                return '' if s.lower() in ('nan', 'none', '[]') else s
            df_squidly[col] = df_squidly[col].apply(_norm)

        # Remove entries with *no* catalytic residues from either source
        mask_empty = (df_squidly['Squidly_CR_Position'] == '') & (df_squidly['vina_residues'] == '')
        empty_entries = df_squidly.loc[mask_empty, 'Entry'].tolist()
        if empty_entries:
            log_boxed_note(
                'Removing entries without catalytic residues and without specified residues for vina docking: '
                + ', '.join(empty_entries)
            )
        df_squidly = df_squidly[~mask_empty].reset_index(drop=True)

        # Prefer user-specified vina residues when provided; else use Squidly
        use_vina = df_squidly['vina_residues'] != ''
        df_squidly['catalytic_residues'] = df_squidly['vina_residues'].where(use_vina, df_squidly['Squidly_CR_Position'])

        output_path = os.path.join(self.output_dir, 'squidly.pkl')
        df_squidly.to_pickle(output_path)
        log_boxed_note("Finished predicting active site residues")
        return df_squidly

    def _run_chai(self, df_squidly):
        log_subsection("Docking using Chai")
        chai_dir = Path(self.output_dir) / 'chai'
        chai_dir.mkdir(exist_ok=True, parents=True)
        if 'cofactor_smiles' not in df_squidly.columns:
            df_squidly['cofactor_smiles'] = ''
        df_chai = df_squidly << (Chai('Entry', 'Sequence', 'substrate_smiles', 'cofactor_smiles', chai_dir, self.num_threads) >> Save(Path(self.output_dir)/'chai.pkl'))
        df_chai.rename(columns = {'output_dir':'chai_dir'}, inplace=True)
        return df_chai

    def _run_boltz(self, df_chai):
        log_subsection("Docking using Boltz")
        boltz_dir = Path(self.output_dir) / 'boltz/'
        boltz_dir.mkdir(exist_ok=True, parents=True)
        if 'cofactor_smiles' not in df_chai.columns:
            df_chai['cofactor_smiles'] = None
        df_boltz = df_chai << (Boltz('Entry', 'Sequence', 'substrate_smiles', 'cofactor_smiles', boltz_dir, self.num_threads) 
                            >> Save(Path(self.output_dir)/'boltz.pkl'))
        df_boltz.rename(columns = {'output_dir':'boltz_dir'}, inplace=True)
        return df_boltz

    def _run_vina(self, df_boltz):
        log_subsection("Docking using Vina")
        vina_dir = Path(self.output_dir) / 'vina/'
        vina_dir.mkdir(exist_ok=True, parents=True)
        delete_empty_subdirs(vina_dir)

        if self.metagenomic_enzymes == 1:
            if self.alternative_strucuture_for_vina == 'Chai':
                log_boxed_note('Fallback to Chai structures for docking due to missing AF2 structures.' )    
                df_boltz['structure'] = df_boltz['chai_dir'].apply(generate_chai_structure_path)
            elif self.alternative_strucuture_for_vina == 'Boltz':
                log_boxed_note('Fallback to Boltz structures for docking due to missing AF2 structures.' )    
                df_boltz['structure'] = df_boltz['boltz_dir'].apply(generate_boltz_structure_path)

        else: 
            df_boltz['structure'] = None # or path to AF structure
        
        # Initial Vina docking attempt
        df_vina = df_boltz << (Vina('Entry', 'structure', 'Sequence', 'substrate_smiles', 'substrate_name', 'catalytic_residues', vina_dir, self.num_threads))
        df_vina.rename(columns = {'output_dir':'vina_dir'}, inplace=True)

        # Handle missing AF2 structures
        if df_vina['vina_dir'].isnull().any() == True: 
            missing_entries  = df_vina[df_vina['vina_dir'].isnull()]['Entry'].unique()
            delete_empty_subdirs(vina_dir)    

            # Prepare missing entries with Chai structure
            if self.alternative_structure_for_vina == 'Chai':
                log_boxed_note('Fallback to Chai structures for docking due to missing AF2 structures.' + f'Entries: {list(missing_entries)}')    
                df_missing = df_vina[df_vina['vina_dir'].isnull()].copy()
                df_missing['structure'] = df_missing['chai_dir'].apply(generate_chai_structure_path)  
                print(df_missing.structure)              


            # Prepare missing entries with Boltz structure
            elif self.alternative_structure_for_vina == 'Boltz':
                log_boxed_note('Fallback to Boltz structures for docking due to missing AF2 structures.' + f'Entries: {list(missing_entries)}')    
                df_missing = df_vina[df_vina['vina_dir'].isnull()].copy()
                df_missing['structure'] = df_missing['boltz_dir'].apply(generate_boltz_structure_path)

            df_missing_docked = df_missing << (Vina('Entry', 'structure', 'Sequence', 'substrate_smiles', 'substrate_name', 'catalytic_residues', vina_dir, self.num_threads))
            df_missing_docked.rename(columns = {'output_dir':'vina_dir_missing'}, inplace=True)

            # Merge both docking attempts
            df_vina_combined = pd.merge(
                df_vina, df_missing_docked[['Entry', 'vina_dir_missing']], on='Entry', how='left'
            )

            # Choose first attempt if available, otherwise fallback
            df_vina_combined['vina_dir'] = df_vina_combined.apply(
                lambda row: row['vina_dir'] if pd.notnull(row['vina_dir']) else row['vina_dir_missing'],
                axis=1
            )
            df_vina_combined.drop(columns=['vina_dir_missing'], inplace=True)
            df_vina = df_vina_combined.copy()
  
        df_vina.to_pickle(Path(self.output_dir)/'vina.pkl') 
        return df_vina
   
    def _extract_docking_quality_metrics(self, df):
        log_subsection('Extracting docking quality metrics')
        df = df[df["vina_dir"].notna()].copy()
        df_metrics = df << (DockingMetrics(input_dir = Path(self.output_dir), output_dir = Path(self.output_dir)) 
                        >> Save(Path(self.output_dir) / 'dockingmetrics.pkl'))
        return df_metrics


class Superimposition:
    def __init__(self, maxMatches, input_dir="pipeline_output", output_dir="pipeline_output", num_threads=1):
        self.maxMatches = maxMatches
        self.num_threads = num_threads
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def run(self):
        
        log_section('Superimposition')
        log_subsection('Superimposing docked structures')
        df_prep = self._prepare_files_for_superimposition()
        df_sup = self._superimposition(df_prep)
        log_subsection('Calculating protein RMSDs')
        df_proteinRMSD = self._proteinRMSD(df_sup)
        log_subsection('Calculating ligand RMSDs')
        df_ligandRMSD = self._ligandRMSD(df_proteinRMSD)
        return df_ligandRMSD


    def _prepare_files_for_superimposition(self):
        df_metrics = pd.read_pickle(Path(self.input_dir) / 'dockingmetrics.pkl')
        preparedfiles_dir = Path(self.output_dir) / 'preparedfiles_for_superimposition/'
        df_metrics << (PrepareVina('vina_dir', 'substrate_name',  preparedfiles_dir)
                >> PrepareChai('chai_dir', preparedfiles_dir, 1)
                >> PrepareBoltz('boltz_dir' , preparedfiles_dir, 1))
        return df_metrics

    def _superimposition(self,  df):                   
        output_sup_dir = Path(self.output_dir) / 'superimposed_structures'

        df = df[df['vina_files_for_superimposition'].apply(valid_file_list)]
        df = df[df['chai_files_for_superimposition'].apply(valid_file_list)]

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
        input_dir = Path(self.output_dir)  / 'superimposed_structures'
        df_best_structures = df << (LigandRMSD('Entry', input_dir = input_dir, output_dir = ligandRMSD_dir, visualize_heatmaps= True, maxMatches = self.maxMatches))
        df_best_structures_w_metrics = add_metrics_to_best_structures(df_best_structures, pd.read_pickle(Path(self.output_dir).parent / 'docking/dockingmetrics.pkl'))
        df_best_structures_w_metrics.to_pickle(Path(self.output_dir) / 'best_structures.pkl')
        return df_best_structures_w_metrics


class GeometricFilters:
    def __init__(self,  df, esterase = 0, input_dir="superimposition", output_dir="geometricfiltering", num_threads=1):
        self.esterase = esterase
        self.num_threads = num_threads
        self.df = df.copy()
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)


    def run(self):
        
        log_section('Running geometric filtering')
        log_subsection('Calculate catalytic residue/cofactor - ligand distances')
        df_filter = self._run_geometric_filtering()
        log_subsection('Calculate active site volume')
        df_ASvolume = self._active_site_volume(df_filter)
        log_subsection('Calculate ligand surface exposure')
        df_ASvolume = self._ligand_surface_exposure(df_ASvolume)
        log_subsection('Running PLIP to identify protein-ligand interactions')
        df_final = self._plip_interactions(df_ASvolume)

        out_final = Path(self.output_dir) / 'structural_features_final.pkl'
        df_final.to_pickle(out_final)

        log_boxed_note('Pipeline finished!')
        return df_final

    def _run_geometric_filtering(self):
        if self.esterase == 1: 
            df_geo_filter = self.df << (EsteraseGeometricFiltering(
                                            preparedfiles_dir=Path(self.input_dir) / 'preparedfiles_for_superimposition',
                                            output_dir=self.output_dir)
                                    >> Save(Path(self.output_dir) / 'geometricfiltering.pkl'))
        else: 
            df_geo_filter = self.df << (GeneralGeometricFiltering(
                                            preparedfiles_dir=Path(self.input_dir) / 'preparedfiles_for_superimposition',
                                            output_dir=self.output_dir)
                                    >> Save(Path(self.output_dir) / 'geometricfiltering.pkl'))
        return df_geo_filter

    def _active_site_volume(self, df):
        fpocket_dir = Path(self.output_dir) / 'ASVolume'
        fpocket_dir.mkdir(exist_ok=True, parents=True)
        df_ASVolume = df << (Fpocket(preparedfiles_dir=Path(self.input_dir) / 'preparedfiles_for_superimposition', output_dir = fpocket_dir)  
            >> Save(Path(self.output_dir) / 'ASvolume.pkl'))
        return df_ASVolume

    def _ligand_surface_exposure(self, df):
        ligandSASA_dir = Path(self.output_dir) / 'LigandSASA'
        df_ligandSASA = df << (LigandSASA(input_dir = Path(self.input_dir)/ 'preparedfiles_for_superimposition', output_dir = ligandSASA_dir)
                            >> Save(Path(self.output_dir) / 'ligandSASA.pkl'))
        return df_ligandSASA
    
    def _plip_interactions(self, df):
        df_plip = df << (PLIP(input_dir = Path(self.input_dir) / 'preparedfiles_for_superimposition', output_dir = self.output_dir)
                        >> Save(Path(self.output_dir) / 'plip_interactions.pkl'))
        return df_plip


class Pipeline:
    """Full pipeline"""
    def __init__(self,
                df, 
                max_matches: int = 1000,
                esterase: int = 0,
                metagenomic_enzymes: int = 0,
                skip_catalytic_residue_prediction: bool = False,
                alternative_structure_for_vina: str = 'Boltz', 
                num_threads: int = 1,
                squidly_dir: Union[str, Path] = '',
                base_output_dir: Union[str, Path] = "pipeline_output"
                ):
                 
        self.df = df.copy()
        self.max_matches = max_matches
        self.esterase = esterase
        self.metagenomic_enzymes = metagenomic_enzymes
        self.skip_catalytic_residue_prediction = skip_catalytic_residue_prediction
        self.alternative_structure_for_vina = alternative_structure_for_vina
        self.num_threads = num_threads
        self.squidly_dir = squidly_dir
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True, parents=True)

    def run(self):
        # Docking
        docking = Docking(
            df=self.df,
            output_dir= Path(self.base_output_dir) / "docking",
            squidly_dir=Path(self.squidly_dir),
            metagenomic_enzymes= self.metagenomic_enzymes,
            skip_catalytic_residue_prediction = self.skip_catalytic_residue_prediction,
            alternative_structure_for_vina = self.alternative_structure_for_vina, 
            num_threads=self.num_threads,
        )
        docking.run()

        # Superimposition
        superimp = Superimposition(
            maxMatches=self.max_matches,
            input_dir=Path(self.base_output_dir) / "docking",
            output_dir=Path(self.base_output_dir) / "superimposition",
            num_threads=self.num_threads,
        )
        superimp.run()  



        # Geometric filtering for all structures

        # Geometric filtering for best structure only
        gf = GeometricFilters(
            df = pd.read_pickle(Path(self.base_output_dir) / 'superimposition/best_structures.pkl'),
            esterase=self.esterase,
            input_dir=Path(self.base_output_dir) / "superimposition",
            output_dir=Path(self.base_output_dir) / "geometricfiltering",
            num_threads=self.num_threads,
        )
        gf.run()

