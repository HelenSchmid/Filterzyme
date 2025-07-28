import pandas as pd
from tempfile import TemporaryDirectory
from pathlib import Path
import logging
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
import os
import subprocess
import shutil
import uuid
import re

from filtering_pipeline.steps.step import Step

# How to run fpocket in terminal: fpocket -f /home/helen/cec_degrader/generalize/alphafold_structures/A1RRK1_structure.pdb
# -r string: (None) This parameter allows you to run fpocket in a restricted mode. Let's suppose you have a very shallow or large pocket with a ligand inside and the automatic pocket prediction always splits up you pocket or you have only a part of the pocket found. Specifying your ligand residue with -r allows you to detect and characterize you ligand binding site explicitly. 
# For instance for `1UYD.pdb` you can specify `-r 1224:PU8:A` (residue number of the ligand: residue name of the ligand: chain of the ligand)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def extract_fpocket_features(fpocket_pdb_path: Path) -> dict:
    """
    Parses the HEADER section of an fpocket output PDB file
    to extract pocket features.
    """
    features = {
        "fpocket_Pocket_Score": None,
        "fpocket_Drug_Score": None,
        "fpocket_Num_alpha_spheres": None,
        "fpocket_Mean_alpha_sphere_radius": None,
        "fpocket_Mean_alpha_sphere_Solvent_Acc": None,
        "fpocket_Mean_B_factor": None,
        "fpocket_Hydrophobicity_Score": None,
        "fpocket_Polarity_Score": None,
        "fpocket_Amino_Acid_based_volume_Score": None,
        "fpocket_Pocket_volume_Monte_Carlo": None,
        "fpocket_Pocket_volume_convex_hull": None,
        "fpocket_Charge_Score": None,
        "fpocket_Local_hydrophobic_density_Score": None,
        "fpocket_Num_apolar_alpha_sphere": None,
        "fpocket_Proportion_apolar_alpha_sphere": None,
    }

    if not fpocket_pdb_path.exists():
        logger.warning(f"Pocket PDB file not found for parsing: {fpocket_pdb_path}")
        return features

    try:
        with open(fpocket_pdb_path, 'r') as f:
            for line in f:
                if line.startswith("HEADER"):
                    # Use regex to extract the score value
                    match_score = re.search(r"Pocket Score\s+:\s+([-?\d.]+)", line)
                    if match_score:
                        features["fpocket_Pocket_Score"] = float(match_score.group(1))

                    match_drug = re.search(r"Drug Score\s+:\s+([-?\d.]+)", line)
                    if match_drug:
                        features["fpocket_Drug_Score"] = float(match_drug.group(1))
                    
                    match_num_alpha = re.search(r"Number of alpha spheres\s+:\s+(-?\d+)", line)
                    if match_num_alpha:
                        features["fpocket_Num_alpha_spheres"] = int(match_num_alpha.group(1))

                    match_mean_rad = re.search(r"Mean alpha-sphere radius\s+:\s+([-?\d.]+)", line)
                    if match_mean_rad:
                        features["fpocket_Mean_alpha_sphere_radius"] = float(match_mean_rad.group(1))
                    
                    match_mean_solv_acc = re.search(r"Mean alpha-sphere Solvent Acc.\s+:\s+([-?\d.]+)", line)
                    if match_mean_solv_acc:
                        features["fpocket_Mean_alpha_sphere_Solvent_Acc"] = float(match_mean_solv_acc.group(1))

                    match_mean_bfactor = re.search(r"Mean B-factor of pocket residues\s+:\s+([-?\d.]+)", line)
                    if match_mean_bfactor:
                        features["fpocket_Mean_B_factor"] = float(match_mean_bfactor.group(1))

                    match_hydro = re.search(r"Hydrophobicity Score\s+:\s+([-?\d.]+)", line)
                    if match_hydro:
                        features["fpocket_Hydrophobicity_Score"] = float(match_hydro.group(1))

                    match_polar = re.search(r"Polarity Score\s+:\s+(-?\d+)", line)
                    if match_polar:
                        features["fpocket_Polarity_Score"] = int(match_polar.group(1))

                    match_aa_vol = re.search(r"Amino Acid based volume Score\s+:\s+([-?\d.]+)", line)
                    if match_aa_vol:
                        features["fpocket_Amino_Acid_based_volume_Score"] = float(match_aa_vol.group(1))

                    match_mc_vol = re.search(r"Pocket volume \(Monte Carlo\)\s+:\s+([-?\d.]+)", line)
                    if match_mc_vol:
                        features["fpocket_Pocket_volume_Monte_Carlo"] = float(match_mc_vol.group(1))

                    match_ch_vol = re.search(r"Pocket volume \(convex hull\)\s+:\s+([-?\d.]+)", line)
                    if match_ch_vol:
                        features["fpocket_Pocket_volume_convex_hull"] = float(match_ch_vol.group(1))
                    
                    match_charge = re.search(r"Charge Score\s+:\s+(-?\d+)", line)
                    if match_charge:
                        features["fpocket_Charge_Score"] = int(match_charge.group(1))

                    match_local_hydro = re.search(r"Local hydrophobic density Score\s+:\s+([-?\d.]+)", line)
                    if match_local_hydro:
                        features["fpocket_Local_hydrophobic_density_Score"] = float(match_local_hydro.group(1))

                    match_num_apolar = re.search(r"Number of apolar alpha sphere\s+:\s+(-?\d+)", line)
                    if match_num_apolar:
                        features["fpocket_Num_apolar_alpha_sphere"] = int(match_num_apolar.group(1))

                    match_prop_apolar = re.search(r"Proportion of apolar alpha sphere\s+:\s+([-?\d.]+)", line)
                    if match_prop_apolar:
                        features["fpocket_Proportion_apolar_alpha_sphere"] = float(match_prop_apolar.group(1))
                    
                elif line.startswith("ATOM"): # Stop reading headers once ATOM records start
                    break
    except Exception as e:
        logger.error(f"Error parsing fpocket output file {fpocket_pdb_path}: {e}")
        # Ensure all features remain None in case of parsing error
        return {k: None for k in features} 

    return features 

def extract_SASA(fpocket_txt_path: Path) -> dict: 
    """
    Parses the fpocket output txt file to extract SASA features.
    """
    sasa_values = {
        "total_sasa": None,
        "polar_sasa": None,
        "apolar_sasa": None
    }

    if not fpocket_txt_path.exists():
        print(f"Error: Fpocket txt output file not found at {fpocket_txt_path}")
        return sasa_values

    try:
        with open(fpocket_txt_path, 'r') as f:
            report_text = f.read() # Read the entire file content into a string
        
        # Regex patterns to capture the floating-point numbers
        total_sasa_match = re.search(r"Total SASA\s*:\s*([\d.]+)", report_text)
        if total_sasa_match:
            sasa_values["total_sasa"] = float(total_sasa_match.group(1))

        polar_sasa_match = re.search(r"Polar SASA\s*:\s*([\d.]+)", report_text)
        if polar_sasa_match:
            sasa_values["polar_sasa"] = float(polar_sasa_match.group(1))

        apolar_sasa_match = re.search(r"Apolar SASA\s*:\s*([\d.]+)", report_text)
        if apolar_sasa_match:
            sasa_values["apolar_sasa"] = float(apolar_sasa_match.group(1))

    except Exception as e:
        print(f"An error occurred while reading or parsing the file {fpocket_txt_path}: {e}")
        # Return initialized dictionary with Nones in case of error
        return {
            "total_sasa": None,
            "polar_sasa": None,
            "apolar_sasa": None,
        }

    return sasa_values



class Fpocket(Step):
    def __init__(self, preparedfiles_dir: str = 'filteringpipeline/preparedfiles', output_dir: str = '', num_threads: int = 1):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.preparedfiles_dir = Path(preparedfiles_dir)
        self.num_threads = num_threads

    # This function processes a SINGLE row and returns its result (a pd.Series)
    def _process_single_row_with_fpocket(self, row: pd.Series) -> pd.Series:
        best_structure_name = row['best_structure']
        pdb_file_path = self.preparedfiles_dir / f"{best_structure_name}.pdb"
        
        row_results = {"ASvolume_dir": None}

        if not pdb_file_path.exists():
            logger.error(f"PDB file not found for processing: {pdb_file_path}. Skipping.")
            return pd.Series(row_results, index=row_results.keys())

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            temp_pdb_path = temp_path / pdb_file_path.name
            
            try:
                shutil.copy(str(pdb_file_path), str(temp_pdb_path))
            except Exception as e:
                logger.error(f"Error copying PDB file {pdb_file_path} to {temp_pdb_path}: {e}")
                return pd.Series(row_results, index=row_results.keys())

            logger.info(f"Running fpocket on {temp_pdb_path.name}")
            
            # fpocket command
            fpocket_cmd = ["fpocket", "-f", str(temp_pdb_path), "-r", "1:LIG:B"]

            result = subprocess.run(
                fpocket_cmd,
                cwd=temp_path,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                logger.error(f"fpocket failed on {pdb_file_path.name} (from {temp_path}). STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
                return pd.Series(row_results, index=row_results.keys())

            expected_out_dir_in_temp = temp_path / f"{temp_pdb_path.stem}_out"
            if not expected_out_dir_in_temp.exists():
                logger.warning(f"fpocket output directory not found in temp dir after successful run for {pdb_file_path.stem}: {expected_out_dir_in_temp}")
                return pd.Series(row_results, index=row_results.keys())
            
            final_out_dir = self.output_dir / f"{pdb_file_path.stem}_fpocket_output"
            
            if final_out_dir.exists():
                logger.warning(f"Existing fpocket output for {pdb_file_path.stem} found at {final_out_dir}. Removing old fpocket output.")
                try:
                    shutil.rmtree(final_out_dir)
                except Exception as e:
                    logger.error(f"Failed to remove existing output directory {final_out_dir}: {e}. This might cause the move operation to fail. Returning initial None results.")
                    return pd.Series(row_results, index=row_results.keys())

            try:
                # Move the entire output folder from temp_dir to the final self.output_dir
                shutil.move(str(expected_out_dir_in_temp), str(final_out_dir))
                expected_log_file_in_temp = temp_path / f"{temp_pdb_path.stem}.log"
                if expected_log_file_in_temp.exists():
                    shutil.move(str(expected_log_file_in_temp), str(final_out_dir / expected_log_file_in_temp.name))
                
                row_results['ASvolume_dir'] = str(final_out_dir)

                # Parse fpocket output and add pocket features
                pocket_1_pdb_path = final_out_dir / 'pockets' /'pocket1_atm.pdb' 
                               
                if pocket_1_pdb_path.exists():
                    extracted_features = extract_fpocket_features(pocket_1_pdb_path)
                    row_results.update(extracted_features) # Add extracted features to results
                else:
                    logger.warning(f"Pocket 1 PDB file not found at {pocket_1_pdb_path}. ")

                # Extract SASA values from txt output file
                txt_file = list(final_out_dir.rglob('*.txt'))
                extracted_SASA = extract_SASA(txt_file[0])
                row_results.update(extracted_SASA) # Add extracted SASA features to results 
                
            except Exception as e:
                logger.error(f"Failed to move {expected_out_dir_in_temp} to {final_out_dir} or parse output for {pdb_file_path.name}: {e}")
                # Ensure results are still a Series with None for features if an error occurs here
                return pd.Series(row_results, index=row_results.keys()) 

        # Return the Series with fpocket_dir and all extracted features for this row
        return pd.Series(row_results, index=row_results.keys())


    def __execute(self, df: pd.DataFrame) -> pd.DataFrame:

        if not self.preparedfiles_dir.exists():
            raise FileNotFoundError(f"Prepared files directory does not exist: {self.preparedfiles_dir}")
        
        results_list = []
        if self.num_threads > 1:
            with ThreadPool(self.num_threads) as pool:
                # `pool.map` applies _process_single_row_with_fpocket to each row in parallel
                results_list = pool.map(self._process_single_row_with_fpocket, [row for _, row in df.iterrows()])
        else:
            for _, row in df.iterrows():
                results_list.append(self._process_single_row_with_fpocket(row))
        
        # Convert the list of series (results from each row) into a df
        results_df = pd.DataFrame(results_list, index=df.index)
        
        return results_df


    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.output_dir:
            logger.error('No output directory provided.')

        fpocket_results_df = self.__execute(df)
        
        df_out = pd.concat([df, fpocket_results_df], axis=1)
        
        return df_out
