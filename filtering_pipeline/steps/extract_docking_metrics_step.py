import os
import sys
import pandas as pd
from pathlib import Path
import numpy as np
import json

from filtering_pipeline.steps.step import Step

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)




def parse_vina_output(file_path):
    docking_results = {}

    with open(file_path, 'r') as f:
        lines = f.readlines()

    parsing = False
    for line in lines:
        line = line.strip()

        if line.startswith("mode |"):  # start of the table
            parsing = True
            continue

        if parsing:
            if not line or not line[0].isdigit():
                continue  # skip headers or footers

            parts = line.split()
            try:
                mode = int(parts[0])
                affinity = float(parts[1])
                docking_results[mode] = affinity
            except Exception as e:
                logger.warning(f"Could not parse line: '{line}' — {e}")
                continue
    return docking_results


def extract_docking_metrics_vina(vina_dir: Path, entry_name: str, ligand_name: str):
    """
    Extracts binding affinity from a Vina output .txt file named like <entry>-<ligand>.txt
    """
    txt_file = vina_dir / f"{entry_name}-{ligand_name}_log.txt"

    if not txt_file.exists():
        logger.warning(f"Vina output not found: {txt_file}")
        return None

    vina_dict = parse_vina_output(txt_file)
    return vina_dict


def extract_chai_metrics(npz_path):
    '''
    Extract chai metrics from npz files.
    '''
    data = np.load(npz_path)
    
    return {
        "filename": npz_path.name,
        "aggregate_score": float(data["aggregate_score"][0]),
        "ptm": float(data["ptm"][0]),
        "iptm": float(data["iptm"][0]),
        "per_chain_ptm": data["per_chain_ptm"][0].tolist(),
        "per_chain_pair_iptm": data["per_chain_pair_iptm"][0].tolist(),
        "has_inter_chain_clashes": bool(data["has_inter_chain_clashes"][0]),
        "chain_chain_clashes": data["chain_chain_clashes"][0].tolist()
    }


def round_sig(x, sig=5):
    """
    Round to significant digits.
    """
    if isinstance(x, float):
        return round(x, sig - int(np.floor(np.log10(abs(x)))) - 1) if x != 0 else 0.0
    if isinstance(x, list):
        return [round_sig(v, sig) for v in x]
    if isinstance(x, (np.ndarray, tuple)):
        return [round_sig(float(v), sig) for v in x]
    return x


def extract_boltz2_metrics(json_path):
    """
    Extracts and rounds Boltz2 metrics from a .json file into a dictionary.
    Each value is rounded to 5 significant digits.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    return {
        "filename": json_path.stem,
        "boltz2_confidence_score": round_sig(data.get("confidence_score")),
        "boltz2_ptm": round_sig(data.get("ptm")),
        "boltz2_iptm": round_sig(data.get("iptm")),
        "boltz2_ligand_iptm": round_sig(data.get("ligand_iptm")),
        "boltz2_protein_iptm": round_sig(data.get("protein_iptm")),
        "boltz2_complex_plddt": round_sig(data.get("complex_plddt")),
        "boltz2_complex_iplddt": round_sig(data.get("complex_iplddt")),
        "boltz2_complex_pde": round_sig(data.get("complex_pde")),
        "boltz2_complex_ipde": round_sig(data.get("complex_ipde")),
        "boltz2_chains_ptm": round_sig(data.get("chains_ptm", {})),
        "boltz2_pair_chains_iptm": round_sig(data.get("pair_chains_iptm", {})),
    }



def extract_boltz2_metrics_from_json(conf_json_path):
    with open(conf_json_path, "r") as f:
        data = json.load(f)

    return {
        "confidence_score": round(data.get("confidence_score", 0.0), 5),
        "ptm": round(data.get("ptm", 0.0), 5),
        "iptm": round(data.get("iptm", 0.0), 5),
        "ligand_iptm": round(data.get("ligand_iptm", 0.0), 5),
        "protein_iptm": round(data.get("protein_iptm", 0.0), 5),
        "complex_plddt": round(data.get("complex_plddt", 0.0), 5),
        "complex_iplddt": round(data.get("complex_iplddt", 0.0), 5),
        "complex_pde": round(data.get("complex_pde", 0.0), 5),
        "complex_ipde": round(data.get("complex_ipde", 0.0), 5),
        "chains_ptm": data.get("chains_ptm", {}),
        "pair_chains_iptm": data.get("pair_chains_iptm", {}),
    }

class DockingMetrics(Step):
    def __init__(self, input_dir='pipeline_output/docking', output_dir='pipeline_output/docking'):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def __execute(self, df: pd.DataFrame, tmp_dir: str) -> list:
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Directory does not exist: {self.input_dir}")
        
        results = []

        for _, row in df.iterrows():
            entry_name = row['Entry']
            ligand_name = row['substrate_name']

            # ---Extract vina docking metrics---
            vina_dir = Path(row['vina_dir']).parent
            vina_metrics = extract_docking_metrics_vina(vina_dir, entry_name, ligand_name)
            row_result = {'vina_affinities': vina_metrics if vina_metrics else {}}
            results.append(row_result)

            # ---Exract chai docking metrics---
            chai_dir = Path(row['chai_dir']) / 'chai'
            npz_files = list(chai_dir.rglob("*.npz"))

            # Initialize per-metric dicts
            aggregate_dict = {}
            ptm_dict = {}
            iptm_dict = {}
            per_chain_ptm_dict = {}
            per_chain_pair_iptm_dict = {}
            has_clashes_dict = {}
            chain_chain_clashes_dict = {}

            for npz_file in npz_files:
                try:
                    metrics = extract_chai_metrics(npz_file)
                    fname = npz_file.stem  # no .npz extension

                    aggregate_dict[fname] = round_sig(metrics["aggregate_score"])
                    ptm_dict[fname] = round_sig(metrics["ptm"])
                    iptm_dict[fname] = round_sig(metrics["iptm"])
                    per_chain_ptm_dict[fname] = round_sig(metrics["per_chain_ptm"])
                    per_chain_pair_iptm_dict[fname] = round_sig(metrics["per_chain_pair_iptm"])
                    has_clashes_dict[fname] = metrics["has_inter_chain_clashes"]  # boolean
                    chain_chain_clashes_dict[fname] = round_sig(metrics["chain_chain_clashes"])

                except Exception as e:
                    print(f"Skipped {npz_file.name}: {e}")
                    continue

            # Add all dicts to row
            row_result.update({
                'chai_aggregate_score': aggregate_dict,
                'chai_ptm': ptm_dict,
                'chai_iptm': iptm_dict,
                'chai_per_chain_ptm': per_chain_ptm_dict,
                'chai_per_chain_pair_iptm': per_chain_pair_iptm_dict,
                'chai_has_clashes': has_clashes_dict,
                'chai_chain_chain_clashes': chain_chain_clashes_dict,
            })

            # ---Exract boltz2 docking metrics---
            boltz2_dir = Path(row['boltz_dir']) / f'boltz_results_{entry_name}' / 'predictions' / entry_name

            # Initialize per-metric dicts                        
            boltz2_metrics_per_model = {
                "boltz2_confidence_score": {},
                "boltz2_ptm": {},
                "boltz2_iptm": {},
                "boltz2_ligand_iptm": {},
                "boltz2_protein_iptm": {},
                "boltz2_complex_plddt": {},
                "boltz2_complex_iplddt": {},
                "boltz2_complex_pde": {},
                "boltz2_complex_ipde": {},
                "boltz2_chains_ptm": {},
                "boltz2_pair_chains_iptm": {}
            }

            for json_file in sorted(boltz2_dir.glob("confidence_*.json")):
                model_name = json_file.stem.replace("confidence_", "").replace(".json", "")
                try:
                    metrics = extract_boltz2_metrics_from_json(json_file)

                    boltz2_metrics_per_model["boltz2_confidence_score"][model_name] = metrics["confidence_score"]
                    boltz2_metrics_per_model["boltz2_ptm"][model_name] = metrics["ptm"]
                    boltz2_metrics_per_model["boltz2_iptm"][model_name] = metrics["iptm"]
                    boltz2_metrics_per_model["boltz2_ligand_iptm"][model_name] = metrics["ligand_iptm"]
                    boltz2_metrics_per_model["boltz2_protein_iptm"][model_name] = metrics["protein_iptm"]
                    boltz2_metrics_per_model["boltz2_complex_plddt"][model_name] = metrics["complex_plddt"]
                    boltz2_metrics_per_model["boltz2_complex_iplddt"][model_name] = metrics["complex_iplddt"]
                    boltz2_metrics_per_model["boltz2_complex_pde"][model_name] = metrics["complex_pde"]
                    boltz2_metrics_per_model["boltz2_complex_ipde"][model_name] = metrics["complex_ipde"]
                    boltz2_metrics_per_model["boltz2_chains_ptm"][model_name] = metrics["chains_ptm"]
                    boltz2_metrics_per_model["boltz2_pair_chains_iptm"][model_name] = metrics["pair_chains_iptm"]

                except Exception as e:
                    print(f"ailed to parse {json_file.name}: {e}")

            row_result.update(boltz2_metrics_per_model)
            results.append(row_result)

        return results

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.output_dir:
            print("No output directory provided")
            return df

        results = self.__execute(df, self.output_dir)
        results_df = pd.DataFrame(results)
        output_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)

        return output_df

