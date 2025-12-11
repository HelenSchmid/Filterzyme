"""
Structure Metrics Calculator
Computes structure-based quality scores: ESM-IF, ProteinMPNN (AlphaFold2 pLDDT disabled)

Input: PDB files in 'pdbs/' directory
Output: CSV file with structure metrics
"""

import os
import tempfile
import subprocess
from pathlib import Path
from glob import glob
import logging
import argparse
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import esm
    from biotite.structure.io import pdb
    import torch
    HAS_DEEP_LEARNING = True
except ImportError:
    HAS_DEEP_LEARNING = False
    logger.warning("Deep learning dependencies not found. Some metrics will be skipped.")


def add_metric(results, protein_id, metric_name, value):
    """Add a computed metric to the results dictionary"""
    if protein_id not in results:
        results[protein_id] = {}
    results[protein_id][metric_name] = value


class StructureMetricsCalculator:
    """Compute structure-based metrics for protein PDB files"""
    
    def __init__(self, pdb_dir='pdbs', output_dir='output', device='cuda:0'):
        """
        Args:
            pdb_dir: Directory containing PDB files
            output_dir: Directory to save output CSV
            device: Device for deep learning models (cuda:0, cuda:1, cpu, etc.)
        """
        self.pdb_dir = Path(pdb_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.device = device
        self.results = {}
        
        if not self.pdb_dir.exists():
            logger.warning(f"PDB directory {self.pdb_dir} does not exist. Creating it.")
            self.pdb_dir.mkdir(exist_ok=True, parents=True)
    
    def compute_esm_if(self):
        """Compute ESM-IF (Inverse Folding) scores"""
        if not HAS_DEEP_LEARNING:
            logger.info("Skipping ESM-IF (PyTorch not available)")
            return
        
        logger.info("Computing ESM-IF scores...")
        try:
            esm_if_model, esm_if_alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
            esm_if_model.eval()
            
            pdb_files = list(self.pdb_dir.glob("*.pdb"))
            if not pdb_files:
                logger.warning(f"No PDB files found in {self.pdb_dir}")
                return
            
            for pdb_file in pdb_files:
                try:
                    fstem = pdb_file.stem
                    coords, seq = esm.inverse_folding.util.load_coords(str(pdb_file), "A")
                    ll, _ = esm.inverse_folding.util.score_sequence(
                        esm_if_model, esm_if_alphabet, coords, str(seq))
                    add_metric(self.results, fstem, "ESM-IF", float(ll))
                    logger.info(f"  {fstem}: {ll:.4f}")
                except Exception as e:
                    logger.error(f"  Error processing {pdb_file}: {e}")
            
            del esm_if_model
            del esm_if_alphabet
            torch.cuda.empty_cache() if 'cuda' in self.device else None
            logger.info("ESM-IF complete")
        
        except Exception as e:
            logger.error(f"Failed to compute ESM-IF: {e}")
    
    def compute_proteinmpnn(self):
        """Compute ProteinMPNN scores"""
        logger.info("Computing ProteinMPNN scores...")
        pdb_files = list(self.pdb_dir.glob("*.pdb"))
        
        if not pdb_files:
            logger.warning(f"No PDB files found in {self.pdb_dir}")
            return
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            for i, pdb_file in enumerate(pdb_files):
                try:
                    fstem = pdb_file.stem
                    command_line_arguments = [
                        "python",
                        "ProteinMPNN/vanilla_proteinmpnn/protein_mpnn_run.py",
                        "--pdb_path", str(pdb_file),
                        "--pdb_path_chains", "A",
                        "--score_only", "1",
                        "--save_score", "1",
                        "--out_folder", tmp_dir,
                        "--batch_size", "1"
                    ]
                    
                    outfile = Path(tmp_dir) / f"outfile_{i}.txt"
                    with open(outfile, "w") as fh:
                        proc = subprocess.run(command_line_arguments, stdout=subprocess.PIPE, check=True)
                        print(proc.stdout.decode('utf-8'), file=fh)
                    
                    with open(outfile, "r") as score_file_h:
                        score_file_lines = score_file_h.readlines()
                    
                    score_line = score_file_lines[-2].split(",")
                    score_parts = score_line[1].strip().split(": ")
                    assert score_parts[0] == "mean"
                    score = -1 * float(score_parts[1])
                    add_metric(self.results, fstem, "ProteinMPNN", score)
                    logger.info(f"  {fstem}: {score:.4f}")
                
                except Exception as e:
                    logger.error(f"  Error processing {pdb_file}: {e}")
        
        logger.info("ProteinMPNN complete")
    
    # def compute_plddt(self):
    #     """Compute AlphaFold2 pLDDT scores from b-factors"""
    #     if not HAS_DEEP_LEARNING:
    #         logger.info("Skipping pLDDT (biotite not available)")
    #         return
    #     
    #     logger.info("Computing AlphaFold2 pLDDT scores...")
    #     pdb_files = list(self.pdb_dir.glob("*.pdb"))
    #     
    #     if not pdb_files:
    #         logger.warning(f"No PDB files found in {self.pdb_dir}")
    #         return
    #     
    #     for pdb_file in pdb_files:
    #         try:
    #             fstem = pdb_file.stem
    #             pdb_data = pdb.PDBFile.read(str(pdb_file))
    #             atoms = pdb_data.get_structure(extra_fields=['b_factor'])
    #             
    #             prev_residue = -1
    #             plddt_sum = 0
    #             residue_count = 0
    #             
    #             for a in atoms[0]:
    #                 if a.res_id != prev_residue:
    #                     prev_residue = a.res_id
    #                     residue_count += 1
    #                     plddt_sum += a.b_factor
    #             
    #             if residue_count > 0:
    #                 plddt = plddt_sum / residue_count
    #                 add_metric(self.results, fstem, "AlphaFold2_pLDDT", plddt)
    #                 logger.info(f"  {fstem}: {plddt:.2f}")
    #         
    #         except Exception as e:
    #             logger.error(f"  Error processing {pdb_file}: {e}")
    #     
    #     logger.info("pLDDT complete")
    
    def run(self, esm_if=True, proteinmpnn=True, plddt=False):
        """Run selected structure metric calculations"""
        logger.info(f"Starting structure metrics calculation...")
        logger.info(f"PDB directory: {self.pdb_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        
        if esm_if and HAS_DEEP_LEARNING:
            self.compute_esm_if()
        
        if proteinmpnn:
            self.compute_proteinmpnn()
        
        # pLDDT computation disabled - AlphaFold b-factors not available
        # if plddt and HAS_DEEP_LEARNING:
        #     self.compute_plddt()
        
        return self.results
    
    def save_results(self, filename=None):
        """Save results to CSV file"""
        if not self.results:
            logger.warning("No results to save")
            return None
        
        if filename is None:
            filename = "structure_metrics.csv"
        
        # Check if filename is a full path or just a filename
        output_path = Path(filename)
        if not output_path.is_absolute() and output_path.parent == Path("."):
            # It's just a filename, construct the full path
            output_path = self.output_dir / filename
        
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame.from_dict(self.results, orient="index")
        df.to_csv(output_path)
        logger.info(f"Results saved to {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(description="Compute structure metrics for protein PDB files")
    parser.add_argument('--pdb_dir', default='pdbs', help='Directory containing PDB files')
    parser.add_argument('--output_dir', default='output', help='Output directory for CSV results')
    parser.add_argument('--device', default='cuda:0', help='Device for computation (cuda:0, cuda:1, cpu, etc.)')
    parser.add_argument('--esm_if', action='store_true', default=False, help='Compute ESM-IF scores')
    parser.add_argument('--proteinmpnn', action='store_true', default=False, help='Compute ProteinMPNN scores')
    # parser.add_argument('--plddt', action='store_true', default=False, help='Compute AlphaFold2 pLDDT scores (disabled)')
    parser.add_argument('--all', action='store_true', help='Compute all metrics')
    parser.add_argument('--output_file', default='structure_metrics.csv', help='Output CSV filename')
    
    args = parser.parse_args()
    
    # If --all is specified, enable all available metrics
    if args.all:
        args.esm_if = True
        args.proteinmpnn = True
    
    # At least one metric must be selected
    if not (args.esm_if or args.proteinmpnn):
        logger.info("No metrics specified. Use --all, or select individual metrics with --esm_if, --proteinmpnn")
        return
    
    calculator = StructureMetricsCalculator(
        pdb_dir=args.pdb_dir,
        output_dir=args.output_dir,
        device=args.device
    )
    
    calculator.run(
        esm_if=args.esm_if,
        proteinmpnn=args.proteinmpnn,
        plddt=False  # pLDDT disabled - no b-factor data available
    )
    
    calculator.save_results(args.output_file)


if __name__ == '__main__':
    main()
