"""
Single Sequence Metrics Calculator
Computes sequence-based quality scores: ESM-1v, ESM-1v mask6, CARP-640m, longest repeats

Input: FASTA files in 'target_seqs/' directory
Output: CSV file with single sequence metrics
"""

import tempfile
import subprocess
from pathlib import Path
from glob import glob
import logging
import argparse
import pandas as pd

# Optional imports
try:
    from pgen.utils import parse_fasta
    HAS_PGEN = True
except ImportError:
    HAS_PGEN = False
    # Fallback parse_fasta function
    def parse_fasta(filename, return_names=False, clean=None):
        """Simple FASTA parser as fallback"""
        names = []
        seqs = []
        with open(filename, 'r') as f:
            current_name = None
            current_seq = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    if current_name is not None:
                        seqs.append(''.join(current_seq))
                        names.append(current_name)
                    current_name = line[1:]
                    current_seq = []
                else:
                    current_seq.append(line)
            if current_name is not None:
                seqs.append(''.join(current_seq))
                names.append(current_name)
        if return_names:
            return names, seqs
        return seqs

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def add_metric(results, protein_id, metric_name, value):
    """Add a computed metric to the results dictionary"""
    if protein_id not in results:
        results[protein_id] = {}
    results[protein_id][metric_name] = value


def find_longest_repeat(seq, k):
    """Find the longest k-length repeat in a sequence"""
    longest = [1] * len(seq)
    pattern = [None] * len(seq)

    seq_len = len(seq)
    for i in range(seq_len):
        if i + k <= seq_len:
            pattern[i] = seq[i:i+k]
        if i - k >= 0:
            if pattern[i-k] == pattern[i]:
                longest[i] = longest[i-k] + 1
    return -1 * max(longest)


class SingleSequenceMetricsCalculator:
    """Compute single-sequence metrics for protein sequences"""
    
    def __init__(self, target_seqs_dir='target_seqs', output_dir='output', device='cuda:0'):
        """
        Args:
            target_seqs_dir: Directory containing FASTA files with target sequences
            output_dir: Directory to save output CSV
            device: Device for deep learning (cuda:0, cuda:1, cpu, etc.)
        """
        self.target_seqs_dir = Path(target_seqs_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.device = device
        self.results = {}
        
        if not self.target_seqs_dir.exists():
            logger.warning(f"Target sequences directory {self.target_seqs_dir} does not exist. Creating it.")
            self.target_seqs_dir.mkdir(exist_ok=True, parents=True)
    
    def compute_carp_640m(self):
        """Compute CARP-640M logp scores"""
        if not HAS_TORCH:
            logger.info("Skipping CARP-640m (PyTorch not available)")
            return
        
        logger.info("Computing CARP-640m scores...")
        target_seqs_file = Path(tempfile.gettempdir()) / "target_seqs_carp.fasta"
        
        # Concatenate all fasta files
        with open(target_seqs_file, "w") as fh:
            for target_fasta in self.target_seqs_dir.glob("*.fasta"):
                for name, seq in zip(*parse_fasta(str(target_fasta), return_names=True, clean="unalign")):
                    print(f">{name}\n{seq}", file=fh)
        
        if target_seqs_file.stat().st_size == 0:
            logger.warning("No target sequences found in fasta files")
            return
        
        try:
            with tempfile.TemporaryDirectory() as output_dir:
                proc = subprocess.run(
                    ['python', "/tmp/extract.py", "carp_640M", str(target_seqs_file), 
                     output_dir + "/", "--repr_layers", "logits", "--include", "logp", 
                     "--device", self.device],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True
                )
                logger.debug(proc.stdout.decode())
                
                # Read results
                results_file = Path(output_dir) / 'carp_640M_logp.tsv'
                if results_file.exists():
                    df = pd.read_table(results_file)
                    df = df.rename(columns={'name': 'id', 'logp': 'carp640m_logp'})
                    for _, row in df.iterrows():
                        add_metric(self.results, row["id"], "CARP-640m", row["carp640m_logp"])
                        logger.info(f"  {row['id']}: {row['carp640m_logp']:.4f}")
            
            logger.info("CARP-640m complete")
        
        except Exception as e:
            logger.error(f"Failed to compute CARP-640m: {e}")
    
    def compute_esm1v_unmasked(self):
        """Compute ESM-1v (unmasked) scores"""
        if not HAS_TORCH:
            logger.info("Skipping ESM-1v (PyTorch not available)")
            return
        
        logger.info("Computing ESM-1v (unmasked) scores...")
        
        for target_fasta in self.target_seqs_dir.glob("*.fasta"):
            try:
                with tempfile.TemporaryDirectory() as output_dir:
                    outfile = Path(output_dir) / "esm_results.tsv"
                    proc = subprocess.run(
                        ['python', "protein_gibbs_sampler/src/pgen/likelihood_esm.py",
                         "-i", str(target_fasta),
                         "-o", str(outfile),
                         "--model", "esm1v",
                         "--masking_off",
                         "--score_name", "score",
                         "--device", "gpu"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=True
                    )
                    
                    if outfile.exists():
                        df = pd.read_table(outfile)
                        for i, row in df.iterrows():
                            add_metric(self.results, row["id"], "ESM-1v", row["score"])
                            logger.info(f"  {row['id']}: {row['score']:.4f}")
            
            except Exception as e:
                logger.error(f"Error processing {target_fasta}: {e}")
        
        logger.info("ESM-1v complete")
    
    def compute_esm1v_mask6(self):
        """Compute ESM-1v with masking (mask distance 6) scores"""
        if not HAS_TORCH:
            logger.info("Skipping ESM-1v mask6 (PyTorch not available)")
            return
        
        logger.info("Computing ESM-1v mask6 scores...")
        
        for target_fasta in self.target_seqs_dir.glob("*.fasta"):
            try:
                with tempfile.TemporaryDirectory() as output_dir:
                    outfile = Path(output_dir) / "esm_results.tsv"
                    proc = subprocess.run(
                        ['python', "protein_gibbs_sampler/src/pgen/likelihood_esm.py",
                         "-i", str(target_fasta),
                         "-o", str(outfile),
                         "--model", "esm1v",
                         "--mask_distance", "6",
                         "--score_name", "score",
                         "--device", "gpu"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=True
                    )
                    
                    if outfile.exists():
                        df = pd.read_table(outfile)
                        for i, row in df.iterrows():
                            add_metric(self.results, row["id"], "ESM-1v_mask6", row["score"])
                            logger.info(f"  {row['id']}: {row['score']:.4f}")
            
            except Exception as e:
                logger.error(f"Error processing {target_fasta}: {e}")
        
        logger.info("ESM-1v mask6 complete")
    
    def compute_longest_repeats(self, k_values=[1, 2, 3, 4]):
        """Compute longest k-length repeats"""
        logger.info(f"Computing longest repeats (k={k_values})...")
        
        for target_fasta in self.target_seqs_dir.glob("*.fasta"):
            try:
                for name, seq in zip(*parse_fasta(str(target_fasta), return_names=True, clean="unalign")):
                    for k in k_values:
                        score = find_longest_repeat(seq, k)
                        add_metric(self.results, name, f"longest_repeat_{k}", score)
                    
                    logger.info(f"  {name}: computed repeats for k={k_values}")
            
            except Exception as e:
                logger.error(f"Error processing {target_fasta}: {e}")
        
        logger.info("Longest repeats complete")
    
    def run(self, carp_640m=False, esm1v=False, esm1v_mask6=False, repeats=False):
        """Run selected single-sequence metric calculations"""
        logger.info(f"Starting single-sequence metrics calculation...")
        logger.info(f"Target sequences directory: {self.target_seqs_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        
        if carp_640m:
            self.compute_carp_640m()
        
        if esm1v:
            self.compute_esm1v_unmasked()
        
        if esm1v_mask6:
            self.compute_esm1v_mask6()
        
        if repeats:
            self.compute_longest_repeats()
        
        return self.results
    
    def save_results(self, filename=None):
        """Save results to CSV file"""
        if not self.results:
            logger.warning("No results to save")
            return None
        
        if filename is None:
            filename = "single_sequence_metrics.csv"
        
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
    parser = argparse.ArgumentParser(description="Compute single-sequence metrics for protein sequences")
    parser.add_argument('--target_seqs_dir', default='target_seqs', help='Directory containing FASTA files')
    parser.add_argument('--output_dir', default='output', help='Output directory for CSV results')
    parser.add_argument('--device', default='cuda:0', help='Device for computation')
    parser.add_argument('--carp_640m', action='store_true', default=False, help='Compute CARP-640m scores')
    parser.add_argument('--esm1v', action='store_true', default=False, help='Compute ESM-1v (unmasked) scores')
    parser.add_argument('--esm1v_mask6', action='store_true', default=False, help='Compute ESM-1v mask6 scores')
    parser.add_argument('--repeats', action='store_true', default=False, help='Compute longest repeat scores')
    parser.add_argument('--all', action='store_true', help='Compute all metrics')
    parser.add_argument('--output_file', default='single_sequence_metrics.csv', help='Output CSV filename')
    
    args = parser.parse_args()
    
    if args.all:
        args.carp_640m = True
        args.esm1v = True
        args.esm1v_mask6 = True
        args.repeats = True
    
    if not (args.carp_640m or args.esm1v or args.esm1v_mask6 or args.repeats):
        logger.info("No metrics specified. Use --all or select individual metrics")
        return
    
    calculator = SingleSequenceMetricsCalculator(
        target_seqs_dir=args.target_seqs_dir,
        output_dir=args.output_dir,
        device=args.device
    )
    
    calculator.run(
        carp_640m=args.carp_640m,
        esm1v=args.esm1v,
        esm1v_mask6=args.esm1v_mask6,
        repeats=args.repeats
    )
    
    calculator.save_results(args.output_file)


if __name__ == '__main__':
    main()
