"""
Alignment-Based Metrics Calculator
Computes alignment-based quality scores: ESM-MSA, Identity, BLOSUM62/PFASUM15

Input: 
  - Target sequences: FASTA files in 'target_seqs/' directory
  - Reference sequences: FASTA files in 'reference_seqs/' directory
Output: CSV file with alignment-based metrics
"""

import os
import tempfile
import subprocess
from pathlib import Path
from glob import glob
import logging
import argparse
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist

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


def add_metric(results, protein_id, metric_name, value):
    """Add a computed metric to the results dictionary"""
    if protein_id not in results:
        results[protein_id] = {}
    results[protein_id][metric_name] = value


class AlignmentMetricsCalculator:
    """Compute alignment-based metrics for protein sequences"""
    
    def __init__(self, target_seqs_dir='target_seqs', reference_seqs_dir='reference_seqs',
                 output_dir='output', substitution_matrix='BLOSUM62'):
        """
        Args:
            target_seqs_dir: Directory containing FASTA files with target sequences
            reference_seqs_dir: Directory containing FASTA files with reference sequences
            output_dir: Directory to save output CSV
            substitution_matrix: Either BLOSUM62 or PFASUM15
        """
        self.target_seqs_dir = Path(target_seqs_dir)
        self.reference_seqs_dir = Path(reference_seqs_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.substitution_matrix = substitution_matrix
        self.results = {}
        self.gap_open = 10
        self.gap_extend = 2
        
        # Validation
        if substitution_matrix not in ['BLOSUM62', 'PFASUM15']:
            raise ValueError(f"Substitution matrix must be BLOSUM62 or PFASUM15, got {substitution_matrix}")
        
        # Create directories if missing
        for d in [self.target_seqs_dir, self.reference_seqs_dir]:
            if not d.exists():
                logger.warning(f"Directory {d} does not exist. Creating it.")
                d.mkdir(exist_ok=True, parents=True)
    
    def concatenate_sequences(self):
        """Concatenate all FASTA files into single files"""
        target_seqs_file = Path(tempfile.gettempdir()) / "target_seqs_aligned.fasta"
        reference_seqs_file = Path(tempfile.gettempdir()) / "reference_seqs_aligned.fasta"
        
        # Target sequences
        with open(target_seqs_file, "w") as fh:
            for target_fasta in self.target_seqs_dir.glob("*.fasta"):
                for name, seq in zip(*parse_fasta(str(target_fasta), return_names=True, clean="unalign")):
                    print(f">{name}\n{seq}", file=fh)
        
        # Reference sequences
        with open(reference_seqs_file, "w") as fh:
            for ref_fasta in self.reference_seqs_dir.glob("*.fasta"):
                for name, seq in zip(*parse_fasta(str(ref_fasta), return_names=True, clean="unalign")):
                    print(f">{name}\n{seq}", file=fh)
        
        if target_seqs_file.stat().st_size == 0:
            logger.warning("No target sequences found")
        if reference_seqs_file.stat().st_size == 0:
            logger.warning("No reference sequences found")
        
        return target_seqs_file, reference_seqs_file
    
    def compute_esm_msa(self, target_seqs_file, reference_seqs_file):
        """Compute ESM-MSA scores using pHMMER for MSA construction"""
        logger.info("Computing ESM-MSA scores...")
        
        try:
            with tempfile.TemporaryDirectory() as output_dir:
                outfile = Path(output_dir) / "esm_msa_results.tsv"
                
                # Set up environment with PYTHONPATH for protein_gibbs_sampler
                env = os.environ.copy()
                gibbs_src = str(Path(__file__).parent / "protein_gibbs_sampler" / "src")
                if "PYTHONPATH" in env:
                    env["PYTHONPATH"] = gibbs_src + ":" + env["PYTHONPATH"]
                else:
                    env["PYTHONPATH"] = gibbs_src
                
                proc = subprocess.run(
                    ['python', "protein_gibbs_sampler/src/pgen/likelihood_esm_msa.py",
                     "-i", str(target_seqs_file),
                     "-o", str(outfile),
                     "--reference_msa", str(reference_seqs_file),
                     "--subset_strategy", "top_hits",
                     "--alignment_size", "31",
                     "--count_gaps",
                     "--mask_distance", "6",
                     "--device", "gpu"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                    env=env
                )
                
                if outfile.exists():
                    df = pd.read_table(outfile)
                    for i, row in df.iterrows():
                        add_metric(self.results, row["id"], "ESM-MSA", row["esm-msa"])
                        logger.info(f"  {row['id']}: {row['esm-msa']:.4f}")
                    logger.info("ESM-MSA complete")
                else:
                    logger.error("ESM-MSA output file not created")
        
        except Exception as e:
            logger.error(f"Failed to compute ESM-MSA: {e}")
    
    def compute_substitution_scores(self, target_seqs_file, reference_seqs_file):
        """Compute substitution matrix scores and identity to closest reference"""
        logger.info(f"Computing {self.substitution_matrix} and Identity scores...")
        
        try:
            # Determine substitution matrix file path
            # Try multiple possible locations with case variations
            matrix_name_lower = self.substitution_matrix.lower()
            possible_paths = [
                f'/nvme2/helen/EnzymeStructuralFiltering/fasta-36.3.8i/data/{matrix_name_lower}.mat',
                f'/fasta-36.3.8i/data/{matrix_name_lower}.mat',
                f'/opt/bin/../data/{matrix_name_lower}.mat',
                f'/fasta-36.3.8i/data/{self.substitution_matrix}.mat',
            ]
            
            substitution_matrix_file = None
            for path in possible_paths:
                if Path(path).exists():
                    substitution_matrix_file = path
                    logger.info(f"Using substitution matrix: {substitution_matrix_file}")
                    break
            
            if substitution_matrix_file is None:
                logger.warning(f"Substitution matrix file not found for {self.substitution_matrix}")
                logger.info("Skipping substitution matrix scoring")
                return
            
            # Run ggsearch36 for sequence similarity search
            search_results_file = self.output_dir / f"ggsearch_results_{self.substitution_matrix}.txt"
            
            with open(search_results_file, "w") as fh:
                proc = subprocess.run(
                    ['ggsearch36', '-f', str(self.gap_open), '-g', str(self.gap_extend),
                     '-s', substitution_matrix_file, '-b', '1',
                     str(target_seqs_file), str(reference_seqs_file)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True
                )
                print(proc.stdout.decode('utf-8'), file=fh)
            
            # Load substitution matrix (skip comments and handle whitespace)
            df_subst = pd.read_csv(substitution_matrix_file, delimiter=r"\s+", comment='#', index_col=0)
            subst_dict = {}
            for aa1 in df_subst.columns:
                for aa2 in df_subst.index:
                    subst_dict[(aa1, aa2)] = df_subst.loc[aa2, aa1]  # Note: may be symmetric but use explicit order
            
            # Parse sequences
            n_train, s_train = parse_fasta(str(reference_seqs_file), return_names=True)
            n_query, s_query = parse_fasta(str(target_seqs_file), return_names=True)
            n_query = [nq.strip() for nq in n_query]
            n_train = [nt.strip() for nt in n_train]
            
            # Build dicts using both full header and first token (for matching ggsearch output)
            train_seqs = {nt: st for st, nt in zip(s_train, n_train)}
            train_seqs_by_token = {nt.split()[0]: st for st, nt in zip(s_train, n_train)}
            
            query_seqs = {nq: sq for sq, nq in zip(s_query, n_query)}
            query_seqs_by_token = {nq.split()[0]: sq for sq, nq in zip(s_query, n_query)}
            
            # Parse ggsearch results
            with open(search_results_file) as f:
                lines = f.readlines()
            
            train_coming = False
            qns = []  # query names
            tns = []  # target names
            for i, line in enumerate(lines):
                if '!! No sequences' in line:
                    tns.append(None)
                
                if not train_coming:
                    if 'The best scores are:' in line:
                        train_coming = True
                else:
                    tns.append(line.split()[0])
                    train_coming = False
                
                if 'Library: ' in line:
                    qns.append(lines[i - 1].split('>')[-1].split()[0])
            
            # Compute metrics for each query-target pair from ggsearch results
            for qn, tn in zip(qns, tns):
                if tn is None:  # No hits found
                    add_metric(self.results, qn, f"Closest_reference_{self.substitution_matrix}", "")
                    add_metric(self.results, qn, self.substitution_matrix, 0)
                    add_metric(self.results, qn, "Identity", 0)
                else:
                    # Try to get full target name from ggsearch output for lookup
                    target_name = tn
                    target_seq = train_seqs_by_token.get(tn) or train_seqs.get(tn)
                    query_seq = query_seqs_by_token.get(qn) or query_seqs.get(qn)
                    
                    if query_seq is None or target_seq is None:
                        logger.warning(f"Could not find sequences for query={qn}, target={tn}")
                        add_metric(self.results, qn, f"Closest_reference_{self.substitution_matrix}", "")
                        add_metric(self.results, qn, self.substitution_matrix, 0)
                        add_metric(self.results, qn, "Identity", 0)
                        continue
                    
                    # Compute simple metrics directly from sequences without needle
                    # Identity: % of matching amino acids
                    min_len = min(len(query_seq), len(target_seq))
                    matches = sum(1 for i in range(min_len) if query_seq[i] == target_seq[i])
                    identity = matches / max(len(query_seq), len(target_seq)) if len(query_seq) > 0 else 0
                    
                    # Substitution score: sum of BLOSUM scores for aligned positions
                    subst_score = 0
                    for i in range(min_len):
                        aa1, aa2 = query_seq[i], target_seq[i]
                        if aa1 != '-' and aa2 != '-':
                            subst_score += subst_dict.get((aa1, aa2), 0)
                    
                    # Average substitution score per aligned position
                    if min_len > 0:
                        avg_subst_score = subst_score / min_len
                    else:
                        avg_subst_score = 0
                    
                    add_metric(self.results, qn, f"Closest_reference_{self.substitution_matrix}", target_name)
                    add_metric(self.results, qn, self.substitution_matrix, avg_subst_score)
                    add_metric(self.results, qn, "Identity", identity)
                    logger.info(f"  {qn} vs {target_name}: identity={identity:.4f}, subst_score={avg_subst_score:.4f}")
        
        except Exception as e:
            logger.error(f"Failed to compute substitution scores: {e}")
    
    def run(self, esm_msa=False, substitution=False):
        """Run selected alignment-based metric calculations"""
        logger.info(f"Starting alignment-based metrics calculation...")
        logger.info(f"Target sequences directory: {self.target_seqs_dir}")
        logger.info(f"Reference sequences directory: {self.reference_seqs_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Substitution matrix: {self.substitution_matrix}")
        
        # Concatenate sequences
        target_seqs_file, reference_seqs_file = self.concatenate_sequences()
        
        if esm_msa:
            self.compute_esm_msa(target_seqs_file, reference_seqs_file)
        
        if substitution:
            self.compute_substitution_scores(target_seqs_file, reference_seqs_file)
        
        return self.results
    
    def save_results(self, filename=None):
        """Save results to CSV file"""
        if not self.results:
            logger.warning("No results to save")
            return None
        
        if filename is None:
            filename = "alignment_metrics.csv"
        
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
    parser = argparse.ArgumentParser(description="Compute alignment-based metrics for protein sequences")
    parser.add_argument('--target_seqs_dir', default='target_seqs', help='Directory with target FASTA files')
    parser.add_argument('--reference_seqs_dir', default='reference_seqs', help='Directory with reference FASTA files')
    parser.add_argument('--output_dir', default='output', help='Output directory for CSV results')
    parser.add_argument('--substitution_matrix', default='BLOSUM62', choices=['BLOSUM62', 'PFASUM15'],
                        help='Substitution matrix to use')
    parser.add_argument('--esm_msa', action='store_true', default=False, help='Compute ESM-MSA scores')
    parser.add_argument('--substitution', action='store_true', default=False,
                        help='Compute substitution matrix and identity scores')
    parser.add_argument('--all', action='store_true', help='Compute all metrics')
    parser.add_argument('--output_file', default='alignment_metrics.csv', help='Output CSV filename')
    
    args = parser.parse_args()
    
    if args.all:
        args.esm_msa = True
        args.substitution = True
    
    if not (args.esm_msa or args.substitution):
        logger.info("No metrics specified. Use --all or select individual metrics")
        return
    
    calculator = AlignmentMetricsCalculator(
        target_seqs_dir=args.target_seqs_dir,
        reference_seqs_dir=args.reference_seqs_dir,
        output_dir=args.output_dir,
        substitution_matrix=args.substitution_matrix
    )
    
    calculator.run(
        esm_msa=args.esm_msa,
        substitution=args.substitution
    )
    
    calculator.save_results(args.output_file)


if __name__ == '__main__':
    main()
