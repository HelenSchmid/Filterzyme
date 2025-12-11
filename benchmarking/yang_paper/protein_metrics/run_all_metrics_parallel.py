"""
Parallel Metrics Runner
Executes all three metric calculators simultaneously using multiprocessing.

Since the three metric calculators are completely independent (different input data,
different external tools, separate output files), they can run in parallel:
  - Structure metrics: GPU (ESM-IF, ProteinMPNN)
  - Single sequence metrics: CPU/GPU (ESM-1v, CARP-640M)
  - Alignment metrics: CPU (ggsearch36, needle)

Usage:
    python run_all_metrics_parallel.py --all
    python run_all_metrics_parallel.py --structure --single_sequence --alignment
"""

import argparse
import logging
import sys
from pathlib import Path
from multiprocessing import Process, Queue
import time
import pandas as pd

from structure_metrics import StructureMetricsCalculator
from single_sequence_metrics import SingleSequenceMetricsCalculator
from alignment_metrics import AlignmentMetricsCalculator
from merge_metrics import merge_metrics, create_summary_stats

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(processName)s] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_structure_metrics(queue, pdb_dir, output_dir, device, output_file):
    """Run structure metrics in a separate process"""
    try:
        logger.info("Starting structure metrics calculation...")
        calculator = StructureMetricsCalculator(
            pdb_dir=pdb_dir,
            output_dir=output_dir,
            device=device
        )
        calculator.run(esm_if=True, proteinmpnn=True, plddt=False)
        calculator.save_results(output_file)
        logger.info(f"Structure metrics completed. Results saved to {output_file}")
        queue.put(("structure", True, output_file))
    except Exception as e:
        logger.error(f"Structure metrics failed: {e}", exc_info=True)
        queue.put(("structure", False, str(e)))


def run_single_sequence_metrics(queue, target_seqs_dir, output_dir, device, output_file):
    """Run single sequence metrics in a separate process"""
    try:
        logger.info("Starting single sequence metrics calculation...")
        calculator = SingleSequenceMetricsCalculator(
            target_seqs_dir=target_seqs_dir,
            output_dir=output_dir,
            device=device
        )
        calculator.run(carp_640m=True, esm1v=True, esm1v_mask6=True, repeats=True)
        calculator.save_results(output_file)
        logger.info(f"Single sequence metrics completed. Results saved to {output_file}")
        queue.put(("single_sequence", True, output_file))
    except Exception as e:
        logger.error(f"Single sequence metrics failed: {e}", exc_info=True)
        queue.put(("single_sequence", False, str(e)))


def run_alignment_metrics(queue, target_seqs_dir, reference_seqs_dir, output_dir, 
                         substitution_matrix, output_file):
    """Run alignment metrics in a separate process"""
    try:
        logger.info("Starting alignment metrics calculation...")
        calculator = AlignmentMetricsCalculator(
            target_seqs_dir=target_seqs_dir,
            reference_seqs_dir=reference_seqs_dir,
            output_dir=output_dir,
            substitution_matrix=substitution_matrix
        )
        calculator.run(esm_msa=True, substitution=True)
        calculator.save_results(output_file)
        logger.info(f"Alignment metrics completed. Results saved to {output_file}")
        queue.put(("alignment", True, output_file))
    except Exception as e:
        logger.error(f"Alignment metrics failed: {e}", exc_info=True)
        queue.put(("alignment", False, str(e)))


def main():
    parser = argparse.ArgumentParser(
        description="Run all metric calculators in parallel"
    )
    parser.add_argument('--pdb_dir', default='pdbs', help='Directory containing PDB files')
    parser.add_argument('--target_seqs_dir', default='target_seqs', 
                       help='Directory containing target FASTA files')
    parser.add_argument('--reference_seqs_dir', default='reference_seqs',
                       help='Directory containing reference FASTA files')
    parser.add_argument('--output_dir', default='output', help='Output directory for CSV results')
    parser.add_argument('--device', default='cuda:0', 
                       help='Device for computation (cuda:0, cuda:1, cpu, etc.)')
    parser.add_argument('--substitution_matrix', default='BLOSUM62',
                       help='Substitution matrix for alignment (BLOSUM62 or PFASUM15)')
    
    parser.add_argument('--structure', action='store_true', help='Run structure metrics')
    parser.add_argument('--single_sequence', action='store_true', help='Run single sequence metrics')
    parser.add_argument('--alignment', action='store_true', help='Run alignment metrics')
    parser.add_argument('--all', action='store_true', help='Run all metrics in parallel')
    parser.add_argument('--merge', action='store_true', 
                       help='Merge results into single CSV (requires all metrics to complete)')
    parser.add_argument('--stats', action='store_true', help='Compute statistics on merged results')
    
    args = parser.parse_args()
    
    # Determine which metrics to run
    if args.all:
        args.structure = True
        args.single_sequence = True
        args.alignment = True
    
    if not (args.structure or args.single_sequence or args.alignment):
        logger.error("Please specify which metrics to run: --structure, --single_sequence, "
                    "--alignment, or --all")
        return 1
    
    # Output filenames
    structure_output = f"{args.output_dir}/structure_metrics.csv"
    single_seq_output = f"{args.output_dir}/single_sequence_metrics.csv"
    alignment_output = f"{args.output_dir}/alignment_metrics.csv"
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    
    logger.info("=" * 80)
    logger.info("STARTING PARALLEL METRICS CALCULATION")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  PDB directory: {args.pdb_dir}")
    logger.info(f"  Target sequences directory: {args.target_seqs_dir}")
    logger.info(f"  Reference sequences directory: {args.reference_seqs_dir}")
    logger.info(f"  Device: {args.device}")
    logger.info(f"  Metrics to compute:")
    if args.structure:
        logger.info(f"    ✓ Structure metrics (ESM-IF, ProteinMPNN)")
    if args.single_sequence:
        logger.info(f"    ✓ Single sequence metrics (ESM-1v, CARP-640M, repeats)")
    if args.alignment:
        logger.info(f"    ✓ Alignment metrics (ESM-MSA, BLOSUM62/PFASUM15)")
    logger.info("=" * 80)
    
    # Queue to collect results from processes
    queue = Queue()
    processes = []
    
    start_time = time.time()
    
    # Start all requested processes
    if args.structure:
        p = Process(
            target=run_structure_metrics,
            args=(queue, args.pdb_dir, args.output_dir, args.device, structure_output),
            name="StructureMetrics"
        )
        p.start()
        processes.append(("structure", p, structure_output))
    
    if args.single_sequence:
        p = Process(
            target=run_single_sequence_metrics,
            args=(queue, args.target_seqs_dir, args.output_dir, args.device, single_seq_output),
            name="SingleSequenceMetrics"
        )
        p.start()
        processes.append(("single_sequence", p, single_seq_output))
    
    if args.alignment:
        p = Process(
            target=run_alignment_metrics,
            args=(queue, args.target_seqs_dir, args.reference_seqs_dir, args.output_dir,
                  args.substitution_matrix, alignment_output),
            name="AlignmentMetrics"
        )
        p.start()
        processes.append(("alignment", p, alignment_output))
    
    # Wait for all processes to complete
    logger.info(f"Running {len(processes)} metric calculators in parallel...")
    
    results = {}
    completed = 0
    failed = 0
    
    for _ in range(len(processes)):
        metric_name, success, output_file = queue.get()
        results[metric_name] = (success, output_file)
        if success:
            completed += 1
        else:
            failed += 1
    
    # Join all processes
    for _, p, _ in processes:
        p.join()
    
    elapsed_time = time.time() - start_time
    
    logger.info("=" * 80)
    logger.info("PARALLEL EXECUTION COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Completed: {completed}/{len(processes)} metrics")
    logger.info(f"Failed: {failed}/{len(processes)} metrics")
    logger.info(f"Total time: {elapsed_time:.1f} seconds")
    logger.info("=" * 80)
    
    # Print results
    for metric_name, (success, output_file) in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"{metric_name:20s}: {status} - {output_file}")
    
    # Merge results if requested and all succeeded
    if args.merge and failed == 0:
        logger.info("=" * 80)
        logger.info("MERGING RESULTS")
        logger.info("=" * 80)
        
        merged_file = f"{args.output_dir}/all_metrics_combined.csv"
        
        try:
            merge_metrics(input_dir=args.output_dir, output_file=merged_file)
            logger.info(f"✓ Merged metrics saved to {merged_file}")
            
            # Compute statistics if requested
            if args.stats:
                merged_df = pd.read_csv(merged_file, index_col=0)
                create_summary_stats(merged_df, output_dir=args.output_dir)
                logger.info(f"✓ Statistics saved to {args.output_dir}/metrics_statistics.csv")
        except Exception as e:
            logger.error(f"Merging failed: {e}", exc_info=True)
            return 1
    
    # Return exit code based on failures
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
