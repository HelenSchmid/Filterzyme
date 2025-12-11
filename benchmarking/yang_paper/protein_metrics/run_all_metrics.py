#!/usr/bin/env python
"""
Convenience runner script to execute all three metric calculators in sequence.
Provides a unified interface for computing structure, single-sequence, and alignment metrics.
"""

import argparse
import logging
from pathlib import Path
from structure_metrics import StructureMetricsCalculator
from single_sequence_metrics import SingleSequenceMetricsCalculator
from alignment_metrics import AlignmentMetricsCalculator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_all_metrics(
    pdbs_dir='pdbs',
    targets_dir='target_seqs',
    references_dir='reference_seqs',
    output_dir='output',
    device='cuda:0',
    structure=True,
    single_seq=True,
    alignment=True,
    substitution_matrix='BLOSUM62'
):
    """
    Run all metric calculators
    
    Args:
        pdbs_dir: Directory with PDB files
        targets_dir: Directory with target FASTA files
        references_dir: Directory with reference FASTA files
        output_dir: Output directory for CSVs
        device: GPU device for computation
        structure: Run structure metrics
        single_seq: Run single-sequence metrics
        alignment: Run alignment metrics
        substitution_matrix: BLOSUM62 or PFASUM15
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    logger.info("=" * 80)
    logger.info("PROTEIN METRICS CALCULATOR - UNIFIED RUNNER")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_path}")
    logger.info(f"GPU Device: {device}")
    logger.info("")
    
    results_files = []
    
    # Run Structure Metrics
    if structure:
        logger.info("=" * 80)
        logger.info("RUNNING STRUCTURE METRICS")
        logger.info("=" * 80)
        try:
            calc = StructureMetricsCalculator(
                pdb_dir=pdbs_dir,
                output_dir=output_dir,
                device=device
            )
            calc.run(esm_if=True, proteinmpnn=True, plddt=False)
            output_file = calc.save_results('structure_metrics.csv')
            results_files.append(output_file)
            logger.info(f"✓ Structure metrics saved to {output_file}\n")
        except Exception as e:
            logger.error(f"✗ Structure metrics failed: {e}\n")
    
    # Run Single-Sequence Metrics
    if single_seq:
        logger.info("=" * 80)
        logger.info("RUNNING SINGLE-SEQUENCE METRICS")
        logger.info("=" * 80)
        try:
            calc = SingleSequenceMetricsCalculator(
                target_seqs_dir=targets_dir,
                output_dir=output_dir,
                device=device
            )
            calc.run(carp_640m=True, esm1v=True, esm1v_mask6=True, repeats=True)
            output_file = calc.save_results('single_sequence_metrics.csv')
            results_files.append(output_file)
            logger.info(f"✓ Single-sequence metrics saved to {output_file}\n")
        except Exception as e:
            logger.error(f"✗ Single-sequence metrics failed: {e}\n")
    
    # Run Alignment Metrics
    if alignment:
        logger.info("=" * 80)
        logger.info("RUNNING ALIGNMENT-BASED METRICS")
        logger.info("=" * 80)
        try:
            calc = AlignmentMetricsCalculator(
                target_seqs_dir=targets_dir,
                reference_seqs_dir=references_dir,
                output_dir=output_dir,
                substitution_matrix=substitution_matrix
            )
            calc.run(esm_msa=True, substitution=True)
            output_file = calc.save_results('alignment_metrics.csv')
            results_files.append(output_file)
            logger.info(f"✓ Alignment metrics saved to {output_file}\n")
        except Exception as e:
            logger.error(f"✗ Alignment metrics failed: {e}\n")
    
    # Summary
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Completed {len(results_files)} metric calculations")
    logger.info("\nOutput files:")
    for f in results_files:
        logger.info(f"  - {f}")
    
    logger.info("\nTo combine all metrics into a single file, use:")
    logger.info(f"  python merge_metrics.py --input_dir {output_dir} --output all_metrics.csv")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Run all protein metric calculators (structure, single-sequence, alignment)"
    )
    
    # Input/Output
    parser.add_argument('--pdbs', default='pdbs', help='PDB files directory')
    parser.add_argument('--targets', default='target_seqs', help='Target FASTA sequences directory')
    parser.add_argument('--references', default='reference_seqs', help='Reference FASTA sequences directory')
    parser.add_argument('--output', default='output', help='Output directory for results')
    
    # Computation
    parser.add_argument('--device', default='cuda:0', help='GPU device (cuda:0, cuda:1, cpu, etc.)')
    parser.add_argument('--substitution_matrix', default='BLOSUM62', choices=['BLOSUM62', 'PFASUM15'])
    
    # Which calculators to run
    parser.add_argument('--structure', action='store_true', default=False, help='Run structure metrics')
    parser.add_argument('--single_seq', action='store_true', default=False, help='Run single-sequence metrics')
    parser.add_argument('--alignment', action='store_true', default=False, help='Run alignment metrics')
    parser.add_argument('--all', action='store_true', help='Run all metric calculators')
    
    args = parser.parse_args()
    
    # If --all specified, run all calculators
    if args.all:
        args.structure = True
        args.single_seq = True
        args.alignment = True
    
    # If nothing specified, show help
    if not (args.structure or args.single_seq or args.alignment):
        parser.print_help()
        print("\nExample: python run_all_metrics.py --all --output results/")
        return
    
    # Run the calculators
    run_all_metrics(
        pdbs_dir=args.pdbs,
        targets_dir=args.targets,
        references_dir=args.references,
        output_dir=args.output,
        device=args.device,
        structure=args.structure,
        single_seq=args.single_seq,
        alignment=args.alignment,
        substitution_matrix=args.substitution_matrix
    )


if __name__ == '__main__':
    main()
