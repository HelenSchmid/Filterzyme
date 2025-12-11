#!/usr/bin/env python
"""
Utility script to merge metric CSV files into a single combined CSV.
Matches rows by protein/sequence ID and combines all columns.
"""

import argparse
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def merge_metrics(input_dir='output', output_file='all_metrics_combined.csv', pattern='*_metrics.csv'):
    """
    Merge all metric CSV files in a directory
    
    Args:
        input_dir: Directory containing metric CSV files
        output_file: Output CSV filename or full path
        pattern: Glob pattern to find metric files (default: *_metrics.csv)
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_path}")
        return None
    
    logger.info(f"Merging metrics from: {input_path}")
    
    # Find all metric CSV files
    metric_files = list(input_path.glob(pattern))
    
    if not metric_files:
        logger.error(f"No metric files found matching pattern '{pattern}' in {input_path}")
        return None
    
    logger.info(f"Found {len(metric_files)} metric files:")
    for f in metric_files:
        logger.info(f"  - {f.name}")
    
    # Load all metrics
    dfs = {}
    for metric_file in sorted(metric_files):
        try:
            metric_name = metric_file.stem.replace('_metrics', '')
            df = pd.read_csv(metric_file, index_col=0)
            dfs[metric_name] = df
            logger.info(f"  Loaded {metric_name}: {df.shape[0]} rows, {df.shape[1]} columns")
        except Exception as e:
            logger.error(f"  Error loading {metric_file}: {e}")
    
    if not dfs:
        logger.error("No metric files could be loaded")
        return None
    
    # Merge on index (protein/sequence ID)
    logger.info(f"\nMerging {len(dfs)} dataframes by index (protein ID)...")
    
    try:
        # Start with first dataframe
        merged = list(dfs.values())[0].copy()
        
        # Concatenate remaining dataframes
        for df in list(dfs.values())[1:]:
            merged = pd.concat([merged, df], axis=1)
        
        logger.info(f"Merged shape: {merged.shape[0]} rows × {merged.shape[1]} columns")
        
        # Determine output path
        output_path = Path(output_file)
        if not output_path.is_absolute() and output_path.parent == Path("."):
            # It's just a filename, combine with input directory
            output_path = input_path / output_file
        
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        merged.to_csv(output_path)
        logger.info(f"\n✓ Merged metrics saved to: {output_path}")
        
        # Print summary
        logger.info(f"\nMerged metric columns ({merged.shape[1]} total):")
        for col in merged.columns:
            n_nonnull = merged[col].notna().sum()
            logger.info(f"  - {col:<40} ({n_nonnull}/{merged.shape[0]} values)")
        
        # Check for missing values
        missing = merged.isna().sum().sum()
        if missing > 0:
            logger.warning(f"\n⚠ Total missing values: {missing}")
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error merging dataframes: {e}")
        return None


def create_summary_stats(merged_df, output_dir='output'):
    """Create summary statistics for merged metrics"""
    output_path = Path(output_dir)
    
    logger.info("\nGenerating summary statistics...")
    
    try:
        # Select only numeric columns
        numeric_cols = merged_df.select_dtypes(include=['number']).columns
        
        stats = merged_df[numeric_cols].describe().T
        stats_file = output_path / 'metrics_summary_statistics.csv'
        stats.to_csv(stats_file)
        
        logger.info(f"Summary statistics saved to: {stats_file}")
        
        # Print to log
        logger.info("\nMetric Summary Statistics:")
        logger.info(stats.to_string())
        
        return stats_file
    
    except Exception as e:
        logger.warning(f"Could not generate summary statistics: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Merge protein metric CSV files into a single combined CSV"
    )
    parser.add_argument('--input_dir', default='output', help='Directory with metric CSV files')
    parser.add_argument('--output_file', default='all_metrics_combined.csv', help='Output CSV filename')
    parser.add_argument('--pattern', default='*_metrics.csv', help='Glob pattern to find metric files')
    parser.add_argument('--stats', action='store_true', help='Generate summary statistics')
    
    args = parser.parse_args()
    
    # Merge metrics
    output_file = merge_metrics(
        input_dir=args.input_dir,
        output_file=args.output_file,
        pattern=args.pattern
    )
    
    if output_file and args.stats:
        # Generate summary statistics
        merged_df = pd.read_csv(output_file, index_col=0)
        create_summary_stats(merged_df, args.input_dir)


if __name__ == '__main__':
    main()
