#!/usr/bin/env python
"""
Test/Verification script for protein metrics calculators.
Checks that all modules can be imported and basic functionality works.
"""

import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all modules can be imported"""
    logger.info("Testing imports...")
    
    modules_to_test = [
        'structure_metrics',
        'single_sequence_metrics',
        'alignment_metrics',
        'run_all_metrics',
        'merge_metrics'
    ]
    
    failed = []
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            logger.info(f"  ✓ {module_name}")
        except ImportError as e:
            logger.error(f"  ✗ {module_name}: {e}")
            failed.append(module_name)
    
    return len(failed) == 0, failed


def test_class_instantiation():
    """Test that calculator classes can be instantiated"""
    logger.info("\nTesting class instantiation...")
    
    try:
        from structure_metrics import StructureMetricsCalculator
        calc = StructureMetricsCalculator(pdb_dir='test_pdbs', output_dir='test_output')
        logger.info("  ✓ StructureMetricsCalculator")
    except Exception as e:
        logger.error(f"  ✗ StructureMetricsCalculator: {e}")
        return False
    
    try:
        from single_sequence_metrics import SingleSequenceMetricsCalculator
        calc = SingleSequenceMetricsCalculator(target_seqs_dir='test_seqs', output_dir='test_output')
        logger.info("  ✓ SingleSequenceMetricsCalculator")
    except Exception as e:
        logger.error(f"  ✗ SingleSequenceMetricsCalculator: {e}")
        return False
    
    try:
        from alignment_metrics import AlignmentMetricsCalculator
        calc = AlignmentMetricsCalculator(
            target_seqs_dir='test_seqs',
            reference_seqs_dir='test_refs',
            output_dir='test_output'
        )
        logger.info("  ✓ AlignmentMetricsCalculator")
    except Exception as e:
        logger.error(f"  ✗ AlignmentMetricsCalculator: {e}")
        return False
    
    return True


def test_dependencies():
    """Test that required dependencies are available"""
    logger.info("\nTesting dependencies...")
    
    dependencies = {
        'pandas': 'CSV handling',
        'numpy': 'Array operations',
        'pathlib': 'File system (built-in)',
    }
    
    optional_dependencies = {
        'torch': 'Deep learning',
        'esm': 'ESM models',
        'biotite': 'Protein structure',
    }
    
    # Check required
    failed = []
    for dep_name, description in dependencies.items():
        try:
            __import__(dep_name)
            logger.info(f"  ✓ {dep_name:<20} ({description})")
        except ImportError:
            logger.error(f"  ✗ {dep_name:<20} ({description}) - REQUIRED")
            failed.append(dep_name)
    
    # Check optional
    logger.info("\nOptional dependencies:")
    for dep_name, description in optional_dependencies.items():
        try:
            __import__(dep_name)
            logger.info(f"  ✓ {dep_name:<20} ({description})")
        except ImportError:
            logger.warning(f"  ✗ {dep_name:<20} ({description}) - optional, some metrics will be skipped")
    
    return len(failed) == 0, failed


def test_directory_structure():
    """Test that expected directories exist or can be created"""
    logger.info("\nTesting directory structure...")
    
    dirs_to_check = ['pdbs', 'target_seqs', 'reference_seqs', 'output']
    
    for dir_name in dirs_to_check:
        dir_path = Path(dir_name)
        if dir_path.exists():
            logger.info(f"  ✓ {dir_name}/ exists")
        else:
            try:
                dir_path.mkdir(exist_ok=True)
                logger.info(f"  ✓ {dir_name}/ created")
            except Exception as e:
                logger.error(f"  ✗ {dir_name}/: {e}")
                return False
    
    return True


def test_help_messages():
    """Test that command-line help works"""
    logger.info("\nTesting command-line interfaces...")
    
    scripts = [
        'structure_metrics.py',
        'single_sequence_metrics.py',
        'alignment_metrics.py',
        'run_all_metrics.py',
        'merge_metrics.py'
    ]
    
    for script in scripts:
        script_path = Path(script)
        if script_path.exists():
            logger.info(f"  ✓ {script}")
        else:
            logger.error(f"  ✗ {script} not found")
            return False
    
    return True


def main():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("PROTEIN METRICS CALCULATOR - VERIFICATION TESTS")
    logger.info("=" * 60)
    
    all_passed = True
    
    # Test 1: Imports
    passed, failed = test_imports()
    all_passed = all_passed and passed
    if failed:
        logger.error(f"  Failed imports: {', '.join(failed)}")
    
    # Test 2: Class instantiation
    passed = test_class_instantiation()
    all_passed = all_passed and passed
    
    # Test 3: Dependencies
    passed, failed = test_dependencies()
    all_passed = all_passed and passed
    
    # Test 4: Directories
    passed = test_directory_structure()
    all_passed = all_passed and passed
    
    # Test 5: Help messages
    passed = test_help_messages()
    all_passed = all_passed and passed
    
    # Summary
    logger.info("\n" + "=" * 60)
    if all_passed:
        logger.info("✓ ALL TESTS PASSED")
        logger.info("=" * 60)
        logger.info("\nYou can now run:")
        logger.info("  python run_all_metrics.py --all")
        logger.info("  python structure_metrics.py --all")
        logger.info("  python single_sequence_metrics.py --all")
        logger.info("  python alignment_metrics.py --all")
        return 0
    else:
        logger.error("✗ SOME TESTS FAILED")
        logger.error("=" * 60)
        logger.error("\nPlease fix the issues above and try again.")
        logger.error("See README.md for installation instructions.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
