#!/usr/bin/env python3
"""
Script to generate training set for metabolic modeling.

Usage:
    python generate_training_set.py [OPTIONS]
    ./generate_training_set.py --cobraname iML1515 --mediumname glucose_exp --method EXP
"""

import os
import sys
import argparse
import warnings
import numpy as np
import subprocess
from pathlib import Path
import logging

def setup_logging(verbose=False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def is_running_in_colab():
    """Check if the script is running in Google Colab environment"""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Generate training set for metabolic modeling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --cobraname iML1515_ec_duplicated --mediumname iML1515_ec_EXP
  %(prog)s --method pFBA --mediumbound LB --reduce --seed 42
  %(prog)s --colab --colab-repo-path "/content/drive/My Drive/Github/amn_release/"
        """
    )
    
    # Training set parameters
    training_group = parser.add_argument_group('Training Set Parameters')
    training_group.add_argument('--seed', type=int, default=10,
                        help='Random seed for reproducibility (default: 10)')
    training_group.add_argument('--cobraname', type=str, default='iML1515_ec_duplicated',
                        help='Name of the COBRA model (default: iML1515_ec_duplicated)')
    training_group.add_argument('--mediumbound', type=str, default='UB',
                        choices=['UB', 'LB', 'BOTH'],
                        help='Medium bound type: UB (Upper Bound), LB (Lower Bound), BOTH (default: UB)')
    training_group.add_argument('--mediumname', type=str, default='iML1515_ec_EXP',
                        help='Name of experimental file (default: iML1515_ec_EXP)')
    training_group.add_argument('--method', type=str, default='EXP',
                        choices=['FBA', 'pFBA', 'EXP'],
                        help='Method for training set generation: FBA, pFBA, or EXP (default: EXP)')
    training_group.add_argument('--reduce', action='store_true',
                        help='Reduce the model (default: False)')
    training_group.add_argument('--mediumsize', type=int, default=38,
                        help='Medium size parameter (default: 38)')
    training_group.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    # Path parameters
    path_group = parser.add_argument_group('Path Parameters')
    path_group.add_argument('--base-dir', type=str, default=None,
                        help='Base directory for the repository (default: current directory)')
    path_group.add_argument('--input-dir', type=str, default='Dataset_input',
                        help='Input directory name (default: Dataset_input)')
    path_group.add_argument('--output-dir', type=str, default='Dataset_model',
                        help='Output directory name (default: Dataset_model)')
    path_group.add_argument('--output-filename', type=str, default=None,
                        help='Custom output filename (default: <mediumname>_<mediumbound>)')
    
    # Colab specific parameters
    colab_group = parser.add_argument_group('Colab Specific Parameters')
    colab_group.add_argument('--colab', action='store_true',
                        help='Force Colab mode (auto-detected by default)')
    colab_group.add_argument('--no-colab', action='store_true',
                        help='Force disable Colab mode')
    colab_group.add_argument('--colab-repo-path', type=str,
                        default='/content/drive/My Drive/Github/amn_release/',
                        help='Repository path in Google Drive (Colab only)')
    colab_group.add_argument('--skip-drive-mount', action='store_true',
                        help='Skip Google Drive mounting (Colab only)')
    colab_group.add_argument('--skip-env-update', action='store_true',
                        help='Skip conda environment update (Colab only)')
    colab_group.add_argument('--force-remount', action='store_true',
                        help='Force remount Google Drive (Colab only)')
    
    # Advanced parameters
    advanced_group = parser.add_argument_group('Advanced Parameters')
    advanced_group.add_argument('--font', type=str, default=None,
                        help='Font to use (default: Liberation Sans in Colab, arial locally)')
    advanced_group.add_argument('--no-warnings', action='store_true',
                        help='Suppress all warnings')
    
    # Execution control
    exec_group = parser.add_argument_group('Execution Control')
    exec_group.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without executing')
    exec_group.add_argument('--config-file', type=str,
                       help='Load configuration from JSON/YAML file')
    
    return parser.parse_args()

def load_config_file(config_file):
    """Load configuration from JSON or YAML file"""
    try:
        if config_file.endswith('.json'):
            import json
            with open(config_file, 'r') as f:
                return json.load(f)
        elif config_file.endswith(('.yaml', '.yml')):
            import yaml
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        else:
            logger.error(f"Unsupported config file format: {config_file}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        sys.exit(1)

def setup_colab(args, logger):
    """Setup Google Colab environment"""
    logger.info("Setting up Google Colab environment...")
    
    # Check condacolab
    try:
        import condacolab
        condacolab.check()
    except ImportError:
        logger.info("condacolab not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "condacolab"], check=True)
        import condacolab
        condacolab.check()
    
    # Mount Google Drive
    if not args.skip_drive_mount:
        from google.colab import drive
        force_remount = args.force_remount or False
        logger.info(f"Mounting Google Drive (force_remount={force_remount})...")
        drive.mount('/content/drive', force_remount=force_remount)
    
    # Set base directory
    base_dir = args.colab_repo_path
    logger.info(f"Setting base directory to: {base_dir}")
    
    # Change to repository directory
    os.chdir(base_dir)
    
    # Update conda environment
    if not args.skip_env_update:
        logger.info("Updating conda environment...")
        subprocess.run(["mamba", "env", "update", "-n", "base", "-f", "environment_amn_light.yml"], check=True)
    
    # Set font for Colab
    font = args.font if args.font else 'Liberation Sans'
    
    return base_dir, font

def setup_local(args, logger):
    """Setup local environment"""
    logger.info("Setting up local environment...")
    
    # Set base directory
    if args.base_dir:
        base_dir = args.base_dir
        logger.info(f"Using specified base directory: {base_dir}")
    else:
        base_dir = './'
        logger.info("Using current directory as base")
    
    # Change to base directory
    os.chdir(base_dir)
    
    # Set font for local
    font = args.font if args.font else 'arial'
    
    return base_dir, font

def build_paths(args, base_dir):
    """Build all required file paths"""
    input_dir = os.path.join(base_dir, args.input_dir)
    output_dir = os.path.join(base_dir, args.output_dir)
    
    # Input files
    cobrafile = os.path.join(input_dir, args.cobraname)
    mediumfile = os.path.join(input_dir, args.mediumname)
    
    # Output file
    if args.output_filename:
        trainingfile = os.path.join(output_dir, args.output_filename)
    else:
        trainingfile = os.path.join(output_dir, f'{args.mediumname}_{args.mediumbound}')
    
    return {
        'input_dir': input_dir,
        'output_dir': output_dir,
        'cobrafile': cobrafile,
        'mediumfile': mediumfile,
        'trainingfile': trainingfile
    }

def validate_paths(paths, logger, dry_run=False):
    """Validate that required paths exist"""
    if not dry_run:
        # Check input directory
        if not os.path.exists(paths['input_dir']):
            logger.error(f"Input directory not found: {paths['input_dir']}")
            sys.exit(1)
        
        # Check if input files exist (warn if not, but continue)
        if not os.path.exists(paths['cobrafile']):
            logger.warning(f"Cobra file not found: {paths['cobrafile']}")
        
        if not os.path.exists(paths['mediumfile']):
            logger.warning(f"Medium file not found: {paths['mediumfile']}")
        
        # Create output directory if it doesn't exist
        os.makedirs(paths['output_dir'], exist_ok=True)

def print_config(args, paths, logger):
    """Print configuration summary"""
    logger.info("=" * 60)
    logger.info("CONFIGURATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Cobra Model: {args.cobraname}")
    logger.info(f"Medium Name: {args.mediumname}")
    logger.info(f"Medium Bound: {args.mediumbound}")
    logger.info(f"Method: {args.method}")
    logger.info(f"Reduce Model: {args.reduce}")
    logger.info(f"Medium Size: {args.mediumsize}")
    logger.info(f"Verbose: {args.verbose}")
    logger.info("-" * 60)
    logger.info(f"Cobra File: {paths['cobrafile']}")
    logger.info(f"Medium File: {paths['mediumfile']}")
    logger.info(f"Output File: {paths['trainingfile']}")
    logger.info("=" * 60)

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logger
    logger = setup_logging(args.verbose)
    
    # Load config file if specified
    if args.config_file:
        config = load_config_file(args.config_file)
        # Override arguments with config file values
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    # Determine if running in Colab
    if args.no_colab:
        RunningInCOLAB = False
    elif args.colab:
        RunningInCOLAB = True
    else:
        RunningInCOLAB = is_running_in_colab()
    
    # Setup warnings
    if args.no_warnings:
        warnings.filterwarnings('ignore')
    else:
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    # Setup environment
    if RunningInCOLAB:
        base_dir, font = setup_colab(args, logger)
    else:
        base_dir, font = setup_local(args, logger)
    
    # Build paths
    paths = build_paths(args, base_dir)
    
    # Print directory contents
    logger.info(f"Current directory: {os.getcwd()}")
    logger.info(f"Directory contents: {os.listdir(base_dir)}")
    
    # Validate paths
    validate_paths(paths, logger, args.dry_run)
    
    # Print configuration
    print_config(args, paths, logger)
    
    # Dry run mode
    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be made")
        return
    
    # Import required module
    try:
        sys.path.append(base_dir)
        from Library.Build_Dataset import TrainingSet
    except ImportError as e:
        logger.error(f"Error importing required modules: {e}")
        logger.error("Please ensure the required modules are installed and available.")
        sys.exit(1)
    
    # Set random seed
    np.random.seed(seed=args.seed)
    
    # Generate training set
    logger.info(f"\nGenerating training set...")
    try:
        parameter = TrainingSet(
            cobraname=paths['cobrafile'],
            mediumname=paths['mediumfile'],
            mediumbound=args.mediumbound,
            mediumsize=args.mediumsize,
            method=args.method,
            verbose=args.verbose
        )
    except Exception as e:
        logger.error(f"Error generating training set: {e}")
        sys.exit(1)
    
    # Save training set
    logger.info(f"Saving to: {paths['trainingfile']}")
    try:
        parameter.save(paths['trainingfile'], reduce=args.reduce)
        logger.info(f"Training set saved successfully")
    except Exception as e:
        logger.error(f"Error saving training set: {e}")
        sys.exit(1)
    
    # Verify saved file
    if args.verbose:
        logger.info("\nVerifying saved file...")
        try:
            parameter = TrainingSet()
            parameter.load(paths['trainingfile'])
            parameter.printout()
        except Exception as e:
            logger.warning(f"Error verifying saved file: {e}")
    
    logger.info(f"\nTraining set generation complete!")
    logger.info(f"Output saved at: {paths['trainingfile']}")

if __name__ == "__main__":
    main()