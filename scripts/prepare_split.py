#!/usr/bin/env python3
"""Script to prepare data splits for all dataset loaders."""

import sys
import os
import argparse
from pathlib import Path
from typing import Dict, Any
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dft_hybrid.data.mptrj import MPtrjDataset, create_mptrj_splits
from dft_hybrid.data.ocp_lmdb import OCPLMDBDataset, create_ocp_splits
from dft_hybrid.data.mp_api import MaterialsProjectDataset, create_mp_splits


def prepare_mptrj_splits(data_path: str, **kwargs):
    """Prepare splits for MPtrj dataset."""
    print(f"Preparing MPtrj splits from {data_path}")
    
    # Load dataset
    dataset = MPtrjDataset(
        data_path=data_path,
        max_atoms=kwargs.get('max_atoms'),
        include_magmoms=kwargs.get('include_magmoms', True)
    )
    
    # Create splits
    create_mptrj_splits(
        dataset,
        train_ratio=kwargs.get('train_ratio', 0.8),
        val_ratio=kwargs.get('val_ratio', 0.1),
        test_ratio=kwargs.get('test_ratio', 0.1),
        split_by=kwargs.get('split_by', 'structure')
    )
    
    # Print summary
    split_sizes = dataset.get_split_sizes()
    stats = dataset.get_statistics()
    
    print(f"MPtrj dataset prepared:")
    print(f"  Total entries: {len(dataset)}")
    print(f"  Split sizes: {split_sizes}")
    print(f"  Energy range: {stats['energy']['min']:.3f} to {stats['energy']['max']:.3f} eV")
    print(f"  Unique structures: {stats['n_unique_structures']}")
    
    return dataset


def prepare_ocp_splits(data_path: str, dataset_type: str = "s2ef", **kwargs):
    """Prepare splits for OCP dataset."""
    print(f"Preparing OCP {dataset_type} splits from {data_path}")
    
    # Load dataset
    dataset = OCPLMDBDataset(
        data_path=data_path,
        dataset_type=dataset_type,
        max_atoms=kwargs.get('max_atoms'),
        use_relaxed=kwargs.get('use_relaxed', True)
    )
    
    # Create splits
    create_ocp_splits(
        dataset,
        train_ratio=kwargs.get('train_ratio', 0.8),
        val_ratio=kwargs.get('val_ratio', 0.1),
        test_ratio=kwargs.get('test_ratio', 0.1)
    )
    
    # Print summary
    split_sizes = dataset.get_split_sizes()
    stats = dataset.get_statistics()
    
    print(f"OCP {dataset_type} dataset prepared:")
    print(f"  Total entries: {len(dataset)}")
    print(f"  Split sizes: {split_sizes}")
    print(f"  Atoms range: {stats['n_atoms']['min']} to {stats['n_atoms']['max']}")
    if 'energy' in stats:
        print(f"  Energy range: {stats['energy']['min']:.3f} to {stats['energy']['max']:.3f} eV")
    
    return dataset


def prepare_mp_splits(api_key: str = None, **kwargs):
    """Prepare splits for Materials Project dataset."""
    print("Preparing Materials Project splits")
    
    # Load dataset
    dataset = MaterialsProjectDataset(
        api_key=api_key,
        max_atoms=kwargs.get('max_atoms'),
        include_charge_density=kwargs.get('include_charge_density', False)
    )
    
    # Fetch data if needed
    if len(dataset) == 0:
        print("No cached data found. Fetching from Materials Project API...")
        
        # Fetch common materials
        criteria = kwargs.get('criteria', {
            "nelements": {"$lte": 3},
            "is_stable": True
        })
        
        max_entries = kwargs.get('max_entries', 1000)
        entries = dataset.fetch_structures(criteria, max_entries=max_entries)
        
        if entries:
            dataset.add_entries(entries)
            print(f"Fetched and cached {len(entries)} entries")
        else:
            print("No entries fetched from API")
            return dataset
    
    # Create splits
    create_mp_splits(
        dataset,
        train_ratio=kwargs.get('train_ratio', 0.8),
        val_ratio=kwargs.get('val_ratio', 0.1),
        test_ratio=kwargs.get('test_ratio', 0.1)
    )
    
    # Print summary
    split_sizes = dataset.get_split_sizes()
    stats = dataset.get_statistics()
    
    print(f"Materials Project dataset prepared:")
    print(f"  Total entries: {len(dataset)}")
    print(f"  Split sizes: {split_sizes}")
    print(f"  Atoms range: {stats['n_atoms']['min']} to {stats['n_atoms']['max']}")
    if 'energy' in stats:
        print(f"  Energy range: {stats['energy']['min']:.3f} to {stats['energy']['max']:.3f} eV/atom")
    if 'band_gap' in stats:
        print(f"  Band gap range: {stats['band_gap']['min']:.3f} to {stats['band_gap']['max']:.3f} eV")
    
    return dataset


def main():
    """Main function for prepare_split script."""
    parser = argparse.ArgumentParser(description="Prepare data splits for dataset loaders")
    
    # Dataset selection
    parser.add_argument('--dataset', choices=['mptrj', 'ocp', 'mp'], required=True,
                       help='Dataset type to prepare splits for')
    
    # Data paths
    parser.add_argument('--data-path', type=str,
                       help='Path to dataset (required for mptrj and ocp)')
    parser.add_argument('--dataset-type', choices=['s2ef', 'is2re', 'is2rs'], default='s2ef',
                       help='OCP dataset type (for ocp dataset)')
    
    # API configuration
    parser.add_argument('--api-key', type=str,
                       help='Materials Project API key (for mp dataset)')
    
    # Split configuration
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Training set ratio (default: 0.8)')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='Validation set ratio (default: 0.1)')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                       help='Test set ratio (default: 0.1)')
    
    # Filtering options
    parser.add_argument('--max-atoms', type=int,
                       help='Maximum number of atoms per structure')
    parser.add_argument('--max-entries', type=int, default=1000,
                       help='Maximum number of entries to fetch (for mp dataset)')
    
    # Dataset-specific options
    parser.add_argument('--include-magmoms', action='store_true', default=True,
                       help='Include magnetic moments (for mptrj dataset)')
    parser.add_argument('--use-relaxed', action='store_true', default=True,
                       help='Use relaxed structures (for ocp dataset)')
    parser.add_argument('--include-charge-density', action='store_true',
                       help='Include charge density data (for mp dataset)')
    parser.add_argument('--split-by', choices=['structure', 'step'], default='structure',
                       help='How to split mptrj data (by structure or step)')
    
    # Output options
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for split files')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.dataset in ['mptrj', 'ocp'] and not args.data_path:
        parser.error(f"--data-path is required for {args.dataset} dataset")
    
    if args.dataset == 'mp' and not args.api_key and not os.getenv('MP_API_KEY'):
        parser.error("--api-key is required for mp dataset (or set MP_API_KEY env var)")
    
    # Validate split ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        parser.error("Split ratios must sum to 1.0")
    
    # Prepare splits based on dataset type
    try:
        if args.dataset == 'mptrj':
            dataset = prepare_mptrj_splits(
                data_path=args.data_path,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                split_by=args.split_by,
                max_atoms=args.max_atoms,
                include_magmoms=args.include_magmoms
            )
        
        elif args.dataset == 'ocp':
            dataset = prepare_ocp_splits(
                data_path=args.data_path,
                dataset_type=args.dataset_type,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                max_atoms=args.max_atoms,
                use_relaxed=args.use_relaxed
            )
        
        elif args.dataset == 'mp':
            # Get API key
            api_key = args.api_key or os.getenv('MP_API_KEY')
            
            dataset = prepare_mp_splits(
                api_key=api_key,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                max_atoms=args.max_atoms,
                max_entries=args.max_entries,
                include_charge_density=args.include_charge_density
            )
        
        print(f"\n✅ Successfully prepared {args.dataset} dataset splits!")
        
        # Save configuration if output directory specified
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            config = {
                'dataset': args.dataset,
                'data_path': args.data_path,
                'dataset_type': args.dataset_type,
                'train_ratio': args.train_ratio,
                'val_ratio': args.val_ratio,
                'test_ratio': args.test_ratio,
                'max_atoms': args.max_atoms,
                'split_sizes': dataset.get_split_sizes(),
                'statistics': dataset.get_statistics()
            }
            
            config_file = output_dir / f"{args.dataset}_config.json"
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"Configuration saved to {config_file}")
    
    except Exception as e:
        print(f"❌ Error preparing {args.dataset} dataset: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()



