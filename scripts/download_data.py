#!/usr/bin/env python3
"""Download datasets for the hybrid DFT→GNN→QNN pipeline."""

import sys
import os
import json
import hashlib
import requests
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import argparse
import logging
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dft_hybrid.data.mptrj import MPtrjDataset
from dft_hybrid.data.ocp_lmdb import OCPLMDBDataset
from dft_hybrid.data.mp_api import MaterialsProjectDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DownloadManager:
    """Manages file downloads with resume and checksum verification."""
    
    def __init__(self, data_dir: Path = Path("data")):
        """Initialize download manager.
        
        Args:
            data_dir: Base directory for downloaded data
        """
        self.data_dir = data_dir
        self.data_dir.mkdir(exist_ok=True)
    
    def download_file(self, 
                     url: str, 
                     output_path: Path, 
                     expected_checksum: Optional[str] = None,
                     chunk_size: int = 8192,
                     resume: bool = True) -> bool:
        """Download a file with resume and checksum verification.
        
        Args:
            url: URL to download from
            output_path: Local path to save file
            expected_checksum: Expected SHA256 checksum (optional)
            chunk_size: Chunk size for download
            resume: Whether to resume partial downloads
            
        Returns:
            True if download successful, False otherwise
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if file already exists and is valid
        if output_path.exists():
            if expected_checksum:
                if self._verify_checksum(output_path, expected_checksum):
                    logger.info(f"File already exists and checksum verified: {output_path}")
                    return True
                else:
                    logger.warning(f"Existing file checksum mismatch, re-downloading: {output_path}")
                    output_path.unlink()
            else:
                logger.info(f"File already exists: {output_path}")
                return True
        
        # Prepare for download
        headers = {}
        start_byte = 0
        
        if resume and output_path.exists():
            start_byte = output_path.stat().st_size
            headers['Range'] = f'bytes={start_byte}-'
            logger.info(f"Resuming download from byte {start_byte}")
        
        try:
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()
            
            # Handle partial content (resume)
            if response.status_code == 206:  # Partial Content
                mode = 'ab'
            else:
                mode = 'wb'
                start_byte = 0
            
            total_size = int(response.headers.get('content-length', 0))
            if start_byte > 0:
                total_size += start_byte
            
            logger.info(f"Downloading {url} to {output_path}")
            if total_size > 0:
                logger.info(f"Total size: {total_size / (1024**3):.2f} GB")
            
            # Download with progress
            downloaded = start_byte
            start_time = time.time()
            
            with open(output_path, mode) as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Progress update every 100MB
                        if downloaded % (100 * 1024 * 1024) == 0:
                            elapsed = time.time() - start_time
                            speed = downloaded / elapsed / (1024**2)  # MB/s
                            logger.info(f"Downloaded: {downloaded / (1024**3):.2f} GB, Speed: {speed:.2f} MB/s")
            
            logger.info(f"Download completed: {output_path}")
            
            # Verify checksum if provided
            if expected_checksum:
                if self._verify_checksum(output_path, expected_checksum):
                    logger.info("Checksum verification passed")
                    return True
                else:
                    logger.error("Checksum verification failed")
                    output_path.unlink()
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False
    
    def _verify_checksum(self, file_path: Path, expected_checksum: str) -> bool:
        """Verify SHA256 checksum of a file.
        
        Args:
            file_path: Path to file
            expected_checksum: Expected SHA256 hash
            
        Returns:
            True if checksum matches, False otherwise
        """
        logger.info(f"Verifying checksum for {file_path}")
        
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        actual_checksum = sha256_hash.hexdigest()
        return actual_checksum == expected_checksum
    
    def download_multiple(self, 
                         downloads: List[Dict[str, Any]], 
                         max_workers: int = 4) -> List[bool]:
        """Download multiple files in parallel.
        
        Args:
            downloads: List of download info dicts
            max_workers: Maximum number of parallel downloads
            
        Returns:
            List of success status for each download
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_download = {
                executor.submit(
                    self.download_file,
                    download['url'],
                    download['output_path'],
                    download.get('checksum'),
                    download.get('chunk_size', 8192),
                    download.get('resume', True)
                ): download for download in downloads
            }
            
            for future in as_completed(future_to_download):
                download = future_to_download[future]
                try:
                    success = future.result()
                    results.append(success)
                    if success:
                        logger.info(f"✅ Downloaded: {download['output_path']}")
                    else:
                        logger.error(f"❌ Failed: {download['output_path']}")
                except Exception as e:
                    logger.error(f"❌ Exception downloading {download['url']}: {e}")
                    results.append(False)
        
        return results


def download_mptrj(data_dir: Path, resume: bool = True) -> bool:
    """Download MPtrj dataset from figshare.
    
    Args:
        data_dir: Base data directory
        resume: Whether to resume partial downloads
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("Downloading MPtrj dataset from figshare...")
    
    # Note: The MPtrj dataset URL needs to be updated with the current figshare link
    # This is a placeholder URL - users should check the current MPtrj publication
    logger.warning("⚠️  MPtrj dataset URL needs to be updated with current figshare link")
    logger.info("Please check the latest MPtrj publication for the current download URL")
    
    # Alternative: Create a sample MPtrj file for testing
    output_dir = data_dir / "mptrj"
    output_path = output_dir / "mptrj_sample.json"
    
    logger.info("Creating sample MPtrj data for testing...")
    
    # Create sample MPtrj data structure
    sample_data = {
        "sample_structure_001": {
            "trajectory": [
                {
                    "lattice": [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                    "positions": [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.0, 0.96, 0.0]],
                    "atomic_numbers": [8, 1, 1],
                    "energy": -76.0,
                    "forces": [[0.1, 0.0, 0.0], [-0.05, 0.0, 0.0], [0.0, -0.05, 0.0]],
                    "stress": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    "magmoms": [0.0, 0.0, 0.0],
                    "temperature": 300.0,
                    "pressure": 1.0
                },
                {
                    "lattice": [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                    "positions": [[0.0, 0.0, 0.0], [0.95, 0.1, 0.0], [0.1, 0.95, 0.0]],
                    "atomic_numbers": [8, 1, 1],
                    "energy": -75.9,
                    "forces": [[0.08, 0.0, 0.0], [-0.04, 0.0, 0.0], [0.0, -0.04, 0.0]],
                    "stress": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    "magmoms": [0.0, 0.0, 0.0],
                    "temperature": 310.0,
                    "pressure": 1.0
                }
            ]
        },
        "sample_structure_002": {
            "trajectory": [
                {
                    "lattice": [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                    "positions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.87, 0.0]],
                    "atomic_numbers": [8, 1, 1],
                    "energy": -75.8,
                    "forces": [[0.05, 0.0, 0.0], [-0.025, 0.0, 0.0], [0.0, -0.025, 0.0]],
                    "stress": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    "magmoms": [0.0, 0.0, 0.0],
                    "temperature": 300.0,
                    "pressure": 1.0
                }
            ]
        }
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    logger.info(f"✅ Created sample MPtrj data at {output_path}")
    
    # Test the sample data
    try:
        logger.info("Testing sample data with dataset loader...")
        dataset = MPtrjDataset(output_path, max_atoms=10)
        
        success = True
        
        logger.info(f"✅ MPtrj sample dataset created at {output_path}")
        
        # Print dataset statistics
        try:
            logger.info("Computing dataset statistics...")
            stats = dataset.get_statistics()
            logger.info("📊 MPtrj Dataset Statistics:")
            logger.info(f"  Total entries: {stats['n_entries']:,}")
            logger.info(f"  Unique structures: {stats['n_unique_structures']:,}")
            logger.info(f"  Atoms range: {stats['n_atoms']['min']} - {stats['n_atoms']['max']}")
            logger.info(f"  Energy range: {stats['energy']['min']:.3f} - {stats['energy']['max']:.3f} eV")
            logger.info(f"  Volume range: {stats['volume']['min']:.1f} - {stats['volume']['max']:.1f} Å³")
            logger.info(f"  Force magnitude: {stats['forces']['mean_magnitude']:.3f} ± {stats['forces']['std_magnitude']:.3f} eV/Å")
            
        except Exception as e:
            logger.warning(f"Could not compute statistics: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to create sample MPtrj dataset: {e}")
        return False


def download_ocp_dataset(data_dir: Path, 
                        dataset_name: str, 
                        resume: bool = True) -> bool:
    """Download OC20/OC22 dataset LMDB shards.
    
    Args:
        data_dir: Base data directory
        dataset_name: "oc20" or "oc22"
        resume: Whether to resume partial downloads
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Downloading {dataset_name.upper()} dataset...")
    
    # Note: The Open Catalyst Project datasets require special download procedures
    # The URLs and access methods change frequently
    logger.warning("⚠️  Open Catalyst Project datasets require special download procedures")
    logger.info("Please refer to the official Open Catalyst Project documentation:")
    logger.info("  - OC20: https://github.com/Open-Catalyst-Project/ocp")
    logger.info("  - OC22: https://github.com/Open-Catalyst-Project/ocp")
    logger.info("")
    logger.info("Typical download process:")
    logger.info("  1. Clone the OCP repository")
    logger.info("  2. Run their download_data.py script")
    logger.info("  3. Use appropriate flags for dataset size and task type")
    logger.info("")
    logger.info("Example commands:")
    logger.info("  # For OC20 S2EF task (small subset)")
    logger.info("  python download_data.py --task s2ef --split 200k --get-edges")
    logger.info("  # For OC20 IS2RE task")
    logger.info("  python download_data.py --task is2re")
    
    # Create a sample directory structure for demonstration
    output_dir = data_dir / "ocp" / dataset_name.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a README file with instructions
    readme_path = output_dir / "README.md"
    readme_content = f"""# {dataset_name.upper()} Dataset Download Instructions

This directory is intended for {dataset_name.upper()} dataset files.

## Download Instructions

1. **Clone the Open Catalyst Project repository:**
   ```bash
   git clone https://github.com/Open-Catalyst-Project/ocp.git
   cd ocp
   ```

2. **Install dependencies:**
   ```bash
   pip install -e .
   ```

3. **Download the dataset:**
   ```bash
   # For S2EF task (Structure to Energy and Forces)
   python scripts/download_data.py --task s2ef --split 200k --get-edges --num-workers 4
   
   # For IS2RE task (Initial Structure to Relaxed Energy)  
   python scripts/download_data.py --task is2re
   
   # For IS2RS task (Initial Structure to Relaxed Structure)
   python scripts/download_data.py --task is2rs
   ```

4. **Copy the downloaded LMDB files to this directory:**
   ```bash
   cp -r /path/to/ocp/data/{dataset_name.lower()}/* {output_dir}/
   ```

## Dataset Information

- **{dataset_name.upper()} S2EF**: Structure to Energy and Forces task
- **{dataset_name.upper()} IS2RE**: Initial Structure to Relaxed Energy task  
- **{dataset_name.upper()} IS2RS**: Initial Structure to Relaxed Structure task

## File Structure

After download, this directory should contain:
```
{output_dir}/
├── train.lmdb
├── val_id.lmdb
├── val_ood_ads.lmdb
├── val_ood_cat.lmdb
└── test.lmdb
```

## Usage with Dataset Loader

Once downloaded, you can use the dataset with:
```python
from dft_hybrid.data.ocp_lmdb import OCPLMDBDataset

dataset = OCPLMDBDataset("{output_dir}", dataset_type="s2ef")
```

For more information, visit: https://github.com/Open-Catalyst-Project/ocp
"""
    
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    logger.info(f"✅ Created {dataset_name.upper()} download instructions at {readme_path}")
    logger.info(f"📁 Directory structure prepared at {output_dir}")
    
    # Create a sample LMDB structure for testing (empty)
    sample_lmdb_dir = output_dir / "sample"
    sample_lmdb_dir.mkdir(exist_ok=True)
    
    # Create a simple test script
    test_script_path = output_dir / "test_download.py"
    test_script_content = f'''#!/usr/bin/env python3
"""Test script to verify {dataset_name.upper()} dataset download."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

try:
    from dft_hybrid.data.ocp_lmdb import OCPLMDBDataset
    
    # Test with sample directory (will fail until real data is downloaded)
    dataset_path = Path(__file__).parent
    print(f"Testing {dataset_name.upper()} dataset at: {{dataset_path}}")
    
    # This will fail until real LMDB files are present
    try:
        dataset = OCPLMDBDataset(dataset_path, dataset_type="s2ef")
        print(f"✅ {dataset_name.upper()} dataset loaded successfully!")
        print(f"   Entries: {{len(dataset)}}")
        
        stats = dataset.get_statistics()
        print(f"📊 Dataset statistics:")
        print(f"   Entries: {{stats['n_entries']:,}}")
        print(f"   Atoms range: {{stats['n_atoms']['min']}} - {{stats['n_atoms']['max']}}")
        
    except Exception as e:
        print(f"❌ {dataset_name.upper()} dataset not found or invalid: {{e}}")
        print("Please download the dataset using the instructions in README.md")
        
except ImportError as e:
    print(f"❌ Could not import dataset loader: {{e}}")
'''
    
    with open(test_script_path, 'w') as f:
        f.write(test_script_content)
    
    os.chmod(test_script_path, 0o755)
    
    logger.info(f"✅ Created test script at {test_script_path}")
    logger.info(f"📝 Instructions saved to {readme_path}")
    
    return True


def download_mp_data(data_dir: Path, 
                    api_key: Optional[str] = None,
                    mp_ids: Optional[List[str]] = None,
                    max_entries: int = 1000,
                    include_charge_density: bool = False) -> bool:
    """Download Materials Project data via API.
    
    Args:
        data_dir: Base data directory
        api_key: Materials Project API key
        mp_ids: List of specific MP IDs to download (optional)
        max_entries: Maximum number of entries to download
        include_charge_density: Whether to include charge density data
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("Downloading Materials Project data via API...")
    
    # Get API key
    if api_key is None:
        api_key = os.getenv('MP_API_KEY')
        if api_key is None:
            logger.error("MP_API_KEY environment variable not set")
            logger.error("Get your API key from: https://materialsproject.org/api")
            return False
    
    output_dir = data_dir / "mp"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create dataset
        dataset = MaterialsProjectDataset(
            api_key=api_key,
            cache_dir=output_dir,
            include_charge_density=include_charge_density
        )
        
        # Define search criteria
        if mp_ids:
            logger.info(f"Fetching specific MP IDs: {mp_ids}")
            entries = dataset.fetch_structures(
                criteria={"material_id": {"$in": mp_ids}},
                max_entries=len(mp_ids)
            )
        else:
            logger.info(f"Fetching up to {max_entries} common materials...")
            
            # Fetch common materials (using updated API field names)
            criteria_list = [
                {"num_elements": 2, "is_stable": True},  # Binary compounds
                {"num_elements": 3, "is_stable": True},  # Ternary compounds
                {"elements": ["Fe", "O"], "num_elements": 2},  # Iron oxides
                {"elements": ["Ti", "O"], "num_elements": 2},  # Titanium oxides
                {"is_stable": True},  # Stable materials (magnetic filter removed for now)
            ]
            
            all_entries = []
            entries_per_criteria = max_entries // len(criteria_list)
            
            for criteria in criteria_list:
                try:
                    entries = dataset.fetch_structures(criteria, max_entries=entries_per_criteria)
                    all_entries.extend(entries)
                    logger.info(f"Fetched {len(entries)} entries for criteria: {criteria}")
                except Exception as e:
                    logger.warning(f"Failed to fetch for criteria {criteria}: {e}")
                    continue
            
            entries = all_entries
        
        if not entries:
            logger.error("No entries fetched from Materials Project API")
            return False
        
        # Add entries to dataset
        dataset.add_entries(entries)
        logger.info(f"✅ Downloaded {len(entries)} Materials Project entries")
        
        # Print dataset statistics
        stats = dataset.get_statistics()
        logger.info("📊 Materials Project Dataset Statistics:")
        logger.info(f"  Total entries: {stats['n_entries']:,}")
        logger.info(f"  Atoms range: {stats['n_atoms']['min']} - {stats['n_atoms']['max']}")
        
        if 'energy' in stats:
            logger.info(f"  Formation energy: {stats['energy']['min']:.3f} - {stats['energy']['max']:.3f} eV/atom")
        
        if 'band_gap' in stats:
            logger.info(f"  Band gap: {stats['band_gap']['min']:.3f} - {stats['band_gap']['max']:.3f} eV")
        
        logger.info(f"  Magnetic materials: {stats['n_magnetic']}")
        logger.info(f"  Stable materials: {stats['n_stable']}")
        logger.info(f"  Metallic materials: {stats['n_metal']}")
        
        if include_charge_density:
            logger.info("  Charge density data: Included")
        
        return True
        
    except Exception as e:
        logger.error(f"Materials Project download failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function for download_data script."""
    parser = argparse.ArgumentParser(
        description="Download datasets for the hybrid DFT→GNN→QNN pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download MPtrj dataset
  python scripts/download_data.py mptrj

  # Download OC20 dataset with resume
  python scripts/download_data.py oc20 --resume

  # Download OC22 dataset
  python scripts/download_data.py oc22

  # Download Materials Project data
  python scripts/download_data.py mp --max-entries 500 --include-charge-density

  # Download specific MP IDs
  python scripts/download_data.py mp --mp-ids mp-149,mp-3903,mp-1143
        """
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # MPtrj subcommand
    mptrj_parser = subparsers.add_parser('mptrj', help='Download MPtrj dataset from figshare')
    mptrj_parser.add_argument('--data-dir', type=str, default='data',
                            help='Base data directory (default: data)')
    mptrj_parser.add_argument('--resume', action='store_true', default=True,
                            help='Resume partial downloads (default: True)')
    
    # OC20 subcommand
    oc20_parser = subparsers.add_parser('oc20', help='Download OC20 dataset')
    oc20_parser.add_argument('--data-dir', type=str, default='data',
                           help='Base data directory (default: data)')
    oc20_parser.add_argument('--resume', action='store_true', default=True,
                           help='Resume partial downloads (default: True)')
    
    # OC22 subcommand
    oc22_parser = subparsers.add_parser('oc22', help='Download OC22 dataset')
    oc22_parser.add_argument('--data-dir', type=str, default='data',
                           help='Base data directory (default: data)')
    oc22_parser.add_argument('--resume', action='store_true', default=True,
                           help='Resume partial downloads (default: True)')
    
    # Materials Project subcommand
    mp_parser = subparsers.add_parser('mp', help='Download Materials Project data')
    mp_parser.add_argument('--data-dir', type=str, default='data',
                         help='Base data directory (default: data)')
    mp_parser.add_argument('--api-key', type=str,
                         help='Materials Project API key (or set MP_API_KEY env var)')
    mp_parser.add_argument('--mp-ids', type=str,
                         help='Comma-separated list of MP IDs to download')
    mp_parser.add_argument('--max-entries', type=int, default=1000,
                         help='Maximum number of entries to download (default: 1000)')
    mp_parser.add_argument('--include-charge-density', action='store_true',
                         help='Include charge density data (slower, larger files)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Convert data directory to Path
    data_dir = Path(args.data_dir)
    
    # Execute subcommand
    success = False
    
    if args.command == 'mptrj':
        success = download_mptrj(data_dir, resume=args.resume)
    
    elif args.command in ['oc20', 'oc22']:
        success = download_ocp_dataset(data_dir, args.command, resume=args.resume)
    
    elif args.command == 'mp':
        mp_ids = None
        if args.mp_ids:
            mp_ids = [mp_id.strip() for mp_id in args.mp_ids.split(',')]
        
        success = download_mp_data(
            data_dir=data_dir,
            api_key=args.api_key,
            mp_ids=mp_ids,
            max_entries=args.max_entries,
            include_charge_density=args.include_charge_density
        )
    
    if success:
        logger.info(f"✅ {args.command.upper()} download completed successfully!")
    else:
        logger.error(f"❌ {args.command.upper()} download failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
