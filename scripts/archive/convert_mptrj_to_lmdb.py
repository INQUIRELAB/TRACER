#!/usr/bin/env python3
"""Convert MPtrj JSON/NPZ to LMDB shards with streaming and compression."""

import os
import sys
import json
import numpy as np
import lmdb
import msgpack
import lz4.frame
import argparse
from pathlib import Path
from typing import Dict, Any, Iterator, List, Tuple
import gc
import psutil
import torch
import random
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure safe multiprocessing
from dft_hybrid.data.io import set_safe_mp
set_safe_mp()


def check_memory_usage(max_gb: float = 50.0) -> bool:
    """Check if memory usage is within limits."""
    memory_gb = psutil.virtual_memory().used / (1024**3)
    return memory_gb < max_gb


def get_gpu_device():
    """Get available GPU device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return device
    else:
        print("CUDA not available, using CPU (as per user request, this will raise an error later if GPU is strictly required)")
        return None


def convert_pymatgen_structure(structure_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert pymatgen structure to our expected format."""
    try:
        # Extract lattice vectors
        lattice_matrix = structure_data['lattice']['matrix']
        
        # Extract atomic positions and species
        sites = structure_data['sites']
        atomic_numbers = []
        positions = []
        
        for site in sites:
            # Get atomic number from element symbol
            element_symbol = site['species'][0]['element']
            atomic_num = get_atomic_number(element_symbol)
            atomic_numbers.append(atomic_num)
            
            # Get position
            xyz = site['xyz']
            positions.append(xyz)
        
        # Convert to numpy arrays
        atomic_numbers = np.array(atomic_numbers, dtype=np.int16)
        positions = np.array(positions, dtype=np.float32)
        lattice_matrix = np.array(lattice_matrix, dtype=np.float32)
        
        # Create dummy energy, forces, stress (these would come from DFT calculations)
        energy = 0.0  # Placeholder
        forces = np.zeros_like(positions, dtype=np.float32)  # Placeholder
        stress = np.zeros((3, 3), dtype=np.float32)  # Placeholder
        magmom = np.zeros(len(atomic_numbers), dtype=np.float32)  # Placeholder
        
        return {
            'atomic_numbers': atomic_numbers,
            'coords': positions,
            'lattice_vectors': lattice_matrix,
            'energy': energy,
            'forces': forces,
            'stress': stress,
            'magmom': magmom
        }
    except Exception as e:
        print(f"Error converting pymatgen structure: {e}")
        return None


def get_atomic_number(element_symbol: str) -> int:
    """Get atomic number from element symbol."""
    element_map = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
        'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
        'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
        'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
        'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42,
        'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
        'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58,
        'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66,
        'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74,
        'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82,
        'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
        'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98,
        'Es': 99, 'Fm': 100
    }
    return element_map.get(element_symbol, 1)  # Default to H if not found


def read_mptrj_streaming(data_path: Path) -> Iterator[Dict[str, Any]]:
    """Read MPtrj data in streaming mode to minimize memory usage."""
    print(f"Streaming MPtrj data from {data_path}")

    if data_path.suffix == '.json':
        # Use ijson for streaming JSON parsing
        try:
            import ijson
        except ImportError:
            raise ImportError("ijson required for streaming JSON parsing. Install with: pip install ijson")

        with open(data_path, 'rb') as f:
            # The JSON structure is: {"mp-1005792": {"mp-1012897-0-0": {"structure": {...}}}}
            # We need to handle the 3-level nested structure
            try:
                # Parse as a dictionary of material IDs
                parser = ijson.kvitems(f, '')
                for material_id, material_data in parser:
                    # material_data is another dictionary with trajectory IDs as keys
                    if isinstance(material_data, dict):
                        for trajectory_id, trajectory_data in material_data.items():
                            if isinstance(trajectory_data, dict) and 'structure' in trajectory_data:
                                # Extract the structure data
                                structure = trajectory_data['structure']
                                # Convert pymatgen structure to our format
                                processed_structure = convert_pymatgen_structure(structure)
                                if processed_structure:
                                    yield processed_structure
                            elif isinstance(trajectory_data, dict):
                                # If it's a structure directly
                                processed_structure = convert_pymatgen_structure(trajectory_data)
                                if processed_structure:
                                    yield processed_structure
            except Exception as e:
                print(f"Error parsing JSON: {e}")
                # Fallback: try to parse as a simpler structure
                f.seek(0)
                try:
                    parser = ijson.items(f, 'item')
                    for item in parser:
                        processed_structure = convert_pymatgen_structure(item)
                        if processed_structure:
                            yield processed_structure
                except Exception as e2:
                    print(f"Fallback parsing also failed: {e2}")
                    return
    elif data_path.suffix == '.npz':
        # Use numpy.load with mmap_mode for NPZ files
        data = np.load(data_path, mmap_mode='r')
        for key in data.files:
            yield data[key].item()  # Assuming each key stores a dictionary entry
    else:
        raise ValueError(f"Unsupported file type: {data_path.suffix}")


def process_entry(entry: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Process a single entry, ensuring dtypes and moving to GPU."""
    processed = {}

    # Atomic numbers (Z)
    atomic_numbers_gpu = torch.tensor(entry['atomic_numbers'], dtype=torch.int16, device=device) if 'atomic_numbers' in entry else torch.tensor(entry['Z'], dtype=torch.int16, device=device)

    # Positions
    positions_gpu = torch.tensor(entry['coords'], dtype=torch.float32, device=device) if 'coords' in entry else torch.tensor(entry['pos'], dtype=torch.float32, device=device)

    # Cell vectors
    cell_gpu = torch.tensor(entry['lattice_vectors'], dtype=torch.float32, device=device) if 'lattice_vectors' in entry else torch.tensor(entry['cell'], dtype=torch.float32, device=device)

    # Energy
    energy_gpu = torch.tensor(entry['energy'], dtype=torch.float32, device=device)

    # Forces
    forces_gpu = torch.tensor(entry['forces'], dtype=torch.float32, device=device)

    # Stress (optional)
    stress_gpu = torch.tensor(entry['stress'], dtype=torch.float32, device=device) if 'stress' in entry else torch.zeros((3, 3), dtype=torch.float32, device=device)

    # Magnetic moments (optional)
    magmom_gpu = torch.tensor(entry['magmom'], dtype=torch.float32, device=device) if 'magmom' in entry else torch.zeros(len(atomic_numbers_gpu), dtype=torch.float32, device=device)

    # Move back to CPU for storage and convert to Python lists for msgpack
    processed_entry = {
        "Z": atomic_numbers_gpu.cpu().numpy().astype(np.int16).tolist(),
        "pos": positions_gpu.cpu().numpy().astype(np.float32).tolist(),
        "cell": cell_gpu.cpu().numpy().astype(np.float32).tolist(),
        "energy": energy_gpu.cpu().numpy().astype(np.float32).item(),  # .item() for scalar
        "forces": forces_gpu.cpu().numpy().astype(np.float32).tolist(),
        "stress": stress_gpu.cpu().numpy().astype(np.float32).tolist(),
        "magmom": magmom_gpu.cpu().numpy().astype(np.float32).tolist()
    }
    return processed_entry


def write_lmdb_shard(shard_path: Path, entries: List[Dict[str, Any]], shard_idx: int) -> None:
    """Write a list of entries to an LMDB shard."""
    print(f"Writing shard {shard_idx} with {len(entries)} entries to {shard_path}")
    
    # Create directory
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Estimate map size
    map_size = max(len(entries) * 50000, 1024 * 1024 * 1024)  # At least 1GB
    
    with lmdb.open(str(shard_path), map_size=map_size, writemap=True) as env:
        with env.begin(write=True) as txn:
            for i, entry in enumerate(entries):
                if entry is None:
                    continue
                
                # Serialize with msgpack
                key = f"{shard_idx:03d}_{i:06d}".encode()
                msgpack_data = msgpack.packb(entry, use_bin_type=True)
                
                # Compress with lz4
                compressed_data = lz4.frame.compress(msgpack_data)
                
                # Write to LMDB
                txn.put(key, compressed_data)
    
    # Get file size
    file_size_mb = shard_path.stat().st_size / (1024 * 1024)
    print(f"Shard {shard_idx} written: {len(entries)} entries, {file_size_mb:.1f} MB")


def validate_shard(shard_path: Path, shard_idx: int, num_samples: int = 5) -> bool:
    """Validate a shard by reading random keys."""
    print(f"Validating shard {shard_idx}...")
    
    with lmdb.open(str(shard_path), readonly=True) as env:
        with env.begin() as txn:
            # Get all keys
            cursor = txn.cursor()
            keys = [key for key, _ in cursor]
            
            if len(keys) == 0:
                print(f"Warning: Shard {shard_idx} is empty")
                return False
            
            # Check a few random keys
            for _ in range(min(num_samples, len(keys))):
                random_key = random.choice(keys)
                compressed_data = txn.get(random_key)
                if compressed_data is None:
                    print(f"Error: Key {random_key} not found")
                    return False
                
                try:
                    # Decompress and deserialize
                    msgpack_data = lz4.frame.decompress(compressed_data)
                    entry = msgpack.unpackb(msgpack_data, raw=False)
                    
                    # Check required keys
                    required_keys = ["Z", "pos", "cell", "energy", "forces"]
                    if not all(k in entry for k in required_keys):
                        print(f"Error: Missing required keys in {random_key}")
                        return False
                        
                except Exception as e:
                    print(f"Error processing key {random_key}: {e}")
                    return False
    
    print(f"Shard {shard_idx} validated successfully")
    return True


def convert_mptrj_to_lmdb(
    input_path: Path,
    output_dir: Path,
    samples_per_shard: int = 10000,
    max_memory_gb: float = 50.0
) -> None:
    """Convert MPtrj data to LMDB shards."""
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_gpu_device()
    if device is None:
        raise RuntimeError("GPU not available, but is required for this task.")

    current_shard = []
    shard_idx = 0
    total_samples = 0
    
    print(f"Converting {input_path} to LMDB shards in {output_dir}")
    print(f"Samples per shard: {samples_per_shard}")
    print(f"Max memory usage: {max_memory_gb} GB")

    try:
        for i, entry in enumerate(read_mptrj_streaming(input_path)):
            if not check_memory_usage(max_memory_gb):
                print(f"Memory usage exceeded {max_memory_gb} GB. Stopping conversion.")
                break

            try:
                processed_entry = process_entry(entry, device)
                current_shard.append(processed_entry)
            except KeyError as e:
                # print(f"Skipping entry {i} due to missing key: {e}")
                continue  # Skip entries with missing required keys

            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1} samples, Memory: {psutil.virtual_memory().used / (1024**3):.1f} GB")

            if len(current_shard) >= samples_per_shard:
                shard_path = output_dir / f"train_{shard_idx:03d}.lmdb"
                write_lmdb_shard(shard_path, current_shard, shard_idx)
                if not validate_shard(shard_path, shard_idx):
                    print(f"Validation failed for shard {shard_idx}. Aborting.")
                    return
                total_samples += len(current_shard)
                current_shard = []
                gc.collect()  # Trigger garbage collection
                shard_idx += 1

        # Write any remaining samples in the last shard
        if current_shard:
            shard_path = output_dir / f"train_{shard_idx:03d}.lmdb"
            write_lmdb_shard(shard_path, current_shard, shard_idx)
            if not validate_shard(shard_path, shard_idx):
                print(f"Validation failed for final shard {shard_idx}. Aborting.")
                return
            total_samples += len(current_shard)

    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return

    print("Conversion completed!")
    print(f"Total samples: {total_samples}")
    print(f"Total shards: {shard_idx + (1 if current_shard else 0)}")
    print(f"Output directory: {output_dir}")
    print(f"Final memory usage: {psutil.virtual_memory().used / (1024**3):.1f} GB")


def main():
    parser = argparse.ArgumentParser(description="Convert MPtrj JSON/NPZ to LMDB shards.")
    parser.add_argument("input_path", type=Path, help="Path to the input MPtrj JSON or NPZ file.")
    parser.add_argument("output_dir", type=Path, help="Directory to store the LMDB shards.")
    parser.add_argument("--samples_per_shard", type=int, default=10000,
                        help="Number of samples to store per LMDB shard.")
    parser.add_argument("--max_memory_gb", type=float, default=50.0,
                        help="Maximum RAM usage in GB before stopping conversion.")
    args = parser.parse_args()

    convert_mptrj_to_lmdb(args.input_path, args.output_dir, args.samples_per_shard, args.max_memory_gb)


if __name__ == "__main__":
    main()