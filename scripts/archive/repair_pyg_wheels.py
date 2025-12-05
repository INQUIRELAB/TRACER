#!/usr/bin/env python3
"""
PyTorch Geometric Wheel Repair Script

This script fixes PyTorch Geometric extension wheels that have ABI mismatches
with the current PyTorch installation.

Usage:
    python3 scripts/repair_pyg_wheels.py [--force] [--cpu-only]

Options:
    --force      Force reinstall even if wheels appear to work
    --cpu-only   Install CPU-only versions (no CUDA dependencies)
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path


def run_command(cmd, check=True):
    """Run a command and return the result."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error running command: {cmd}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False
    return True


def get_pytorch_info():
    """Get PyTorch version and CUDA version."""
    try:
        import torch
        pytorch_version = torch.__version__
        cuda_version = torch.version.cuda
        print(f"PyTorch version: {pytorch_version}")
        print(f"CUDA version: {cuda_version}")
        return pytorch_version, cuda_version
    except ImportError:
        print("PyTorch not installed!")
        return None, None


def check_pyg_extensions():
    """Check if PyG extensions are working."""
    try:
        import torch_scatter
        import torch_cluster
        import torch_sparse
        print("✅ PyG extensions appear to be working")
        return True
    except ImportError as e:
        print(f"❌ PyG extensions not working: {e}")
        return False
    except Exception as e:
        print(f"❌ PyG extensions error: {e}")
        return False


def uninstall_pyg_extensions():
    """Uninstall existing PyG extensions."""
    print("\n🔄 Uninstalling existing PyG extensions...")
    extensions = ['torch_scatter', 'torch_cluster', 'torch_sparse', 'torch_spline_conv']
    
    for ext in extensions:
        print(f"Uninstalling {ext}...")
        run_command(f"pip uninstall {ext} -y", check=False)


def install_pyg_extensions_cpu():
    """Install CPU-only PyG extensions."""
    print("\n🔄 Installing CPU-only PyG extensions...")
    
    # Get PyTorch version for compatibility
    pytorch_version, _ = get_pytorch_info()
    if not pytorch_version:
        print("❌ Cannot determine PyTorch version")
        return False
    
    # Extract major.minor version
    pytorch_major_minor = '.'.join(pytorch_version.split('.')[:2])
    
    extensions = [
        'torch_scatter',
        'torch_cluster', 
        'torch_sparse',
        'torch_spline_conv'
    ]
    
    for ext in extensions:
        print(f"Installing {ext} (CPU-only)...")
        # Try to install CPU-only version
        cmd = f"pip install {ext} -f https://data.pyg.org/whl/torch-{pytorch_major_minor}+cpu.html --no-cache-dir"
        if not run_command(cmd, check=False):
            # Fallback: install from PyPI
            cmd = f"pip install {ext} --no-cache-dir"
            if not run_command(cmd, check=False):
                print(f"❌ Failed to install {ext}")
                return False
    
    return True


def install_pyg_extensions_cuda():
    """Install CUDA PyG extensions."""
    print("\n🔄 Installing CUDA PyG extensions...")
    
    # Get PyTorch version and CUDA version
    pytorch_version, cuda_version = get_pytorch_info()
    if not pytorch_version or not cuda_version:
        print("❌ Cannot determine PyTorch/CUDA versions")
        return False
    
    # Extract major.minor version
    pytorch_major_minor = '.'.join(pytorch_version.split('.')[:2])
    
    extensions = [
        'torch_scatter',
        'torch_cluster',
        'torch_sparse', 
        'torch_spline_conv'
    ]
    
    for ext in extensions:
        print(f"Installing {ext} (CUDA {cuda_version})...")
        # Try to install CUDA version
        cmd = f"pip install {ext} -f https://data.pyg.org/whl/torch-{pytorch_major_minor}+cu{cuda_version.replace('.', '')}.html --no-cache-dir"
        if not run_command(cmd, check=False):
            # Fallback: install from PyPI
            cmd = f"pip install {ext} --no-cache-dir"
            if not run_command(cmd, check=False):
                print(f"❌ Failed to install {ext}")
                return False
    
    return True


def install_pyg_from_source():
    """Install PyG extensions from source."""
    print("\n🔄 Installing PyG extensions from source...")
    
    extensions = [
        'torch_scatter',
        'torch_cluster',
        'torch_sparse',
        'torch_spline_conv'
    ]
    
    for ext in extensions:
        print(f"Installing {ext} from source...")
        cmd = f"pip install {ext} --no-cache-dir --no-binary={ext}"
        if not run_command(cmd, check=False):
            print(f"❌ Failed to install {ext} from source")
            return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Repair PyTorch Geometric wheels')
    parser.add_argument('--force', action='store_true', 
                       help='Force reinstall even if wheels appear to work')
    parser.add_argument('--cpu-only', action='store_true',
                       help='Install CPU-only versions (no CUDA dependencies)')
    
    args = parser.parse_args()
    
    print("🔧 PyTorch Geometric Wheel Repair Script")
    print("=" * 50)
    
    # Check current PyTorch installation
    pytorch_version, cuda_version = get_pytorch_info()
    if not pytorch_version:
        print("❌ PyTorch not installed. Please install PyTorch first.")
        return 1
    
    # Check if PyG extensions are working
    if not args.force and check_pyg_extensions():
        print("✅ PyG extensions appear to be working. Use --force to reinstall anyway.")
        return 0
    
    # Uninstall existing extensions
    uninstall_pyg_extensions()
    
    # Install new extensions
    if args.cpu_only:
        success = install_pyg_extensions_cpu()
    else:
        # Try CUDA first, fallback to CPU
        success = install_pyg_extensions_cuda()
        if not success:
            print("\n🔄 CUDA installation failed, trying CPU-only...")
            success = install_pyg_extensions_cpu()
    
    if not success:
        print("\n🔄 Binary installation failed, trying from source...")
        success = install_pyg_from_source()
    
    if not success:
        print("\n❌ All installation methods failed!")
        return 1
    
    # Verify installation
    print("\n🔍 Verifying installation...")
    if check_pyg_extensions():
        print("✅ PyG extensions successfully repaired!")
        return 0
    else:
        print("❌ PyG extensions still not working after repair")
        return 1


if __name__ == "__main__":
    sys.exit(main())


