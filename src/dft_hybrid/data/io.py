"""I/O utilities and multiprocessing configuration for data loading."""

import torch
import os


def set_safe_mp():
    """Configure safe multiprocessing settings for PyTorch data loading.
    
    This function addresses common multiprocessing issues in PyTorch:
    - Sets spawn method to avoid fork issues
    - Disables tokenizer parallelism to prevent conflicts
    - Limits thread count to prevent resource contention
    - Sets memory-efficient settings
    """
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(1)
    
    # Memory management settings
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear GPU cache

