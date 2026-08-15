"""
Training Utilities for PrayagGameFormer

This module provides utilities for:
1. Triton/torch.compile() optimization with platform detection
2. Multi-seed training for fair comparison (3 seeds)
3. Reproducibility and seeding
4. HPO seed management (fixed seed=17 for HPO)
"""

import os
import platform
import random
from typing import Optional, List
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


# =============================================================================
# Platform Detection
# =============================================================================

IS_LINUX = platform.system() == 'Linux'
IS_WINDOWS = platform.system() == 'Windows'
IS_MAC = platform.system() == 'Darwin'


def is_linux() -> bool:
    """Check if running on Linux (required for Triton)."""
    return IS_LINUX


def is_triton_available() -> bool:
    """
    Check if Triton is available for torch.compile().
    
    Triton is only supported on Linux with CUDA.
    Returns False on Windows/Mac or without CUDA.
    """
    if not IS_LINUX:
        return False
    
    if not torch.cuda.is_available():
        return False
    
    try:
        import triton
        return True
    except ImportError:
        return False


# =============================================================================
# Seeding Configuration
# =============================================================================

# HPO Seed: Fixed for reproducible hyperparameter optimization
HPO_SEED = 17

# Training Seeds: Generated from master seed for fair comparison
MASTER_SEED_FOR_TRAINING = 17


def get_hpo_seed() -> int:
    """
    Get the fixed seed for HPO runs.
    
    Using a fixed seed (17) ensures:
    - Reproducible HPO search across models
    - Fair comparison of hyperparameter sensitivity
    - Consistent trial ordering across runs
    
    Returns:
        int: HPO seed (17)
    """
    return HPO_SEED


def get_training_seeds() -> List[int]:
    """
    Generate 3 random training seeds from a master seed.
    
    Using a master seed (17) to generate random seeds ensures:
    - Variance estimation for metrics
    - Fair comparison across models (same random seeds for all)
    - Statistical significance testing capability
    
    Returns:
        List[int]: List of 3 randomly generated training seeds
    """
    rng = random.Random(MASTER_SEED_FOR_TRAINING)
    seeds = [rng.randint(0, 2**31 - 1) for _ in range(3)]
    return seeds


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed to set
        deterministic: If True, enable full determinism (slower)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True


# =============================================================================
# Triton Compilation Configuration
# =============================================================================

@dataclass
class TritonConfig:
    """Configuration for Triton/torch.compile() optimization."""
    mode: str = 'reduce-overhead'
    backend: str = 'inductor'
    dynamic: bool = False
    fullgraph: bool = False
    disable: bool = False


def compile_model_with_triton(
    model: nn.Module,
    config: Optional[TritonConfig] = None,
    verbose: bool = True
) -> nn.Module:
    """
    Compile model with Triton backend for 2-5x speedup on Linux.
    
    Args:
        model: PyTorch model to compile
        config: TritonConfig with compilation options
        verbose: Print compilation status
        
    Returns:
        Compiled model (or original model if compilation unavailable/fails)
    """
    if config is None:
        config = TritonConfig()
    
    if config.disable:
        if verbose:
            print("  torch.compile() disabled by config")
        return model
    
    if not hasattr(torch, 'compile'):
        if verbose:
            print("  torch.compile() not available (PyTorch < 2.0)")
        return model
    
    if not torch.cuda.is_available():
        if verbose:
            print("  torch.compile() requires CUDA, using eager mode")
        return model
    
    if IS_WINDOWS:
        if verbose:
            print("  Windows detected: using eager mode (Triton requires Linux)")
        return model
    
    if IS_MAC:
        if verbose:
            print("  macOS detected: using eager mode (Triton requires Linux)")
        return model
    
    if IS_LINUX:
        try:
            if verbose:
                print(f"  Compiling model with torch.compile()")
                print(f"    Mode: {config.mode}, Backend: {config.backend}")
            
            compiled_model = torch.compile(
                model,
                mode=config.mode,
                backend=config.backend,
                dynamic=config.dynamic,
                fullgraph=config.fullgraph
            )
            
            if verbose:
                print("  ✓ Model compiled successfully with Triton backend")
            
            return compiled_model
            
        except Exception as e:
            if verbose:
                print(f"  ✗ torch.compile() failed: {e}")
                print("    Falling back to eager mode")
            return model
    
    return model


def enable_tf32() -> None:
    """Enable TF32 for faster matrix operations on Ampere+ GPUs."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def enable_cudnn_benchmark() -> None:
    """Enable cuDNN auto-tuning for faster convolutions."""
    torch.backends.cudnn.benchmark = True
