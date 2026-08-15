"""
Shared Training Utilities for LaneGameFormer, PrayagGameFormer, PrayagLaneGCN

This module provides common utilities for:
1. Triton/torch.compile() optimization with platform detection
2. Multi-seed training for fair comparison (3 seeds)
3. Reproducibility and seeding
4. HPO seed management (fixed seed=17 for HPO)

Usage:
    from shared_training_utils import (
        compile_model_with_triton,
        get_training_seeds,
        get_hpo_seed,
        set_seed,
        is_linux,
        TritonConfig
    )

Author: LaneGameFormer Research Team
Last Updated: January 2026
"""

import os
import sys
import platform
import random
from typing import Optional, List, Tuple, Any
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
# Using master seed 17 to generate 3 random seeds for reproducibility
# This ensures the same 3 seeds are used across all experiments
MASTER_SEED_FOR_TRAINING = 17

# HPO Configuration (unified across all models for fairness)
HPO_TRIALS = 20
HPO_EPOCHS = 10
HPO_PATIENCE = 3
TRAIN_EPOCHS = 50
TRAIN_PATIENCE = 5
TRAIN_SEEDS_COUNT = 3


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
    - Not hardcoded, but reproducibly generated
    
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
                      If False, allow non-deterministic ops for speed
    
    Note:
        Full determinism (deterministic=True) can be 10-30% slower.
        For HPO, use deterministic=False for speed.
        For final training, consider deterministic=True for exact reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        if deterministic:
            # Full determinism - slower but exact
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            # PyTorch 2.0+ deterministic algorithms
            if hasattr(torch, 'use_deterministic_algorithms'):
                try:
                    torch.use_deterministic_algorithms(True)
                except RuntimeError:
                    # Some ops don't have deterministic implementations
                    pass
        else:
            # Fast mode - allow non-deterministic for speed
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True


# =============================================================================
# Triton Compilation Configuration
# =============================================================================

@dataclass
class TritonConfig:
    """Configuration for Triton/torch.compile() optimization."""
    
    # Compilation mode: 'default', 'reduce-overhead', 'max-autotune'
    # - default: Balanced; works with CPU→GPU transfers in forward()
    # - reduce-overhead: Uses CUDA graphs; all tensors must be on GPU
    # - max-autotune: Maximum performance, longer compile time
    mode: str = 'default'
    
    # Backend: 'inductor' (Triton), 'eager', 'aot_eager'
    backend: str = 'inductor'
    
    # Enable dynamic shapes (for variable batch sizes)
    dynamic: bool = False
    
    # Fullgraph mode (requires no graph breaks)
    fullgraph: bool = False
    
    # Disable compilation (fallback to eager)
    disable: bool = False


def compile_model_with_triton(
    model: nn.Module,
    config: Optional[TritonConfig] = None,
    verbose: bool = True
) -> nn.Module:
    """
    Compile model with Triton backend for 2-5x speedup on Linux.
    
    This function:
    1. Checks platform and CUDA availability
    2. Compiles with torch.compile() on Linux with Triton
    3. Falls back gracefully to eager mode on Windows/Mac
    
    Args:
        model: PyTorch model to compile
        config: TritonConfig with compilation options
        verbose: Print compilation status
        
    Returns:
        Compiled model (or original model if compilation unavailable/fails)
    
    Performance Notes:
        - First forward pass is slow (compilation)
        - Subsequent passes are 2-5x faster
        - Best gains on transformer architectures
        - reduce-overhead mode is best for training
    """
    if config is None:
        config = TritonConfig()
    
    if config.disable:
        if verbose:
            print("  torch.compile() disabled by config")
        return model
    
    # Check PyTorch version
    if not hasattr(torch, 'compile'):
        if verbose:
            print("  torch.compile() not available (PyTorch < 2.0)")
        return model
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        if verbose:
            print("  torch.compile() requires CUDA, using eager mode")
        return model
    
    # Platform-specific handling
    if IS_WINDOWS:
        if verbose:
            print("  Windows detected: torch.compile() has limited support")
            print("  Using eager mode for stability (Triton requires Linux)")
        return model
    
    if IS_MAC:
        if verbose:
            print("  macOS detected: torch.compile() has limited support")
            print("  Using eager mode (Triton requires Linux)")
        return model
    
    # Linux with CUDA - attempt compilation
    if IS_LINUX:
        try:
            if verbose:
                print(f"  Compiling model with torch.compile()")
                print(f"    Mode: {config.mode}")
                print(f"    Backend: {config.backend}")
                print(f"    Dynamic: {config.dynamic}")
            
            compiled_model = torch.compile(
                model,
                mode=config.mode,
                backend=config.backend,
                dynamic=config.dynamic,
                fullgraph=config.fullgraph
            )
            
            if verbose:
                print("  ✓ Model compiled successfully with Triton backend")
                print("    Note: First forward pass will be slow (compilation)")
            
            return compiled_model
            
        except Exception as e:
            if verbose:
                print(f"  ✗ torch.compile() failed: {e}")
                print("    Falling back to eager mode")
            return model
    
    return model


def get_triton_compile_options() -> dict:
    """
    Get recommended torch.compile() options for trajectory prediction.
    
    Returns:
        dict: Keyword arguments for torch.compile()
    """
    return {
        'mode': 'reduce-overhead',  # Best for training with variable workloads
        'backend': 'inductor',       # Triton-based backend
        'dynamic': False,            # Static shapes for better optimization
        'fullgraph': False           # Allow graph breaks for compatibility
    }


# =============================================================================
# Multi-Seed Training Runner
# =============================================================================

def run_multi_seed_training(
    train_fn: callable,
    seeds: Optional[List[int]] = None,
    aggregate_results: bool = True,
    verbose: bool = True,
    **train_kwargs
) -> List[dict]:
    """
    Run training with multiple seeds for fair comparison.
    
    Args:
        train_fn: Training function that accepts seed as keyword argument
        seeds: List of seeds (default: TRAINING_SEEDS = [42, 1337, 2024])
        aggregate_results: Compute mean/std of metrics across seeds
        verbose: Print progress
        **train_kwargs: Additional arguments passed to train_fn
        
    Returns:
        List of result dictionaries from each seed run
        
    Example:
        def train_model(seed, epochs, lr, **kwargs):
            set_seed(seed)
            # ... training code ...
            return {'ade': best_ade, 'fde': best_fde}
        
        results = run_multi_seed_training(
            train_model,
            epochs=50,
            lr=1e-4
        )
    """
    if seeds is None:
        seeds = get_training_seeds()
    
    results = []
    
    for i, seed in enumerate(seeds):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Multi-Seed Training: Run {i+1}/{len(seeds)} (seed={seed})")
            print(f"{'='*60}\n")
        
        # Run training with this seed
        result = train_fn(seed=seed, **train_kwargs)
        results.append(result)
        
        if verbose:
            print(f"\nRun {i+1} complete: {result}")
    
    # Aggregate results
    if aggregate_results and results:
        if verbose:
            print(f"\n{'='*60}")
            print("Multi-Seed Results Summary")
            print(f"{'='*60}")
        
        # Find common numeric keys
        numeric_keys = []
        for key in results[0].keys():
            if isinstance(results[0][key], (int, float)):
                numeric_keys.append(key)
        
        for key in numeric_keys:
            values = [r[key] for r in results if key in r]
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            if verbose:
                print(f"  {key}: {mean_val:.4f} ± {std_val:.4f}")
    
    return results


def aggregate_multi_seed_metrics(results: List[dict]) -> dict:
    """
    Aggregate metrics from multi-seed training runs.
    
    Args:
        results: List of result dicts from each seed
        
    Returns:
        Dict with mean and std for each metric
    """
    if not results:
        return {}
    
    aggregated = {}
    
    # Find all numeric keys
    for key in results[0].keys():
        values = []
        for r in results:
            if key in r and isinstance(r[key], (int, float)):
                values.append(r[key])
        
        if values:
            aggregated[f'{key}_mean'] = float(np.mean(values))
            aggregated[f'{key}_std'] = float(np.std(values))
            aggregated[f'{key}_min'] = float(np.min(values))
            aggregated[f'{key}_max'] = float(np.max(values))
    
    aggregated['num_seeds'] = len(results)
    aggregated['seeds'] = [r.get('seed', None) for r in results]
    
    return aggregated


# =============================================================================
# GPU Optimization Utilities
# =============================================================================

def enable_tf32() -> None:
    """
    Enable TF32 for Ampere+ GPUs (RTX 30xx, A100, etc.).
    
    TF32 provides ~3x speedup for matrix operations with minimal precision loss.
    Recommended for training trajectory prediction models.
    """
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def enable_cudnn_benchmark() -> None:
    """
    Enable cuDNN benchmark mode for auto-tuning convolution algorithms.
    
    This finds the fastest convolution algorithm for your hardware.
    Slight overhead on first run, but faster subsequent runs.
    """
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def clear_gpu_memory() -> None:
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_gpu_memory_info() -> dict:
    """Get current GPU memory usage."""
    if not torch.cuda.is_available():
        return {'available': False}
    
    return {
        'available': True,
        'device': torch.cuda.get_device_name(0),
        'total_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3,
        'allocated_gb': torch.cuda.memory_allocated(0) / 1024**3,
        'cached_gb': torch.cuda.memory_reserved(0) / 1024**3,
        'free_gb': (torch.cuda.get_device_properties(0).total_memory - 
                   torch.cuda.memory_allocated(0)) / 1024**3
    }


# =============================================================================
# Print Configuration Summary
# =============================================================================

def is_flash_attention_available() -> bool:
    """
    Check if Flash Attention 2 is available.
    
    Flash Attention requires:
    - Linux (WSL counts)
    - CUDA GPU with compute capability >= 7.0
    - flash-attn package installed
    """
    if not torch.cuda.is_available():
        return False
    try:
        from flash_attn import flash_attn_func
        return True
    except ImportError:
        return False


def get_unified_hpo_search_space():
    """
    Return the IDENTICAL hyperparameter search space used by all 3 models.
    
    This ensures fairness in the research comparison. All models tune
    the same training hyperparameters with the same ranges.
    Architecture parameters are FIXED per model (not tuned).
    
    Returns:
        dict describing the search space for documentation
    """
    return {
        "lr": {"type": "float", "low": 1e-4, "high": 1e-3, "log": True},
        "weight_decay": {"type": "float", "low": 1e-6, "high": 1e-4, "log": True},
        "grad_clip": {"type": "float", "low": 0.5, "high": 2.0, "log": False},
    }


def print_training_config_summary(seed: int, is_hpo: bool = False) -> None:
    """Print training configuration summary."""
    print("\n" + "="*60)
    print("Training Configuration Summary")
    print("="*60)
    print(f"Platform: {platform.system()} ({platform.machine()})")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"CUDA: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"TF32: {torch.backends.cuda.matmul.allow_tf32}")
        print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    else:
        print("CUDA: Not available")
    
    print(f"\nTriton available: {is_triton_available()}")
    print(f"torch.compile available: {hasattr(torch, 'compile')}")
    print(f"Flash Attention available: {is_flash_attention_available()}")
    
    if is_hpo:
        print(f"\nHPO Mode:")
        print(f"  Fixed seed: {HPO_SEED}")
        print(f"  Trials: {HPO_TRIALS}")
        print(f"  Epochs/trial: {HPO_EPOCHS}")
        print(f"  Patience: {HPO_PATIENCE}")
    else:
        print(f"\nTraining Mode:")
        print(f"  Current seed: {seed}")
        print(f"  Multi-seed seeds: {get_training_seeds()}")
        print(f"  Epochs: {TRAIN_EPOCHS}")
        print(f"  Patience: {TRAIN_PATIENCE}")
    
    print("="*60 + "\n")


if __name__ == '__main__':
    # Self-test
    print("Shared Training Utilities - Self Test")
    print("="*60)
    
    print(f"\nPlatform Detection:")
    print(f"  IS_LINUX: {IS_LINUX}")
    print(f"  IS_WINDOWS: {IS_WINDOWS}")
    print(f"  IS_MAC: {IS_MAC}")
    
    print(f"\nTriton Support:")
    print(f"  is_triton_available(): {is_triton_available()}")
    
    print(f"\nSeeds:")
    print(f"  HPO seed: {get_hpo_seed()}")
    print(f"  Training seeds: {get_training_seeds()}")
    
    print(f"\nGPU Info:")
    print(f"  {get_gpu_memory_info()}")
    
    print("\n✓ Self-test complete")
