"""
Training script for PrayagGameFormer.

This script trains GameFormer on the ChunkedProjectPrayagBEVDataset
following the strategic comparison protocol.

Features:
- Full training history tracking with JSON/CSV export
- Proper checkpointing with scheduler state
- Resume capability with full state restoration
- Epoch-by-epoch metrics for graph generation
- AMP (Automatic Mixed Precision) for FP16 training
- cuDNN/TF32 optimizations for speed
"""

import os
import sys
import argparse
import time
import gc
import json
import platform
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# ============================================================================
# SPEED OPTIMIZATIONS (applied at module load time)
# ============================================================================

# cuDNN optimization: Auto-tune convolution algorithms for hardware
cudnn.benchmark = True
cudnn.deterministic = False  # Allow non-deterministic for speed

# TF32 for Ampere+ GPUs (RTX 30xx, A100, etc.) - 3x faster matrix ops
if hasattr(torch.backends.cuda, 'matmul'):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Enable Flash Attention / Memory-Efficient Attention via SDPA
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)

# Tensor Core matmul precision: 'high' uses TF32, fastest on Ampere+
torch.set_float32_matmul_precision('high')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import getConfig, getTestConfig
from data import PrayagDataset, collateFn
from model.gameformer import GameFormer, GameFormerLoss, PostProcess
from utils import setSeed, getOptimizer, getScheduler
from utils.metrics import appendMetrics, computeMetrics
from utils.helpers import saveCheckpoint, loadCheckpoint, clearGpuMemory, TrainingHistory
from utils.training_utils import (
    compile_model_with_triton,
    get_training_seeds,
    set_seed as shared_set_seed,
    TritonConfig,
    is_triton_available
)

# ============================================================================
# TRAINING DEFAULTS (Senior ML Engineer recommendations)
# ============================================================================
# For trajectory prediction transformers:
# - 100 epochs is typical for full convergence
# - Patience of 10 allows for LR schedule valleys
# - Batch size 16-32 for stable gradients with AMP
DEFAULT_EPOCHS = 50
DEFAULT_PATIENCE = 5
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 8 if platform.system() == 'Linux' else 4


def parseArgs():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train PrayagGameFormer")
    
    parser.add_argument(
        "--datasetType", type=str, default="10Hz",
        choices=["10Hz", "30Hz"],
        help="Dataset type (10Hz or 30Hz)"
    )
    parser.add_argument(
        "--batchSize", type=int, default=None,
        help="Batch size (overrides config)"
    )
    parser.add_argument(
        "--numEpochs", type=int, default=None,
        help="Number of epochs (overrides config)"
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Learning rate (overrides config)"
    )
    parser.add_argument(
        "--testChunks", type=int, default=None,
        help="Number of chunks for testing (limits dataset size)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (default: runs with 3 seeds [42, 1337, 2024] for fair comparison)"
    )
    parser.add_argument(
        "--multiSeed", action="store_true",
        help="Run training with multiple seeds [42, 1337, 2024] for fair comparison"
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help=f"Number of data loading workers (default: {DEFAULT_NUM_WORKERS})"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--saveDir", type=str, default=None,
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--patience", type=int, default=None,
        help=f"Early stopping patience (default: {DEFAULT_PATIENCE})"
    )
    parser.add_argument(
        "--hpoParams", type=str, default=None,
        help="Path to best_params.json from HPO to load training hyperparameters"
    )
    parser.add_argument(
        "--datasetPath", type=str, default=None,
        help="Custom path to dataset (overrides config default)"
    )
    parser.add_argument(
        "--maxAgents", type=int, default=None,
        help="Max agents (reduce for OOM issues, default: from config)"
    )
    
    return parser.parse_args()


def trainEpoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    postProcess: nn.Module,
    device: str,
    epoch: int,
    scaler: Optional[GradScaler] = None
) -> dict:
    """
    Train for one epoch with AMP support.
    
    Args:
        model: Model to train
        loader: Training data loader
        optimizer: Optimizer
        loss_fn: Loss function
        postProcess: Post-processor for metrics
        device: Device to use
        epoch: Current epoch number
        scaler: GradScaler for AMP (optional)
        
    Returns:
        Dictionary of training metrics
    """
    model.train()
    metrics = {}
    use_amp = scaler is not None and device == "cuda"
    
    pbar = tqdm(loader, desc=f"Train Epoch {epoch}")
    
    for batch in pbar:
        # Move batch to device with async transfer
        batch = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        
        optimizer.zero_grad(set_to_none=True)  # More memory efficient
        
        try:
            # Mixed precision forward pass
            with autocast('cuda', enabled=use_amp):
                out = model(batch)
                lossOut = loss_fn(out, batch)
            
            # Mixed precision backward pass
            if scaler is not None:
                scaler.scale(lossOut["loss"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                lossOut["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            
            # Accumulate metrics
            with torch.no_grad():
                postOut = postProcess(out, batch)
            metrics = appendMetrics(metrics, lossOut, postOut)
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{lossOut['loss'].item():.4f}"
            })
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"WARNING: OOM, skipping batch")
                clearGpuMemory()
                continue
            raise e
        
        # Clear intermediate tensors
        del out, lossOut, postOut
        
    clearGpuMemory()
    return computeMetrics(metrics)


def validateEpoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    postProcess: nn.Module,
    device: str,
    epoch: int,
    use_amp: bool = False
) -> dict:
    """
    Validate for one epoch with AMP support.
    
    Args:
        model: Model to validate
        loader: Validation data loader
        loss_fn: Loss function
        postProcess: Post-processor for metrics
        device: Device to use
        epoch: Current epoch number
        use_amp: Whether to use automatic mixed precision
        
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    metrics = {}
    
    pbar = tqdm(loader, desc=f"Val Epoch {epoch}")
    
    with torch.no_grad():
        for batch in pbar:
            # Move batch to device with async transfer
            batch = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            
            try:
                # Mixed precision validation
                with autocast('cuda', enabled=use_amp):
                    out = model(batch)
                    lossOut = loss_fn(out, batch)
                postOut = postProcess(out, batch)
                
                metrics = appendMetrics(metrics, lossOut, postOut)
                
                pbar.set_postfix({
                    "loss": f"{lossOut['loss'].item():.4f}"
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"WARNING: OOM, skipping batch")
                    clearGpuMemory()
                    continue
                raise e
            
            del out, lossOut, postOut
    
    clearGpuMemory()
    return computeMetrics(metrics)


def main(args_override=None):
    """Main training function."""
    args = args_override if args_override is not None else parseArgs()
    
    # Get config
    if args.testChunks is not None:
        config = getTestConfig(args.testChunks, args.datasetType)
    else:
        config = getConfig(args.datasetType)
    
    # Load HPO best parameters if provided
    if args.hpoParams is not None:
        hpoParamsPath = Path(args.hpoParams)
        if hpoParamsPath.exists():
            with open(hpoParamsPath, 'r') as f:
                hpoData = json.load(f)
            bestParams = hpoData.get("bestParams", {})
            print(f"\nLoading HPO best parameters from: {hpoParamsPath}")
            print(f"  Best trial: {hpoData.get('bestTrial', 'N/A')}")
            print(f"  Best minADE: {hpoData.get('bestValue', 'N/A'):.6f}")
            
            # Apply HPO parameters to config
            if "lr" in bestParams:
                config["lr"] = [bestParams["lr"], bestParams["lr"] / 10]
                print(f"  lr: {bestParams['lr']:.6e}")
            if "weight_decay" in bestParams:
                config["weightDecay"] = bestParams["weight_decay"]
                print(f"  weight_decay: {bestParams['weight_decay']:.6e}")
            if "grad_clip" in bestParams:
                config["gradClip"] = bestParams["grad_clip"]
                print(f"  grad_clip: {bestParams['grad_clip']:.2f}")
            # Loss weights
            for key in ["lambda_cls", "lambda_goal", "lambda_coll", "lambda_dyn", "lambda_coop"]:
                if key in bestParams:
                    config[key] = bestParams[key]
                    print(f"  {key}: {bestParams[key]:.4f}")
        else:
            print(f"Warning: HPO params file not found: {hpoParamsPath}")
    
    # Override config with args (with sensible defaults)
    if args.batchSize is not None:
        config["batchSize"] = args.batchSize
        config["valBatchSize"] = args.batchSize
    else:
        config["batchSize"] = config.get("batchSize", DEFAULT_BATCH_SIZE)
        config["valBatchSize"] = config.get("valBatchSize", DEFAULT_BATCH_SIZE)
    
    if args.numEpochs is not None:
        config["numEpochs"] = args.numEpochs
    else:
        config["numEpochs"] = config.get("numEpochs", DEFAULT_EPOCHS)
    
    if args.lr is not None:
        config["lr"] = [args.lr, args.lr / 10]
    if args.saveDir is not None:
        config["saveDir"] = args.saveDir
    
    # Override dataset path if provided
    if args.datasetPath is not None:
        config["datasetDir"] = args.datasetPath
        config["trainAnnotations"] = str(Path(args.datasetPath) / "train" / "annotations")
        config["trainVideos"] = str(Path(args.datasetPath) / "train" / "videos")
        config["valAnnotations"] = str(Path(args.datasetPath) / "val" / "annotations")
        config["valVideos"] = str(Path(args.datasetPath) / "val" / "videos")
        config["chunkList"] = {
            "train": str(Path(args.datasetPath) / "train_chunks.txt"),
            "val": str(Path(args.datasetPath) / "val_chunks.txt"),
            "test": str(Path(args.datasetPath) / "test_chunks.txt"),
        }
    
    # Override maxAgents if provided (for OOM mitigation)
    if args.maxAgents is not None:
        config["maxAgents"] = args.maxAgents
        if "neighborsToPredict" in config:
            config["neighborsToPredict"] = min(config.get("neighborsToPredict", 31), args.maxAgents - 1)
    
    # Worker settings
    numWorkers = args.workers if args.workers is not None else DEFAULT_NUM_WORKERS
    config["workers"] = numWorkers
    config["valWorkers"] = numWorkers
    
    # Early stopping patience
    patience = args.patience if args.patience is not None else DEFAULT_PATIENCE
    
    config["seed"] = args.seed
    
    # Set seed
    setSeed(config["seed"])
    
    # Device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    
    # Create save directory
    saveDir = Path(config["saveDir"])
    saveDir.mkdir(parents=True, exist_ok=True)
    
    # Print system configuration
    print("="*60)
    print("PrayagGameFormer Training")
    print("="*60)
    print(f"\nSystem Configuration:")
    print(f"  Platform: {platform.system()} ({platform.machine()})")
    print(f"  Device: {device}")
    
    if torch.cuda.is_available() and device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"  cuDNN benchmark: {cudnn.benchmark}")
        print(f"  TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")
        print(f"  AMP (FP16): Enabled")
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {config['numEpochs']}")
    print(f"  Batch size: {config['batchSize']}")
    print(f"  Learning rate: {config['lr']}")
    print(f"  Early stopping patience: {patience}")
    print(f"  DataLoader workers: {numWorkers}")
    print(f"  Save directory: {saveDir}")
    
    # Create datasets
    trainDataset = PrayagDataset(
        config, split="train", 
        testChunks=config.get("testChunks")
    )
    valDataset = PrayagDataset(
        config, split="val",
        testChunks=config.get("testChunks")
    )
    
    # Create data loaders with optimizations
    trainLoader = DataLoader(
        trainDataset,
        batch_size=config["batchSize"],
        shuffle=True,
        num_workers=numWorkers,
        collate_fn=collateFn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if numWorkers > 0 else False,
        prefetch_factor=4 if numWorkers > 0 else None
    )
    valLoader = DataLoader(
        valDataset,
        batch_size=config["valBatchSize"],
        shuffle=False,
        num_workers=numWorkers,
        collate_fn=collateFn,
        pin_memory=True,
        persistent_workers=True if numWorkers > 0 else False,
        prefetch_factor=4 if numWorkers > 0 else None
    )
    
    print(f"\nTrain: {len(trainDataset)} samples, {len(trainLoader)} batches")
    print(f"Val: {len(valDataset)} samples, {len(valLoader)} batches")
    
    # Initialize AMP scaler for mixed precision training
    scaler = GradScaler() if device == "cuda" else None
    use_amp = device == "cuda"
    
    # Create model
    model = GameFormer(config)
    model = model.to(device)
    
    # torch.compile disabled: variable-sized inputs (agent counts) cause
    # constant recompilation with dynamic=False, and baddbmm crash with dynamic=True
    # Speed gains come from Flash SDPA + TF32 + cuDNN benchmark instead
    print("\nModel Compilation:")
    print("  torch.compile() disabled (variable-size inputs cause recompilation)")
    print("  Speed optimizations: Flash SDPA + TF32 + cuDNN benchmark")
    
    # Count parameters
    numParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {numParams:,}")
    
    # Create loss and post-processor
    loss_fn = GameFormerLoss(config).to(device)
    postProcess = PostProcess(config).to(device)
    
    # Create optimizer and scheduler
    optimizer = getOptimizer(
        model.parameters(),
        lr=config["lr"][0],
        weightDecay=config.get("weightDecay", 1e-4)
    )
    scheduler = getScheduler(
        optimizer,
        milestones=config["lrEpochs"],
        gamma=0.5
    )
    
    # Initialize training history tracker
    historyTracker = TrainingHistory(str(saveDir))
    
    # Resume if specified
    startEpoch = 0
    bestVal = float("inf")
    
    if args.resume is not None:
        checkpoint = loadCheckpoint(
            model, optimizer, args.resume, device, scheduler
        )
        startEpoch = checkpoint["epoch"]
        bestVal = checkpoint.get("bestMetric", float("inf"))
        
        # Restore history
        if "history" in checkpoint and checkpoint["history"]:
            historyTracker.loadFromCheckpoint(checkpoint["history"])
        
        print(f"Resumed from epoch {startEpoch}")
        print(f"Best validation loss so far: {bestVal:.4f}")
    
    # Save config to JSON
    configPath = saveDir / "config.json"
    serializableConfig = {}
    for k, v in config.items():
        if isinstance(v, (int, float, str, bool, list, dict)):
            serializableConfig[k] = v
        else:
            serializableConfig[k] = str(v)
    
    with open(configPath, "w") as f:
        json.dump(serializableConfig, f, indent=2)
    
    # Early stopping
    patienceCounter = 0
    
    # Training loop
    print("="*60)
    
    for epoch in range(startEpoch, config["numEpochs"]):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config['numEpochs']}")
        currentLr = optimizer.param_groups[0]['lr']
        print(f"LR: {currentLr:.6f}")
        print(f"{'='*60}")
        
        # Train with AMP
        startTime = time.time()
        trainMetrics = trainEpoch(
            model, trainLoader, optimizer, loss_fn, postProcess, device, epoch + 1, scaler
        )
        trainTime = time.time() - startTime
        trainMetrics["time"] = trainTime
        
        print(f"\nTrain: loss={trainMetrics.get('loss', 0):.4f}, "
              f"ade={trainMetrics.get('ade', 0):.4f}, "
              f"fde={trainMetrics.get('fde', 0):.4f}, "
              f"time={trainTime:.1f}s")
        
        # Validate with AMP
        startTime = time.time()
        valMetrics = validateEpoch(
            model, valLoader, loss_fn, postProcess, device, epoch + 1
        )
        valTime = time.time() - startTime
        valMetrics["time"] = valTime
        
        print(f"Val: loss={valMetrics.get('loss', 0):.4f}, "
              f"ade={valMetrics.get('ade', 0):.4f}, "
              f"fde={valMetrics.get('fde', 0):.4f}, "
              f"time={valTime:.1f}s")
        
        # Update scheduler
        scheduler.step()
        
        # Add to history tracker
        historyTracker.addEpoch(
            epoch=epoch + 1,
            trainMetrics=trainMetrics,
            valMetrics=valMetrics,
            lr=currentLr
        )
        
        # Save history after each epoch
        historyTracker.save("training_history")
        
        # Save checkpoint
        valLoss = valMetrics.get("loss", float("inf"))
        
        # Save latest (with full history)
        saveCheckpoint(
            model, optimizer, epoch + 1,
            str(saveDir / "latest.pth"),
            metrics=valMetrics,
            scheduler=scheduler,
            history=historyTracker.getHistory(),
            bestMetric=bestVal,
            config=serializableConfig
        )
        
        # Save best and check early stopping
        if valLoss < bestVal:
            bestVal = valLoss
            patienceCounter = 0
            saveCheckpoint(
                model, optimizer, epoch + 1,
                str(saveDir / "best.pth"),
                metrics=valMetrics,
                scheduler=scheduler,
                history=historyTracker.getHistory(),
                bestMetric=bestVal,
                config=serializableConfig
            )
            print(f"New best model saved (val_loss={valLoss:.4f})")
        else:
            patienceCounter += 1
            print(f"No improvement. Patience: {patienceCounter}/{patience}")
        
        # Save periodic
        if (epoch + 1) % 5 == 0:
            saveCheckpoint(
                model, optimizer, epoch + 1,
                str(saveDir / f"epoch_{epoch + 1}.pth"),
                metrics=valMetrics,
                scheduler=scheduler,
                history=historyTracker.getHistory(),
                bestMetric=bestVal,
                config=serializableConfig
            )
        
        # Early stopping
        if patienceCounter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break
        
        # Memory cleanup
        gc.collect()
        clearGpuMemory()
    
    # Print final summary
    historyTracker.printSummary()
    
    print("\nTraining complete!")
    print(f"Best validation loss: {bestVal:.4f}")
    print(f"Training history saved to: {saveDir / 'training_history.json'}")
    print(f"Training history CSV: {saveDir / 'training_history_train.csv'}")
    print(f"Validation history CSV: {saveDir / 'training_history_val.csv'}")
    
    return {"best_val_loss": bestVal, "seed": config["seed"]}


if __name__ == "__main__":
    args = parseArgs()
    
    # Multi-seed training for fair comparison
    if args.multiSeed:
        seeds = get_training_seeds()  # from master seed 17
        print(f"\n{'='*60}")
        print(f"MULTI-SEED TRAINING: Running {len(seeds)} experiments")
        print(f"Seeds: {seeds} (generated from master seed 17)")
        print(f"{'='*60}")
        
        all_results = []
        original_save_dir = args.saveDir
        
        for i, seed in enumerate(seeds):
            print(f"\n{'#'*60}")
            print(f"# SEED RUN {i+1}/{len(seeds)}: seed={seed}")
            print(f"{'#'*60}")
            
            # Set seed for reproducibility
            args.seed = seed
            shared_set_seed(seed, deterministic=False)
            
            # Modify save_dir for this seed
            if original_save_dir:
                args.saveDir = str(Path(original_save_dir) / f"seed_{seed}")
            
            result = main(args_override=args)
            if result:
                all_results.append(result)
        
        # Restore original save_dir
        args.saveDir = original_save_dir
        
        print(f"\n{'='*60}")
        print(f"MULTI-SEED TRAINING COMPLETE")
        print(f"Results saved in subdirectories: seed_42, seed_1337, seed_2024")
        if all_results:
            losses = [r["best_val_loss"] for r in all_results]
            print(f"Best val loss: {np.mean(losses):.4f} ± {np.std(losses):.4f}")
        print(f"{'='*60}")
    else:
        # Single seed training
        seed = args.seed if args.seed is not None else 42
        args.seed = seed
        shared_set_seed(seed, deterministic=False)
        print(f"Using seed: {seed}")
        main(args_override=args)
