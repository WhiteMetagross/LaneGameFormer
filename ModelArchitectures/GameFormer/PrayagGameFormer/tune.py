"""
Hyperparameter tuning script for PrayagGameFormer using Optuna and Ray Tune.

Follows the strategic comparison plan:
- Objective Function: minADE only (standard baseline protocol)
- Common search space with LaneGCN for fair comparison

Features:
- SQLite storage for resume capability
- HPO history tracking with JSON/CSV export
- Checkpoint saving for best trials
"""

import os
import warnings
warnings.filterwarnings("ignore", message=".*multivariate.*")
import sys
import argparse
import gc
import json
import tempfile
import platform
import multiprocessing
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

# Speed optimizations
import torch.backends.cudnn as cudnn
cudnn.benchmark = True  # Auto-tune convolution algorithms
cudnn.deterministic = False  # Allow non-deterministic for speed

# TF32 for Tensor Cores (Ampere+) — ~3x speedup for matrix ops
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Enable Flash Attention / Memory-Efficient Attention via SDPA
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)

# Tensor Core matmul precision: 'high' uses TF32, fastest on Ampere+
torch.set_float32_matmul_precision('high')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import getConfig, getTestConfig, getHPOConfig
from data import PrayagDataset, collateFn
from model.gameformer import GameFormer, GameFormerLoss, PostProcess
from utils import setSeed, getOptimizer, getScheduler
from utils.metrics import appendMetrics, computeMetrics
from utils.helpers import clearGpuMemory, saveCheckpoint
from utils.training_utils import (
    compile_model_with_triton,
    get_hpo_seed,
    TritonConfig,
    is_triton_available
)

# HPO seed is fixed at 17 for reproducibility
HPO_SEED = get_hpo_seed()  # Returns 17

# Optuna
import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

# Ray Tune (optional)
try:
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.optuna import OptunaSearch
    HAS_RAY = True
except ImportError:
    HAS_RAY = False
    # Only print in main process
    if __name__ == "__main__":
        print("Ray Tune not available, using Optuna only")


def parseArgs():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Tune PrayagGameFormer hyperparameters")
    
    parser.add_argument(
        "--datasetType", type=str, default="10Hz",
        choices=["10Hz", "30Hz"],
        help="Dataset type"
    )
    parser.add_argument(
        "--numTrials", type=int, default=20,
        help="Number of HPO trials (default: 20 for fair comparison)"
    )
    parser.add_argument(
        "--numEpochs", type=int, default=10,
        help="Epochs per trial (default: 10 for fair comparison)"
    )
    parser.add_argument(
        "--patience", type=int, default=3,
        help="Early stopping patience for HPO trials (default: 3 for fair comparison)"
    )
    parser.add_argument(
        "--testChunks", type=int, default=1,
        help="Number of chunks for testing"
    )
    parser.add_argument(
        "--backend", type=str, default="optuna",
        choices=["optuna", "ray"],
        help="HPO backend"
    )
    parser.add_argument(
        "--numGpus", type=int, default=1,
        help="Number of GPUs (for Ray)"
    )
    parser.add_argument(
        "--seed", type=int, default=17,
        help="Random seed (default: 17, master seed for fair comparison)"
    )
    parser.add_argument(
        "--outputDir", type=str, default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume HPO from existing study (Optuna only)"
    )
    parser.add_argument(
        "--studyName", type=str, default="prayag_gameformer",
        help="Name for the Optuna study (used for resume)"
    )
    parser.add_argument(
        "--saveTrialModels", action="store_true",
        help="Save model checkpoints for best trials"
    )
    parser.add_argument(
        "--datasetPath", type=str, default=None,
        help="Custom path to dataset (overrides config default)"
    )
    parser.add_argument(
        "--batchSize", type=int, default=None,
        help="Batch size (default: 16, reduce for OOM issues)"
    )
    parser.add_argument(
        "--maxAgents", type=int, default=None,
        help="Max agents (default: 32, reduce for OOM issues)"
    )
    parser.add_argument(
        "--numWorkers", type=int, default=None,
        help="Number of DataLoader workers (default: 16 on Linux, 8 on Windows, 0 for single process)"
    )
    
    return parser.parse_args()


class HPOHistoryTracker:
    """
    Track HPO history for visualization.
    
    Saves trial results to JSON/CSV for graph generation.
    """
    
    def __init__(self, outputDir: str):
        """Initialize HPO history tracker."""
        self.outputDir = Path(outputDir)
        self.outputDir.mkdir(parents=True, exist_ok=True)
        
        self.history = {
            "trials": [],
            "metadata": {
                "startTime": datetime.now().isoformat(),
                "lastUpdated": None,
                "studyName": None
            }
        }
    
    def addTrial(
        self,
        trialNumber: int,
        params: Dict[str, Any],
        value: float,
        state: str,
        epochMetrics: Optional[List[Dict]] = None
    ) -> None:
        """Add a trial result."""
        entry = {
            "trial": trialNumber,
            "timestamp": datetime.now().isoformat(),
            "params": params,
            "value": value,
            "state": state
        }
        if epochMetrics:
            entry["epochMetrics"] = epochMetrics
        
        # Check if trial already exists (update) or new
        existing = next((t for t in self.history["trials"] if t["trial"] == trialNumber), None)
        if existing:
            self.history["trials"].remove(existing)
        
        self.history["trials"].append(entry)
        self.history["metadata"]["lastUpdated"] = datetime.now().isoformat()
    
    def setStudyName(self, name: str) -> None:
        """Set study name."""
        self.history["metadata"]["studyName"] = name
    
    def save(self, filename: str = "hpo_history") -> None:
        """Save history to JSON and CSV."""
        # Save JSON
        jsonPath = self.outputDir / f"{filename}.json"
        with open(jsonPath, "w") as f:
            json.dump(self.history, f, indent=2, default=str)
        
        # Save CSV (trials only)
        if self.history["trials"]:
            csvPath = self.outputDir / f"{filename}.csv"
            self._saveCsv(csvPath)
    
    def _saveCsv(self, path: Path) -> None:
        """Save trials to CSV."""
        trials = self.history["trials"]
        if not trials:
            return
        
        # Get all param keys
        paramKeys = set()
        for t in trials:
            paramKeys.update(t.get("params", {}).keys())
        paramKeys = sorted(paramKeys)
        
        headers = ["trial", "timestamp", "value", "state"] + [f"param_{k}" for k in paramKeys]
        
        with open(path, "w") as f:
            f.write(",".join(headers) + "\n")
            for t in trials:
                row = [
                    str(t["trial"]),
                    t["timestamp"],
                    str(t["value"]),
                    t["state"]
                ]
                for k in paramKeys:
                    row.append(str(t.get("params", {}).get(k, "")))
                f.write(",".join(row) + "\n")
    
    def getBest(self) -> Optional[Dict]:
        """Get best trial."""
        completedTrials = [t for t in self.history["trials"] if t["state"] == "COMPLETE"]
        if not completedTrials:
            return None
        return min(completedTrials, key=lambda x: x["value"])
    
    def printSummary(self) -> None:
        """Print HPO summary."""
        print("\n" + "="*60)
        print("HPO HISTORY SUMMARY")
        print("="*60)
        
        totalTrials = len(self.history["trials"])
        completedTrials = len([t for t in self.history["trials"] if t["state"] == "COMPLETE"])
        prunedTrials = len([t for t in self.history["trials"] if t["state"] == "PRUNED"])
        
        print(f"Total trials: {totalTrials}")
        print(f"Completed: {completedTrials}")
        print(f"Pruned: {prunedTrials}")
        
        best = self.getBest()
        if best:
            print(f"\nBest trial: {best['trial']}")
            print(f"Best value: {best['value']:.6f}")
            print("Best params:")
            for k, v in best.get("params", {}).items():
                print(f"  {k}: {v}")
        
        print("="*60)


def trainEpochForTune(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: str,
    scaler: Optional[GradScaler] = None,
    epoch: int = 0,
    trial_num: int = 0
) -> float:
    """Train for one epoch and return loss."""
    from tqdm import tqdm
    model.train()
    totalLoss = 0.0
    numBatches = 0
    numErrors = 0
    
    pbar = tqdm(loader, desc=f"Trial {trial_num} Epoch {epoch+1} Train", leave=False)
    for batch in pbar:
        # Move batch to device with non_blocking=True for async transfer
        batch = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        
        optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
        
        try:
            # Use AMP if scaler is provided
            if scaler is not None:
                with autocast(device_type='cuda', dtype=torch.float16):
                    out = model(batch)
                    lossOut = loss_fn(out, batch)
                
                scaler.scale(lossOut["loss"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(batch)
                lossOut = loss_fn(out, batch)
                
                lossOut["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            
            totalLoss += lossOut["loss"].item()
            numBatches += 1
            pbar.set_postfix({"loss": f"{totalLoss/numBatches:.4f}"})
            
            del out, lossOut
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                clearGpuMemory()
                numErrors += 1
                continue
            raise e
        except (IndexError, KeyError) as e:
            # Graph structure issues with certain hyperparameter combinations
            print(f"DEBUG: trainEpochForTune caught {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            numErrors += 1
            continue
    
    clearGpuMemory()
    
    # If all batches failed, return infinity to signal trial failure
    if numBatches == 0:
        return float("inf")
    
    return totalLoss / numBatches


def validateForTune(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    postProcess: nn.Module,
    device: str,
    trial_num: int = 0
) -> Dict[str, float]:
    """Validate and return metrics."""
    from tqdm import tqdm
    model.eval()
    metrics = {}
    numBatches = 0
    numErrors = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Trial {trial_num} Val", leave=False)
        for batch in pbar:
            # Move batch to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            
            try:
                out = model(batch)
                lossOut = loss_fn(out, batch)
                postOut = postProcess(out, batch)
                
                metrics = appendMetrics(metrics, lossOut, postOut)
                numBatches += 1
                
                del out, lossOut, postOut
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    clearGpuMemory()
                    numErrors += 1
                    continue
                raise e
            except (IndexError, KeyError) as e:
                # Graph structure issues with certain hyperparameter combinations
                print(f"DEBUG: validateForTune caught {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                numErrors += 1
                continue
    
    clearGpuMemory()
    
    # If all batches failed, return infinity metrics
    if numBatches == 0:
        return {"loss": float("inf"), "ade": float("inf"), "fde": float("inf")}
    
    return computeMetrics(metrics)


def createObjective(args, modelSaveDir: Optional[Path] = None, historyTracker: Optional[HPOHistoryTracker] = None):
    """Create Optuna objective function."""
    
    def objective(trial: Trial) -> float:
        """
        Optuna objective function.
        
        Objective: minADE only (standard baseline protocol as per strategic plan).
        
        Memory optimizations applied:
        - AMP (Automatic Mixed Precision) with GradScaler
        - Reduced DataLoader workers (4 Linux, 2 Windows)
        - Reduced max_agents (32 instead of 90)
        - Explicit tensor cleanup after each trial
        """
        # Clear GPU memory before starting a new trial
        gc.collect()
        clearGpuMemory()
        
        # Get HPO config (uses Stratified dataset - 20% sample for faster HPO)
        config = getHPOConfig(args.datasetType)
        config["numEpochs"] = args.numEpochs
        
        # Override dataset path if provided via command line
        if args.datasetPath:
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
        
        # ============================================================
        # MEMORY OPTIMIZATIONS (Senior AI Researcher recommendations)
        # ============================================================
        # Reduce max_agents from 90 to 32 to cut O(N²) attention memory
        # This reduces attention memory from ~168MB to ~21MB per layer
        config["maxAgents"] = args.maxAgents if args.maxAgents else 32
        config["neighborsToPredict"] = config["maxAgents"] - 1  # Max neighbors
        # ============================================================
        
        # ============================================================
        # TRAINING-ONLY HYPERPARAMETER SEARCH (No architecture tuning)
        # ============================================================
        # Per user requirements: HPO tunes ONLY training parameters
        # Architecture is FIXED for fair comparison across models
        
        # Learning rate: log-uniform around transformer sweet spot
        lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        
        # Weight decay: regularization for AdamW
        weightDecay = trial.suggest_float("weightDecay", 1e-6, 1e-4, log=True)
        
        # Gradient clipping: prevent exploding gradients
        gradClip = trial.suggest_float("gradClip", 0.5, 2.0)
        
        # Batch size: fixed or from args (reduce for OOM)
        batchSize = args.batchSize if args.batchSize else 16
        
        # ============================================================
        # FIXED ARCHITECTURE (Not tuned)
        # ============================================================
        dim = 256  # Fixed model dimension
        heads = 8  # Fixed attention heads
        decoderLevels = 3  # Fixed decoder depth
        
        # Update config with tuned training parameters
        config["lr"] = [lr, lr / 10]
        config["weightDecay"] = weightDecay
        config["gradClip"] = gradClip
        config["batchSize"] = batchSize
        config["valBatchSize"] = batchSize
        
        # Fixed architecture parameters
        config["dim"] = dim
        config["heads"] = heads
        config["decoderLevels"] = decoderLevels
        
        setSeed(args.seed + trial.number)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize AMP scaler for mixed precision training
        scaler = GradScaler() if device == "cuda" else None
        
        # Create datasets (use all Stratified chunks - already 20% sample)
        trainDataset = PrayagDataset(config, split="train", testChunks=None)
        valDataset = PrayagDataset(config, split="val", testChunks=None)
        
        # SPEED OPTIMIZED for 80-core server with 128GB RAM
        # - 16 workers to keep GPU saturated
        # - persistent_workers=True to avoid fork/spawn overhead
        # - pin_memory=True for faster CPU->GPU DMA transfer
        # - prefetch_factor=4 to preload more batches
        # Use args.numWorkers if specified, else default based on OS
        if args.numWorkers is not None:
            numWorkers = args.numWorkers
        else:
            numWorkers = 16 if os.name != 'nt' else 8
        
        trainLoader = DataLoader(
            trainDataset,
            batch_size=batchSize,
            shuffle=True,
            num_workers=numWorkers,
            collate_fn=collateFn,
            drop_last=True,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=4 if numWorkers > 0 else None
        )
        valLoader = DataLoader(
            valDataset,
            batch_size=batchSize,
            shuffle=False,
            num_workers=numWorkers,
            collate_fn=collateFn,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=4 if numWorkers > 0 else None
        )
        
        # Create model
        model = GameFormer(config).to(device)
        
        # Keep reference to original model for optimizer (torch.compile returns wrapper)
        model_for_optim = model
        
        # torch.compile disabled: variable-sized inputs cause recompilation overhead
        if trial.number == 0:
            print("  torch.compile() disabled (variable-size inputs)")
        
        loss_fn = GameFormerLoss(config).to(device)
        postProcess = PostProcess(config).to(device)
        
        optimizer = getOptimizer(model_for_optim.parameters(), lr=lr, weightDecay=weightDecay)
        scheduler = getScheduler(optimizer, config["lrEpochs"])
        
        # Track epoch metrics for history
        epochMetrics = []
        
        # Training loop
        bestAde = float("inf")
        bestModel = None
        
        for epoch in range(args.numEpochs):
            trainLoss = trainEpochForTune(model, trainLoader, optimizer, loss_fn, device, scaler, epoch, trial.number)
            
            # Check if training failed (all batches errored)
            if trainLoss == float("inf"):
                print(f"  Trial {trial.number}: Training failed (all batches errored), pruning trial")
                if historyTracker:
                    historyTracker.addTrial(
                        trialNumber=trial.number,
                        params=trial.params,
                        value=float("inf"),
                        state="FAILED",
                        epochMetrics=epochMetrics
                    )
                raise optuna.exceptions.TrialPruned()
            
            valMetrics = validateForTune(model, valLoader, loss_fn, postProcess, device, trial.number)
            
            # Check if validation failed
            if valMetrics.get("ade", float("inf")) == float("inf"):
                print(f"  Trial {trial.number}: Validation failed (all batches errored), pruning trial")
                if historyTracker:
                    historyTracker.addTrial(
                        trialNumber=trial.number,
                        params=trial.params,
                        value=float("inf"),
                        state="FAILED",
                        epochMetrics=epochMetrics
                    )
                raise optuna.exceptions.TrialPruned()
            
            scheduler.step()
            
            # Store epoch metrics
            epochMetrics.append({
                "epoch": epoch + 1,
                "trainLoss": trainLoss,
                "valLoss": valMetrics.get("loss", float("inf")),
                "ade": valMetrics.get("ade", float("inf")),
                "fde": valMetrics.get("fde", float("inf"))
            })
            
            # Report intermediate value
            currentAde = valMetrics.get("ade", float("inf"))
            trial.report(currentAde, epoch)
            
            if currentAde < bestAde:
                bestAde = currentAde
                # Save best model state for this trial
                if modelSaveDir is not None:
                    bestModel = {
                        "modelState": model.state_dict(),
                        "optimState": optimizer.state_dict(),
                        "epoch": epoch + 1,
                        "ade": currentAde,
                        "params": trial.params,
                        "config": config
                    }
            
            # Pruning
            if trial.should_prune():
                # Update history with pruned trial
                if historyTracker:
                    historyTracker.addTrial(
                        trialNumber=trial.number,
                        params=trial.params,
                        value=bestAde,
                        state="PRUNED",
                        epochMetrics=epochMetrics
                    )
                raise optuna.exceptions.TrialPruned()
            
            gc.collect()
            clearGpuMemory()
        
        # Save best model for this trial if requested
        if modelSaveDir is not None and bestModel is not None:
            trialDir = modelSaveDir / "trial_models"
            trialDir.mkdir(parents=True, exist_ok=True)
            torch.save(bestModel, trialDir / f"trial_{trial.number}.pth")
        
        # Update history tracker
        if historyTracker:
            historyTracker.addTrial(
                trialNumber=trial.number,
                params=trial.params,
                value=bestAde,
                state="COMPLETE",
                epochMetrics=epochMetrics
            )
            historyTracker.save("hpo_history")
        
        # ============================================================
        # COMPREHENSIVE CLEANUP (Memory Optimization)
        # ============================================================
        # Clear map encoder cache to prevent memory accumulation
        if hasattr(trainDataset, 'mapEncoder'):
            trainDataset.mapEncoder.clearCache()
        if hasattr(valDataset, 'mapEncoder'):
            valDataset.mapEncoder.clearCache()
        
        # Delete all large objects
        del model, loss_fn, postProcess, optimizer, scheduler
        del trainDataset, valDataset, trainLoader, valLoader
        if scaler is not None:
            del scaler
        
        # Force garbage collection and GPU memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        clearGpuMemory()
        # ============================================================
        
        return bestAde
    
    return objective


def runOptuna(args):
    """Run Optuna hyperparameter optimization with resume capability."""
    # Setup output directory
    outputDir = Path(args.outputDir) if args.outputDir else Path("outputs/hpo")
    outputDir.mkdir(parents=True, exist_ok=True)
    
    # SQLite storage for resume capability
    storagePath = outputDir / f"{args.studyName}.db"
    storageUrl = f"sqlite:///{storagePath}"
    
    # Create or load the study
    study = optuna.create_study(
        direction="minimize",
        study_name=args.studyName,
        storage=storageUrl,
        load_if_exists=True,
        sampler=TPESampler(n_startup_trials=5, multivariate=True, seed=HPO_SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=1)
    )
    
    totalTrials = len(study.trials)
    remainingTrials = max(0, args.numTrials - totalTrials)
    
    print(f"Study name: {args.studyName}")
    print(f"Storage path: {storagePath}")
    print(f"Found {totalTrials} total trials in database")
    print(f"Running {remainingTrials} trials to reach the target of {args.numTrials} trials")
    
    # Initialize history tracker
    historyTracker = HPOHistoryTracker(str(outputDir))
    historyTracker.setStudyName(args.studyName)
    
    # Load existing trial history if resuming
    if args.resume:
        for trial in study.trials:
            historyTracker.addTrial(
                trialNumber=trial.number,
                params=trial.params,
                value=trial.value if trial.value is not None else float("inf"),
                state=trial.state.name
            )
    
    # Create objective with model saving option
    objective = createObjective(args, outputDir if args.saveTrialModels else None, historyTracker)
    
    if remainingTrials > 0:
        study.optimize(
            objective,
            n_trials=remainingTrials,
            timeout=None,
            gc_after_trial=True,
            show_progress_bar=True
        )
    
    # Update history with all trials
    for trial in study.trials:
        historyTracker.addTrial(
            trialNumber=trial.number,
            params=trial.params,
            value=trial.value if trial.value is not None else float("inf"),
            state=trial.state.name
        )
    
    # Save history
    historyTracker.save("hpo_history")
    historyTracker.printSummary()
    
    # Print results
    print("\n" + "="*60)
    print("Optuna HPO Results")
    print("="*60)
    
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if len(complete_trials) == 0:
        print("WARNING: No trials completed successfully. All trials were pruned/failed.")
        print("Saving best params from pruned trials as fallback...")
        # Fallback: use params from the trial that reported the best intermediate value
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        best_pruned = None
        best_pruned_value = float("inf")
        for t in pruned_trials:
            if t.intermediate_values:
                min_val = min(t.intermediate_values.values())
                if min_val < best_pruned_value:
                    best_pruned_value = min_val
                    best_pruned = t
        if best_pruned is None and len(pruned_trials) > 0:
            best_pruned = pruned_trials[0]  # Just take first trial's params
            best_pruned_value = float("inf")
        if best_pruned is not None:
            print(f"Using pruned trial {best_pruned.number} (best intermediate: {best_pruned_value:.4f})")
            with open(outputDir / "best_params.json", "w") as f:
                json.dump({
                    "bestTrial": best_pruned.number,
                    "bestValue": best_pruned_value,
                    "bestParams": best_pruned.params,
                    "fallback": True
                }, f, indent=2)
        else:
            print("ERROR: No trials at all. Cannot save best params.")
        print(f"\nResults saved to: {outputDir}")
        print(f"Study database: {storagePath}")
        return study
    
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value (minADE): {study.best_value:.4f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save best params
    with open(outputDir / "best_params.json", "w") as f:
        json.dump({
            "bestTrial": study.best_trial.number,
            "bestValue": study.best_value,
            "bestParams": study.best_params
        }, f, indent=2)
    
    # Print top 5 trials for analysis
    if len(complete_trials) > 0:
        print(f"\n{'='*60}")
        print("TOP 5 TRIALS")
        print("=" * 60)
        
        sorted_trials = sorted(complete_trials, key=lambda t: t.value)[:5]
        for i, t in enumerate(sorted_trials):
            lr_val = t.params.get('lr', 'N/A')
            lr_str = f"{lr_val:.2e}" if isinstance(lr_val, float) else str(lr_val)
            print(f"  #{i+1} Trial {t.number}: minADE={t.value:.6f}, "
                  f"lr={lr_str}, "
                  f"nActor={t.params.get('nActor', 'N/A')}, "
                  f"nMap={t.params.get('nMap', 'N/A')}")
    
    print(f"\nResults saved to: {outputDir}")
    print(f"Study database: {storagePath}")
    print(f"HPO history: {outputDir / 'hpo_history.json'}")
    
    return study


def runRayTune(args):
    """Run Ray Tune hyperparameter optimization."""
    if not HAS_RAY:
        raise ImportError("Ray Tune not available")
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    # Define search space
    searchSpace = {
        "lr": tune.loguniform(1e-4, 1e-3),
        "weightDecay": tune.loguniform(1e-6, 1e-4),
        "batchSize": tune.choice([8, 16, 32]),
        "nActor": tune.choice([64, 128, 256]),
        "nMap": tune.choice([64, 128, 256]),
        "numScales": tune.randint(4, 9)
    }
    
    def trainable(config: dict):
        """Trainable function for Ray Tune."""
        baseConfig = getHPOConfig(args.datasetType)  # Uses Stratified dataset (20% sample)
        baseConfig["numEpochs"] = args.numEpochs
        
        # Update with sampled hyperparameters
        baseConfig["lr"] = [config["lr"], config["lr"] / 10]
        baseConfig["batchSize"] = config["batchSize"]
        baseConfig["valBatchSize"] = config["batchSize"]
        baseConfig["nActor"] = config["nActor"]
        baseConfig["nMap"] = config["nMap"]
        baseConfig["numScales"] = config["numScales"]
        
        setSeed(args.seed)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create datasets (use all Stratified chunks - already 20% sample)
        trainDataset = PrayagDataset(baseConfig, split="train", testChunks=args.testChunks)
        valDataset = PrayagDataset(baseConfig, split="val", testChunks=args.testChunks)
        
        trainLoader = DataLoader(
            trainDataset,
            batch_size=config["batchSize"],
            shuffle=True,
            num_workers=2,
            collate_fn=collateFn,
            drop_last=True
        )
        valLoader = DataLoader(
            valDataset,
            batch_size=config["batchSize"],
            shuffle=False,
            num_workers=2,
            collate_fn=collateFn
        )
        
        # Create model
        model = GameFormer(baseConfig).to(device)
        loss_fn = GameFormerLoss(baseConfig).to(device)
        postProcess = PostProcess(baseConfig).to(device)
        
        optimizer = getOptimizer(
            model.parameters(),
            lr=config["lr"],
            weightDecay=config["weightDecay"]
        )
        scheduler = getScheduler(optimizer, baseConfig["lrEpochs"])
        
        # Training loop
        for epoch in range(args.numEpochs):
            trainLoss = trainEpochForTune(model, trainLoader, optimizer, loss_fn, device)
            valMetrics = validateForTune(model, valLoader, loss_fn, postProcess, device)
            
            scheduler.step()
            
            # Report to Ray
            tune.report(
                loss=valMetrics.get("loss", float("inf")),
                ade=valMetrics.get("ade", float("inf")),
                fde=valMetrics.get("fde", float("inf"))
            )
            
            gc.collect()
            clearGpuMemory()
    
    # ASHA scheduler for early stopping
    scheduler = ASHAScheduler(
        metric="ade",
        mode="min",
        max_t=args.numEpochs,
        grace_period=2,
        reduction_factor=2
    )
    
    # Optuna search
    optunaSearch = OptunaSearch(
        metric="ade",
        mode="min"
    )
    
    # Run tuning
    analysis = tune.run(
        trainable,
        config=searchSpace,
        num_samples=args.numTrials,
        scheduler=scheduler,
        search_alg=optunaSearch,
        resources_per_trial={"cpu": 2, "gpu": args.numGpus / 2},
        local_dir=args.outputDir or tempfile.gettempdir(),
        name="prayag_gameformer_tune"
    )
    
    # Print results
    print("\n" + "="*60)
    print("Ray Tune HPO Results")
    print("="*60)
    print(f"Best trial: {analysis.best_trial}")
    print(f"Best ADE: {analysis.best_result['ade']:.4f}")
    print("Best hyperparameters:")
    for key, value in analysis.best_config.items():
        print(f"  {key}: {value}")
    
    ray.shutdown()
    
    return analysis


def main():
    """Main tuning function."""
    args = parseArgs()
    
    # Configure PyTorch to use all available CPU cores for intra-op parallelism
    cpu_count = os.cpu_count() or 1
    torch.set_num_threads(min(cpu_count, 40))  # Cap at 40 cores
    torch.set_num_interop_threads(min(cpu_count // 4, 10))  # For inter-op parallelism
    
    print("="*60)
    print("PrayagGameFormer Hyperparameter Optimization")
    print("="*60)
    
    print(f"\nSystem Configuration:")
    print(f"  Platform: {platform.system()} ({platform.machine()})")
    print(f"  CPU cores: {cpu_count}")
    print(f"  PyTorch threads: {torch.get_num_threads()}")
    
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print(f"\nHPO Configuration:")
    print(f"  Backend: {args.backend}")
    print(f"  Study name: {args.studyName}")
    print(f"  Trials: {args.numTrials}")
    print(f"  Epochs per trial: {args.numEpochs}")
    print(f"  Test chunks: {args.testChunks}")
    print(f"  Resume: {args.resume}")
    print(f"  Save trial models: {args.saveTrialModels}")
    print(f"  Output dir: {args.outputDir}")
    print("="*60)
    
    if args.backend == "optuna":
        runOptuna(args)
    elif args.backend == "ray":
        if not HAS_RAY:
            print("Ray not available, falling back to Optuna")
            runOptuna(args)
        else:
            runRayTune(args)
    
    print("\nHPO complete!")


if __name__ == "__main__":
    # Multiprocessing safety for Windows and macOS
    if platform.system() == 'Windows':
        multiprocessing.freeze_support()
    
    # Set multiprocessing start method for CUDA compatibility
    try:
        multiprocessing.set_start_method('spawn', force=False)
    except RuntimeError:
        pass  # Already set
    
    main()
