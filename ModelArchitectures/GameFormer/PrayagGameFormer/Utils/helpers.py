"""
Helper utilities for training.
"""

import os
import sys
import json
import random
import numpy as np
import torch
import torch.optim as optim
from typing import Optional, Dict, Any, List
from datetime import datetime


def toNative(d: Dict) -> Dict:
    """
    Convert numpy types to native Python types for JSON serialization.
    
    Args:
        d: Dictionary possibly containing numpy types
        
    Returns:
        Dictionary with native Python types
    """
    result = {}
    for k, v in d.items():
        if isinstance(v, np.floating):
            result[k] = float(v)
        elif isinstance(v, np.integer):
            result[k] = int(v)
        elif isinstance(v, np.ndarray):
            result[k] = v.tolist()
        elif isinstance(v, dict):
            result[k] = toNative(v)
        else:
            result[k] = v
    return result


def setSeed(seed: int) -> None:
    """Set random seed for reproducibility (non-deterministic for speed)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def getOptimizer(
    params,
    lr: float = 1e-3,
    weightDecay: float = 1e-4,
    optimType: str = "adamw"
) -> optim.Optimizer:
    """
    Get optimizer.
    
    Args:
        params: Model parameters
        lr: Learning rate
        weightDecay: Weight decay
        optimType: Optimizer type ("adam", "adamw", "sgd")
        
    Returns:
        Optimizer instance
    """
    if optimType == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weightDecay)
    elif optimType == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weightDecay)
    elif optimType == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=weightDecay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimType}")


def getScheduler(
    optimizer: optim.Optimizer,
    milestones: list,
    gamma: float = 0.5
) -> optim.lr_scheduler.MultiStepLR:
    """
    Get learning rate scheduler.
    
    Args:
        optimizer: Optimizer instance
        milestones: Epochs at which to decay LR
        gamma: Decay factor
        
    Returns:
        Scheduler instance
    """
    return optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=milestones,
        gamma=gamma
    )


class Logger:
    """Logger that writes to both console and file."""
    
    def __init__(self, logPath: str):
        """
        Initialize logger.
        
        Args:
            logPath: Path to log file
        """
        os.makedirs(os.path.dirname(logPath), exist_ok=True)
        self.terminal = sys.stdout
        self.log = open(logPath, "a")
    
    def write(self, message: str) -> None:
        """Write message to both terminal and file."""
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self) -> None:
        """Flush outputs."""
        self.terminal.flush()
        self.log.flush()
    
    def close(self) -> None:
        """Close log file."""
        self.log.close()


def loadCheckpoint(
    model: torch.nn.Module,
    optimizer: Optional[optim.Optimizer],
    checkpointPath: str,
    device: str = "cuda",
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
) -> Dict[str, Any]:
    """
    Load checkpoint.
    
    Args:
        model: Model to load weights into
        optimizer: Optional optimizer to load state
        checkpointPath: Path to checkpoint
        device: Device to load to
        scheduler: Optional scheduler to load state
        
    Returns:
        Dictionary with epoch number, metrics, and training history
    """
    checkpoint = torch.load(checkpointPath, map_location=device)
    model.load_state_dict(checkpoint["modelState"])
    
    if optimizer is not None and "optimState" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimState"])
    
    if scheduler is not None and "schedulerState" in checkpoint:
        scheduler.load_state_dict(checkpoint["schedulerState"])
    
    return {
        "epoch": checkpoint.get("epoch", 0),
        "metrics": checkpoint.get("metrics", {}),
        "history": checkpoint.get("history", {"train": [], "val": []}),
        "bestMetric": checkpoint.get("bestMetric", float("inf")),
        "config": checkpoint.get("config", {})
    }


def saveCheckpoint(
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    savePath: str,
    metrics: Optional[dict] = None,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    history: Optional[Dict[str, List]] = None,
    bestMetric: Optional[float] = None,
    config: Optional[dict] = None
) -> None:
    """
    Save checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        savePath: Path to save checkpoint
        metrics: Optional metrics to save
        scheduler: Optional scheduler to save
        history: Optional training history
        bestMetric: Optional best metric value
        config: Optional config dict
    """
    os.makedirs(os.path.dirname(savePath), exist_ok=True)
    
    state = {
        "epoch": epoch,
        "modelState": model.state_dict(),
        "optimState": optimizer.state_dict()
    }
    
    if scheduler is not None:
        state["schedulerState"] = scheduler.state_dict()
    
    if metrics is not None:
        state["metrics"] = metrics
    
    if history is not None:
        state["history"] = history
    
    if bestMetric is not None:
        state["bestMetric"] = bestMetric
    
    if config is not None:
        state["config"] = config
    
    torch.save(state, savePath)


def clearGpuMemory() -> None:
    """Clear GPU memory cache aggressively."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except RuntimeError:
            # If we get CUDA errors, reset the device
            pass


class TrainingHistory:
    """
    Class to track and save training history for visualization.
    
    Tracks metrics per epoch for both training and validation,
    and can export to JSON/CSV for plotting.
    """
    
    def __init__(self, outputDir: str):
        """
        Initialize training history tracker.
        
        Args:
            outputDir: Directory to save history files
        """
        self.outputDir = outputDir
        os.makedirs(outputDir, exist_ok=True)
        
        self.history = {
            "train": [],
            "val": [],
            "metadata": {
                "startTime": datetime.now().isoformat(),
                "lastUpdated": None
            }
        }
    
    def addEpoch(
        self,
        epoch: int,
        trainMetrics: Dict[str, float],
        valMetrics: Optional[Dict[str, float]] = None,
        lr: Optional[float] = None
    ) -> None:
        """
        Add metrics for an epoch.
        
        Args:
            epoch: Epoch number (1-indexed)
            trainMetrics: Training metrics dict
            valMetrics: Optional validation metrics dict
            lr: Optional current learning rate
        """
        # Convert numpy types to Python types for JSON serialization
        def toNative(d: Dict) -> Dict:
            result = {}
            for k, v in d.items():
                if isinstance(v, np.floating):
                    result[k] = float(v)
                elif isinstance(v, np.integer):
                    result[k] = int(v)
                elif isinstance(v, np.ndarray):
                    result[k] = v.tolist()
                else:
                    result[k] = v
            return result
        
        trainEntry = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            **toNative(trainMetrics)
        }
        if lr is not None:
            trainEntry["lr"] = float(lr)
        
        self.history["train"].append(trainEntry)
        
        if valMetrics is not None:
            valEntry = {
                "epoch": epoch,
                "timestamp": datetime.now().isoformat(),
                **toNative(valMetrics)
            }
            self.history["val"].append(valEntry)
        
        self.history["metadata"]["lastUpdated"] = datetime.now().isoformat()
    
    def loadFromCheckpoint(self, checkpointHistory: Dict[str, List]) -> None:
        """
        Load history from checkpoint.
        
        Args:
            checkpointHistory: History dict from checkpoint
        """
        if "train" in checkpointHistory:
            self.history["train"] = checkpointHistory["train"]
        if "val" in checkpointHistory:
            self.history["val"] = checkpointHistory["val"]
    
    def getHistory(self) -> Dict[str, List]:
        """Get history dict for saving in checkpoint."""
        return {
            "train": self.history["train"],
            "val": self.history["val"]
        }
    
    def save(self, filename: str = "training_history") -> None:
        """
        Save history to JSON and CSV files.
        
        Args:
            filename: Base filename (without extension)
        """
        # Save as JSON
        jsonPath = os.path.join(self.outputDir, f"{filename}.json")
        with open(jsonPath, "w") as f:
            json.dump(self.history, f, indent=2)
        
        # Save train history as CSV
        if self.history["train"]:
            trainCsvPath = os.path.join(self.outputDir, f"{filename}_train.csv")
            self._saveCsv(self.history["train"], trainCsvPath)
        
        # Save val history as CSV
        if self.history["val"]:
            valCsvPath = os.path.join(self.outputDir, f"{filename}_val.csv")
            self._saveCsv(self.history["val"], valCsvPath)
    
    def _saveCsv(self, data: List[Dict], path: str) -> None:
        """Save list of dicts to CSV."""
        if not data:
            return
        
        # Get all keys
        keys = []
        for entry in data:
            for key in entry.keys():
                if key not in keys:
                    keys.append(key)
        
        with open(path, "w") as f:
            # Header
            f.write(",".join(keys) + "\n")
            
            # Data rows
            for entry in data:
                values = [str(entry.get(k, "")) for k in keys]
                f.write(",".join(values) + "\n")
    
    def getBest(self, metric: str = "loss", mode: str = "min", split: str = "val") -> Dict[str, Any]:
        """
        Get best epoch based on a metric.
        
        Args:
            metric: Metric name to compare
            mode: "min" or "max"
            split: "train" or "val"
            
        Returns:
            Dict with best epoch info
        """
        data = self.history.get(split, [])
        if not data:
            return {}
        
        bestEntry = None
        bestValue = float("inf") if mode == "min" else float("-inf")
        
        for entry in data:
            if metric not in entry:
                continue
            value = entry[metric]
            if (mode == "min" and value < bestValue) or \
               (mode == "max" and value > bestValue):
                bestValue = value
                bestEntry = entry
        
        return bestEntry if bestEntry else {}
    
    def printSummary(self) -> None:
        """Print summary of training history."""
        print("\n" + "="*60)
        print("TRAINING HISTORY SUMMARY")
        print("="*60)
        
        numEpochs = len(self.history["train"])
        print(f"Total epochs: {numEpochs}")
        
        if self.history["train"]:
            lastTrain = self.history["train"][-1]
            print(f"\nLast training metrics (epoch {lastTrain.get('epoch', numEpochs)}):")
            for key, value in lastTrain.items():
                if key not in ["epoch", "timestamp"]:
                    if isinstance(value, float):
                        print(f"  {key}: {value:.6f}")
                    else:
                        print(f"  {key}: {value}")
        
        if self.history["val"]:
            lastVal = self.history["val"][-1]
            print(f"\nLast validation metrics:")
            for key, value in lastVal.items():
                if key not in ["epoch", "timestamp"]:
                    if isinstance(value, float):
                        print(f"  {key}: {value:.6f}")
                    else:
                        print(f"  {key}: {value}")
        
        # Best metrics
        bestValLoss = self.getBest("loss", "min", "val")
        if bestValLoss:
            print(f"\nBest validation loss: {bestValLoss.get('loss', 'N/A'):.6f} " + 
                  f"(epoch {bestValLoss.get('epoch', 'N/A')})")
        
        print("="*60)
