"""
Comprehensive evaluation script for PrayagGameFormer.

Computes all metrics required for comparison:
- minADE@1, minADE@4, minFDE@1, minFDE@4
- Miss Rate @10px, @20px
- Normalized FDE
- APD (Diversity)
- NLL (Probabilistic)
- Collision Rate
- Off-Road Rate
- MSS (Model Selection Score)

Saves results in JSON/CSV format for plotting.
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent))

from config import getConfig, getTestConfig
from data import PrayagDataset, collateFn
from model.gameformer import GameFormer, GameFormerLoss, PostProcess
from utils import setSeed
from utils.helpers import loadCheckpoint, clearGpuMemory, toNative
from utils.metrics import (
    MetricsAccumulator,
    computeMinADEK, computeMinFDEK,
    computeMissRate, computeNormFDE, computeAPD,
    computeNLL, computeCollisionRate, computeOffRoadRate, computeMSS
)


def parseArgs():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate PrayagGameFormer")
    
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to checkpoint to evaluate"
    )
    parser.add_argument(
        "--datasetType", type=str, default="10Hz",
        choices=["10Hz", "30Hz"],
        help="Dataset type"
    )
    parser.add_argument(
        "--split", type=str, default="test",
        choices=["train", "val", "test"],
        help="Data split to evaluate"
    )
    parser.add_argument(
        "--outputDir", type=str, default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--batchSize", type=int, default=8,
        help="Batch size"
    )
    parser.add_argument(
        "--testChunks", type=int, default=None,
        help="Limit to N chunks for testing"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (cuda or cpu)"
    )
    parser.add_argument(
        "--saveRaw", action="store_true",
        help="Save raw predictions for analysis"
    )
    
    return parser.parse_args()


def loadRoadMask(datasetDir: str, chunkName: str) -> Optional[np.ndarray]:
    """Load road mask for a chunk."""
    datasetPath = Path(datasetDir)
    
    for split in ["train", "val", "test"]:
        maskPath = datasetPath / split / "annotations" / f"{chunkName}_road_mask.png"
        if maskPath.exists():
            mask = cv2.imread(str(maskPath), cv2.IMREAD_GRAYSCALE)
            return (mask > 0).astype(np.uint8)
    return None


def extractPredictions(out: dict, config: dict) -> np.ndarray:
    """
    Extract predictions from GameFormer output.
    
    GameFormer outputs hierarchical predictions with GMM parameters.
    We use the final level and extract only the mean (x, y).
    
    Args:
        out: Model output dictionary
        config: Configuration
        
    Returns:
        Predictions [B, numModes, predLen, 2] (x, y only)
    """
    # Use highest level predictions
    numLevels = config.get("decoderLevels", 3)
    predKey = f"level_{numLevels}_interactions"
    
    if predKey in out:
        preds = out[predKey]  # [B, N+1, numModes, predLen, 4] (mu_x, mu_y, log_sigma_x, log_sigma_y)
        # Take ego (index 0) predictions
        preds = preds[:, 0]  # [B, numModes, predLen, 4]
    else:
        # Fallback to level 0
        preds = out.get("level_0_interactions", None)
        if preds is not None:
            preds = preds[:, 0]
    
    if preds is None:
        raise ValueError("No predictions found in output")
    
    # Extract only mean (x, y), not the sigma values
    preds = preds[..., :2]  # [B, numModes, predLen, 2]
    
    return preds.detach().cpu().numpy()


def extractScores(out: dict, config: dict) -> Optional[np.ndarray]:
    """Extract mode scores from GameFormer output."""
    numLevels = config.get("decoderLevels", 3)
    scoreKey = f"level_{numLevels}_scores"
    
    if scoreKey in out:
        scores = out[scoreKey]  # [B, N+1, numModes]
        scores = scores[:, 0]  # [B, numModes]
        return scores.detach().cpu().numpy()
    return None


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    postProcess: torch.nn.Module,
    device: str,
    config: dict,
    saveRaw: bool = False
) -> Dict[str, Any]:
    """
    Evaluate model comprehensively.
    
    Args:
        model: Model to evaluate
        loader: Data loader
        loss_fn: Loss function
        postProcess: Post-processor
        device: Device
        config: Configuration
        saveRaw: Whether to save raw predictions
        
    Returns:
        Dictionary of results
    """
    model.eval()
    
    obsLen = config["obsHorizon"]
    predLen = config["predHorizon"]
    datasetDir = config["datasetDir"]
    
    # Accumulators
    accumulator = MetricsAccumulator(obsLen=obsLen, predLen=predLen)
    
    losses = []
    rawPreds = [] if saveRaw else None
    rawGts = [] if saveRaw else None
    
    # Road mask cache
    roadMaskCache = {}
    
    pbar = tqdm(loader, desc="Evaluating")
    
    with torch.no_grad():
        for batch in pbar:
            try:
                # Move to device
                batchDevice = {}
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batchDevice[k] = v.to(device)
                    else:
                        batchDevice[k] = v
                
                # Forward pass
                out = model(batchDevice)
                lossOut = loss_fn(out, batchDevice)
                
                losses.append(lossOut["loss"].item())
                
                # Extract predictions
                preds = extractPredictions(out, config)
                scores = extractScores(out, config)
                
                # Ground truth
                gts = batch["ego_future"].cpu().numpy()  # [B, predLen, 2]
                
                # Neighbor futures for collision detection
                neighborFutures = None
                if "neighbors_future" in batch:
                    neighborFutures = batch["neighbors_future"].cpu().numpy()
                
                # Get road mask
                roadMask = None
                if "chunk" in batch:
                    chunks = batch["chunk"]
                    if isinstance(chunks, list):
                        chunkName = chunks[0]
                    else:
                        chunkName = chunks
                    
                    if chunkName not in roadMaskCache:
                        roadMaskCache[chunkName] = loadRoadMask(datasetDir, chunkName)
                    roadMask = roadMaskCache[chunkName]
                
                # Update accumulator
                accumulator.update(
                    preds=preds,
                    gtPreds=gts,
                    scores=scores,
                    neighborPreds=neighborFutures,
                    roadMask=roadMask
                )
                
                # Save raw data
                if saveRaw:
                    rawPreds.extend(preds.tolist())
                    rawGts.extend(gts.tolist())
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"WARNING: OOM, skipping batch")
                    clearGpuMemory()
                    continue
                raise e
    
    # Compute final metrics
    metrics = accumulator.compute()
    metrics["loss"] = np.mean(losses) if losses else 0.0
    
    # Add raw data if requested
    if saveRaw:
        metrics["rawPreds"] = rawPreds
        metrics["rawGts"] = rawGts
    
    return metrics


def main():
    """Main evaluation function."""
    args = parseArgs()
    
    # Get config
    if args.testChunks is not None:
        config = getTestConfig(args.testChunks)
    else:
        config = getConfig(args.datasetType)
    
    config["batchSize"] = args.batchSize
    config["valBatchSize"] = args.batchSize
    
    # Device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    
    # Output directory
    if args.outputDir:
        outputDir = Path(args.outputDir)
    else:
        outputDir = Path(config["saveDir"]) / "evaluation"
    outputDir.mkdir(parents=True, exist_ok=True)
    
    print(f"Config: datasetType={args.datasetType}, split={args.split}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {outputDir}")
    
    # Set seed
    setSeed(config.get("seed", 42))
    
    # Create dataset
    dataset = PrayagDataset(
        config, split=args.split,
        testChunks=config.get("testChunks")
    )
    
    loader = DataLoader(
        dataset,
        batch_size=config["batchSize"],
        shuffle=False,
        num_workers=config.get("workers", 4),
        collate_fn=collateFn,
        pin_memory=True
    )
    
    print(f"Dataset: {len(dataset)} samples, {len(loader)} batches")
    
    # Create model
    model = GameFormer(config).to(device)
    
    # Load checkpoint
    loadCheckpoint(model, None, args.checkpoint, device)
    print(f"Loaded checkpoint from {args.checkpoint}")
    
    numParams = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {numParams:,}")
    
    # Create loss and post-processor
    loss_fn = GameFormerLoss(config).to(device)
    postProcess = PostProcess(config)
    
    # Evaluate
    print("\nEvaluating...")
    startTime = time.time()
    
    results = evaluate(
        model, loader, loss_fn, postProcess, device, config,
        saveRaw=args.saveRaw
    )
    
    evalTime = time.time() - startTime
    results["evalTime"] = evalTime
    results["checkpoint"] = args.checkpoint
    results["datasetType"] = args.datasetType
    results["split"] = args.split
    results["timestamp"] = datetime.now().isoformat()
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Samples: {results.get('samples', 'N/A')}")
    print(f"Loss: {results.get('loss', 0):.4f}")
    print()
    print("Displacement Metrics:")
    print(f"  minADE@1: {results.get('minADE@1', 0):.2f} px")
    print(f"  minADE@4: {results.get('minADE@4', 0):.2f} px")
    print(f"  minFDE@1: {results.get('minFDE@1', 0):.2f} px")
    print(f"  minFDE@4: {results.get('minFDE@4', 0):.2f} px")
    print()
    print("Miss Rates:")
    print(f"  MR@10px: {results.get('MR@10px', 0)*100:.2f}%")
    print(f"  MR@20px: {results.get('MR@20px', 0)*100:.2f}%")
    print()
    print("Other Metrics:")
    print(f"  NormFDE: {results.get('NormFDE', 0):.4f}")
    print(f"  APD (Diversity): {results.get('APD', 0):.2f} px")
    print(f"  NLL: {results.get('NLL', 0):.4f}")
    print()
    print("Safety Metrics:")
    print(f"  Collision Rate: {results.get('CR', 0)*100:.2f}%")
    print(f"  Off-Road Rate: {results.get('ORR', 0)*100:.2f}%")
    print()
    print("Model Selection Score:")
    print(f"  MSS: {results.get('MSS', 0):.2f}")
    print()
    print(f"Evaluation time: {evalTime:.1f}s")
    print("="*60)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON (full results) - convert numpy types for serialization
    jsonPath = outputDir / f"eval_{args.split}_{timestamp}.json"
    resultsSave = {k: v for k, v in results.items() if k not in ["rawPreds", "rawGts"]}
    resultsSave = toNative(resultsSave)
    with open(jsonPath, "w") as f:
        json.dump(resultsSave, f, indent=2)
    print(f"Results saved to {jsonPath}")
    
    # CSV (metrics table)
    csvPath = outputDir / f"metrics_{args.split}_{timestamp}.csv"
    metricsTable = {
        "Metric": [
            "Samples", "Loss",
            "minADE@1", "minADE@4", "minFDE@1", "minFDE@4",
            "MR@10px", "MR@20px",
            "NormFDE", "APD", "NLL",
            "Collision Rate", "Off-Road Rate", "MSS"
        ],
        "Value": [
            results.get("samples", 0),
            f"{results.get('loss', 0):.4f}",
            f"{results.get('minADE@1', 0):.2f}",
            f"{results.get('minADE@4', 0):.2f}",
            f"{results.get('minFDE@1', 0):.2f}",
            f"{results.get('minFDE@4', 0):.2f}",
            f"{results.get('MR@10px', 0)*100:.2f}%",
            f"{results.get('MR@20px', 0)*100:.2f}%",
            f"{results.get('NormFDE', 0):.4f}",
            f"{results.get('APD', 0):.2f}",
            f"{results.get('NLL', 0):.4f}",
            f"{results.get('CR', 0)*100:.2f}%",
            f"{results.get('ORR', 0)*100:.2f}%",
            f"{results.get('MSS', 0):.2f}"
        ]
    }
    df = pd.DataFrame(metricsTable)
    df.to_csv(csvPath, index=False)
    print(f"Metrics table saved to {csvPath}")
    
    # Save raw predictions if requested
    if args.saveRaw and "rawPreds" in results:
        rawPath = outputDir / f"raw_predictions_{args.split}_{timestamp}.npz"
        np.savez(
            rawPath,
            predictions=np.array(results["rawPreds"]),
            groundTruth=np.array(results["rawGts"])
        )
        print(f"Raw predictions saved to {rawPath}")
    
    return results


if __name__ == "__main__":
    main()
