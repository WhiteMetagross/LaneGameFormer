"""
Unified comparison script for LaneGameFormer, PrayagGameFormer, and PrayagLaneGCN.

This script runs comparative evaluation following the strategic comparison plan,
testing all three models on the ChunkedProjectPrayagBEVDataset.
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Models will be imported dynamically based on availability


def parseArgs():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare trajectory prediction models")
    
    parser.add_argument(
        "--datasetType", type=str, default="10Hz",
        choices=["10Hz", "30Hz"],
        help="Dataset type"
    )
    parser.add_argument(
        "--testChunks", type=int, default=1,
        help="Number of chunks for testing"
    )
    parser.add_argument(
        "--models", type=str, nargs="+",
        default=["lanegameformer", "gameformer", "lanegcn"],
        help="Models to compare"
    )
    parser.add_argument(
        "--checkpointDir", type=str, default=None,
        help="Directory containing model checkpoints"
    )
    parser.add_argument(
        "--outputDir", type=str, default="comparison_results",
        help="Directory for output results"
    )
    parser.add_argument(
        "--batchSize", type=int, default=4,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    
    return parser.parse_args()


def loadLaneGameFormer(config: dict, checkpointPath: Optional[str], device: str):
    """Load LaneGameFormer model."""
    try:
        sys.path.insert(0, str(Path(__file__).parent / "LaneGameFormer"))
        from model.lane_gameformer import LaneGameFormer
        from model.loss import LaneGameFormerLoss
        
        model = LaneGameFormer(config)
        if checkpointPath and os.path.exists(checkpointPath):
            checkpoint = torch.load(checkpointPath, map_location=device)
            model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        
        model = model.to(device)
        model.eval()
        
        return model, LaneGameFormerLoss(config).to(device)
    except ImportError as e:
        print(f"Could not load LaneGameFormer: {e}")
        return None, None


def loadPrayagGameFormer(config: dict, checkpointPath: Optional[str], device: str):
    """Load PrayagGameFormer model."""
    try:
        sys.path.insert(0, str(Path(__file__).parent / "PrayagGameFormer"))
        from model.gameformer import GameFormer, GameFormerLoss, PostProcess
        
        model = GameFormer(config)
        if checkpointPath and os.path.exists(checkpointPath):
            checkpoint = torch.load(checkpointPath, map_location=device)
            model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        
        model = model.to(device)
        model.eval()
        
        return model, GameFormerLoss(config).to(device), PostProcess(config).to(device)
    except ImportError as e:
        print(f"Could not load PrayagGameFormer: {e}")
        return None, None, None


def loadPrayagLaneGCN(config: dict, checkpointPath: Optional[str], device: str):
    """Load PrayagLaneGCN model."""
    try:
        sys.path.insert(0, str(Path(__file__).parent / "PrayagLaneGCN"))
        from model.lanegcn import LaneGCN, getLoss, getPostProcess
        
        model = LaneGCN(config)
        if checkpointPath and os.path.exists(checkpointPath):
            checkpoint = torch.load(checkpointPath, map_location=device)
            model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        
        model = model.to(device)
        model.eval()
        
        return model, getLoss(config).to(device), getPostProcess(config).to(device)
    except ImportError as e:
        print(f"Could not load PrayagLaneGCN: {e}")
        return None, None, None


def evaluateModel(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    postProcess: torch.nn.Module,
    device: str,
    modelName: str
) -> Dict[str, float]:
    """
    Evaluate a model on the dataset.
    
    Returns:
        Dictionary with metrics
    """
    model.eval()
    
    allAde = []
    allFde = []
    allMinAde = []
    allMinFde = []
    allLoss = []
    totalTime = 0.0
    numSamples = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Evaluating {modelName}")
        
        for batch in pbar:
            # Move to device
            if isinstance(batch, dict):
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
            
            startTime = time.time()
            
            try:
                # Forward pass
                out = model(batch)
                
                # Compute loss
                if loss_fn is not None:
                    lossOut = loss_fn(out, batch)
                    allLoss.append(lossOut["loss"].item())
                
                # Compute metrics
                if postProcess is not None:
                    postOut = postProcess(out, batch)
                    allAde.append(postOut.get("ade", 0))
                    allFde.append(postOut.get("fde", 0))
                    allMinAde.append(postOut.get("minAde", postOut.get("ade", 0)))
                    allMinFde.append(postOut.get("minFde", postOut.get("fde", 0)))
                
                totalTime += time.time() - startTime
                numSamples += 1
                
            except Exception as e:
                print(f"Error evaluating batch: {e}")
                continue
            
            pbar.set_postfix({
                "ade": f"{np.mean(allAde) if allAde else 0:.4f}",
                "fde": f"{np.mean(allFde) if allFde else 0:.4f}"
            })
    
    # Aggregate metrics
    results = {
        "model": modelName,
        "ade": float(np.mean(allAde)) if allAde else 0.0,
        "fde": float(np.mean(allFde)) if allFde else 0.0,
        "minAde": float(np.mean(allMinAde)) if allMinAde else 0.0,
        "minFde": float(np.mean(allMinFde)) if allMinFde else 0.0,
        "loss": float(np.mean(allLoss)) if allLoss else 0.0,
        "avgTimePerBatch": totalTime / max(numSamples, 1),
        "numSamples": numSamples
    }
    
    return results


def runComparison(args):
    """Run the comparison between models."""
    baseDir = Path(__file__).parent
    outputDir = Path(args.outputDir)
    outputDir.mkdir(parents=True, exist_ok=True)
    
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    
    results = []
    
    # Evaluate each model
    for modelName in args.models:
        modelName = modelName.lower()
        print(f"\n{'='*60}")
        print(f"Evaluating: {modelName}")
        print(f"{'='*60}")
        
        if modelName in ["lanegcn", "prayaglanegcn"]:
            # Load PrayagLaneGCN
            sys.path.insert(0, str(baseDir / "PrayagLaneGCN"))
            try:
                from config import getTestConfig
                from data import PrayagDataset, collateFn
                
                config = getTestConfig(args.testChunks)
                config["batchSize"] = args.batchSize
                
                checkpointPath = None
                if args.checkpointDir:
                    checkpointPath = Path(args.checkpointDir) / "lanegcn" / "best.pth"
                
                model, loss_fn, postProcess = loadPrayagLaneGCN(config, str(checkpointPath) if checkpointPath else None, device)
                
                if model is not None:
                    dataset = PrayagDataset(config, split="val", testChunks=args.testChunks)
                    loader = DataLoader(
                        dataset,
                        batch_size=args.batchSize,
                        shuffle=False,
                        num_workers=2,
                        collate_fn=collateFn
                    )
                    
                    result = evaluateModel(model, loader, loss_fn, postProcess, device, "PrayagLaneGCN")
                    results.append(result)
                    
                    del model, loss_fn, postProcess, dataset, loader
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
            except Exception as e:
                print(f"Error loading PrayagLaneGCN: {e}")
                import traceback
                traceback.print_exc()
        
        elif modelName in ["gameformer", "prayaggameformer"]:
            # Load PrayagGameFormer
            sys.path.insert(0, str(baseDir / "PrayagGameFormer"))
            try:
                from config import getTestConfig
                from data import PrayagDataset, collateFn
                
                config = getTestConfig(args.testChunks)
                config["batchSize"] = args.batchSize
                
                checkpointPath = None
                if args.checkpointDir:
                    checkpointPath = Path(args.checkpointDir) / "gameformer" / "best.pth"
                
                model, loss_fn, postProcess = loadPrayagGameFormer(config, str(checkpointPath) if checkpointPath else None, device)
                
                if model is not None:
                    dataset = PrayagDataset(config, split="val", testChunks=args.testChunks)
                    loader = DataLoader(
                        dataset,
                        batch_size=args.batchSize,
                        shuffle=False,
                        num_workers=2,
                        collate_fn=collateFn
                    )
                    
                    result = evaluateModel(model, loader, loss_fn, postProcess, device, "PrayagGameFormer")
                    results.append(result)
                    
                    del model, loss_fn, postProcess, dataset, loader
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
            except Exception as e:
                print(f"Error loading PrayagGameFormer: {e}")
                import traceback
                traceback.print_exc()
        
        elif modelName == "lanegameformer":
            # Load LaneGameFormer
            sys.path.insert(0, str(baseDir / "LaneGameFormer"))
            try:
                # Try to import LaneGameFormer modules
                from configs.default import getConfig
                from data.chunked_dataset import ChunkedDataset, collateFn
                from model.lane_gameformer import LaneGameFormer
                from model.loss import LaneGameFormerLoss, PostProcess
                
                config = getConfig()
                config["testChunks"] = args.testChunks
                config["batchSize"] = args.batchSize
                
                checkpointPath = None
                if args.checkpointDir:
                    checkpointPath = Path(args.checkpointDir) / "lanegameformer" / "best.pth"
                
                model = LaneGameFormer(config)
                if checkpointPath and os.path.exists(checkpointPath):
                    checkpoint = torch.load(str(checkpointPath), map_location=device)
                    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
                
                model = model.to(device)
                model.eval()
                
                loss_fn = LaneGameFormerLoss(config).to(device)
                postProcess = PostProcess(config).to(device)
                
                dataset = ChunkedDataset(config, split="val", testChunks=args.testChunks)
                loader = DataLoader(
                    dataset,
                    batch_size=args.batchSize,
                    shuffle=False,
                    num_workers=2,
                    collate_fn=collateFn
                )
                
                result = evaluateModel(model, loader, loss_fn, postProcess, device, "LaneGameFormer")
                results.append(result)
                
                del model, loss_fn, postProcess, dataset, loader
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                print(f"Error loading LaneGameFormer: {e}")
                import traceback
                traceback.print_exc()
    
    # Print summary
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print(f"{'Model':<20} {'ADE':>10} {'FDE':>10} {'minADE':>10} {'minFDE':>10} {'Loss':>10}")
    print("-"*80)
    
    for r in results:
        print(f"{r['model']:<20} {r['ade']:>10.4f} {r['fde']:>10.4f} "
              f"{r['minAde']:>10.4f} {r['minFde']:>10.4f} {r['loss']:>10.4f}")
    
    print("="*80)
    
    # Save results
    outputPath = outputDir / "comparison_results.json"
    with open(outputPath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {outputPath}")
    
    return results


def main():
    """Main function."""
    args = parseArgs()
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("="*60)
    print("Model Comparison on ChunkedProjectPrayagBEVDataset")
    print("="*60)
    print(f"Dataset: {args.datasetType}")
    print(f"Test chunks: {args.testChunks}")
    print(f"Models: {args.models}")
    print(f"Device: {args.device}")
    
    runComparison(args)


if __name__ == "__main__":
    main()
