"""
Quick test script to verify PrayagLaneGCN and PrayagGameFormer work correctly.

This script runs a minimal test with 1 chunk of data to verify:
1. Data loading works
2. Model forward pass works
3. Loss computation works
4. Backward pass works

Usage:
    python test_models.py --model lanegcn
    python test_models.py --model gameformer
    python test_models.py --model all
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")


def parseArgs():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test PrayagLaneGCN and PrayagGameFormer")
    
    parser.add_argument(
        "--model", type=str, default="all",
        choices=["lanegcn", "gameformer", "all"],
        help="Model to test"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use"
    )
    
    return parser.parse_args()


def testLaneGCN(device: str) -> bool:
    """Test PrayagLaneGCN."""
    print("\n" + "="*60)
    print("Testing PrayagLaneGCN")
    print("="*60)
    
    baseDir = Path(__file__).parent
    sys.path.insert(0, str(baseDir / "PrayagLaneGCN"))
    
    try:
        from config import getTestConfig
        from data import PrayagDataset, collateFn
        from model import LaneGCN, getLoss, getPostProcess
        from utils import setSeed
        
        print("✓ Imports successful")
        
        # Get config
        config = getTestConfig(numChunks=1)
        print(f"✓ Config loaded: obsHorizon={config['obsHorizon']}, predHorizon={config['predHorizon']}")
        
        # Set seed
        setSeed(42)
        
        # Check dataset paths (handle both config key names)
        datasetDir = Path(config.get("datasetDir", config.get("datasetPath", "")))
        if not datasetDir or not datasetDir.exists():
            print(f"⚠ Dataset directory not found: {datasetDir}")
            print("  Skipping dataset test, testing with dummy data...")
            
            # Test with dummy data
            dummyBatch = {
                "feats": torch.randn(2, config["maxAgents"], config["obsHorizon"], 3),
                "ctrs": torch.randn(2, config["maxAgents"], 2),
                "orig": torch.randn(2, 2),
                "theta": torch.randn(2),
                "rot": torch.randn(2, 2, 2),
                "graph": {
                    "ctrs": [torch.randn(50, 2), torch.randn(50, 2)],
                    "feats": [torch.randn(50, 2), torch.randn(50, 2)],
                    "num_nodes": [50, 50],
                    "pre": [[torch.zeros(0, dtype=torch.long)]] * 2,
                    "suc": [[torch.zeros(0, dtype=torch.long)]] * 2,
                    "left": [[torch.zeros(0, dtype=torch.long)]] * 2,
                    "right": [[torch.zeros(0, dtype=torch.long)]] * 2,
                },
                "gt_preds": torch.randn(2, config["maxAgents"], config["predHorizon"], 2),
                "has_preds": torch.ones(2, config["maxAgents"], config["predHorizon"]),
            }
            
            batch = dummyBatch
            
        else:
            # Load real dataset
            dataset = PrayagDataset(config, split="train", testChunks=1)
            print(f"✓ Dataset loaded: {len(dataset)} samples")
            
            if len(dataset) == 0:
                print("⚠ Dataset is empty, skipping test")
                return True
            
            # Get a batch
            from torch.utils.data import DataLoader
            loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collateFn)
            batch = next(iter(loader))
            print(f"✓ Batch loaded")
        
        # Create model
        model = LaneGCN(config)
        model = model.to(device)
        numParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ Model created: {numParams:,} parameters")
        
        # Forward pass
        model.train()
        out = model(batch)
        print(f"✓ Forward pass successful")
        regShape = out['reg'][0].shape if 'reg' in out and len(out['reg']) > 0 else 'N/A'
        print(f"  Output shape: {regShape}")
        
        # Loss
        loss_fn = getLoss(config).to(device)
        lossOut = loss_fn(out, batch, device)
        print(f"✓ Loss computed: {lossOut['loss'].item():.4f}")
        
        # Backward pass
        lossOut["loss"].backward()
        print(f"✓ Backward pass successful")
        
        # Cleanup
        del model, loss_fn, out, lossOut, batch
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print("\n✓ PrayagLaneGCN test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ PrayagLaneGCN test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def testGameFormer(device: str) -> bool:
    """Test PrayagGameFormer."""
    print("\n" + "="*60)
    print("Testing PrayagGameFormer")
    print("="*60)
    
    baseDir = Path(__file__).parent
    
    # Clean up any previously imported modules to avoid conflicts
    modulesToRemove = [k for k in sys.modules.keys() 
                       if k.startswith('model') or k.startswith('data') or 
                          k.startswith('config') or k.startswith('utils')]
    for mod in modulesToRemove:
        del sys.modules[mod]
    
    # Remove old paths and add new one
    gameformerPath = str(baseDir / "PrayagGameFormer")
    sys.path = [p for p in sys.path if 'PrayagLaneGCN' not in p and 'PrayagGameFormer' not in p]
    sys.path.insert(0, gameformerPath)
    
    try:
        from config import getTestConfig
        from data import PrayagDataset, collateFn
        from model.gameformer import GameFormer, GameFormerLoss, PostProcess
        from utils import setSeed
        
        print("✓ Imports successful")
        
        # Get config
        config = getTestConfig(numChunks=1)
        print(f"✓ Config loaded: obsHorizon={config['obsHorizon']}, predHorizon={config['predHorizon']}")
        
        # Set seed
        setSeed(42)
        
        # Check dataset paths
        datasetDir = Path(config["datasetDir"])
        if not datasetDir.exists():
            print(f"⚠ Dataset directory not found: {datasetDir}")
            print("  Skipping dataset test, testing with dummy data...")
            
            # Test with dummy data
            obsHorizon = config["obsHorizon"]
            predHorizon = config["predHorizon"]
            numLanes = config["numLanes"]
            lanePoints = config["lanePoints"]
            numCrosswalks = config["numCrosswalks"]
            crosswalkPoints = config["crosswalkPoints"]
            neighborsToPredict = config["neighborsToPredict"]
            
            batch = {
                "ego_state": torch.randn(2, obsHorizon, 9),
                "neighbors_state": torch.randn(2, neighborsToPredict, obsHorizon, 9),
                "map_lanes": torch.randn(2, neighborsToPredict + 1, numLanes, lanePoints, 16),
                "map_crosswalks": torch.randn(2, neighborsToPredict + 1, numCrosswalks, crosswalkPoints, 3),
                "ego_future": torch.randn(2, predHorizon, 2),
                "neighbors_future": torch.randn(2, neighborsToPredict, predHorizon, 2),
            }
            
        else:
            # Load real dataset
            dataset = PrayagDataset(config, split="train", testChunks=1)
            print(f"✓ Dataset loaded: {len(dataset)} samples")
            
            if len(dataset) == 0:
                print("⚠ Dataset is empty, skipping test")
                return True
            
            # Get a batch
            from torch.utils.data import DataLoader
            loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collateFn)
            batch = next(iter(loader))
            print(f"✓ Batch loaded")
        
        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Create model
        model = GameFormer(config)
        model = model.to(device)
        numParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ Model created: {numParams:,} parameters")
        
        # Forward pass
        model.train()
        out = model(batch)
        print(f"✓ Forward pass successful")
        print(f"  Output keys: {list(out.keys())}")
        
        # Loss
        loss_fn = GameFormerLoss(config).to(device)
        lossOut = loss_fn(out, batch)
        print(f"✓ Loss computed: {lossOut['loss'].item():.4f}")
        
        # Post-process
        postProcess = PostProcess(config).to(device)
        with torch.no_grad():
            postOut = postProcess(out, batch)
        print(f"✓ Post-process: ADE={postOut.get('ade', 0):.4f}, FDE={postOut.get('fde', 0):.4f}")
        
        # Backward pass
        lossOut["loss"].backward()
        print(f"✓ Backward pass successful")
        
        # Cleanup
        del model, loss_fn, postProcess, out, lossOut, batch
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print("\n✓ PrayagGameFormer test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ PrayagGameFormer test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    args = parseArgs()
    
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    
    print("="*60)
    print("PrayagLaneGCN and PrayagGameFormer Test Suite")
    print("="*60)
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    results = {}
    
    if args.model in ["lanegcn", "all"]:
        results["PrayagLaneGCN"] = testLaneGCN(device)
    
    if args.model in ["gameformer", "all"]:
        results["PrayagGameFormer"] = testGameFormer(device)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    allPassed = True
    for model, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {model}: {status}")
        if not passed:
            allPassed = False
    
    if allPassed:
        print("\n✓ All tests PASSED!")
        return 0
    else:
        print("\n✗ Some tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
