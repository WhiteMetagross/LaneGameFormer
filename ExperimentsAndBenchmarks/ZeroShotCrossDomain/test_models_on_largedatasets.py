"""
Test Script for HPO and Training on LargeDatasets

This script runs quick HPO and training tests on all three models:
- PrayagGameFormer
- PrayagLaneGCN  
- LaneGameFormer

Memory optimized for RTX 4060 8GB VRAM:
- Reduced batch size (4-8)
- Reduced max_agents (16-24)
- Limited chunks (1-2)
- Half resolution flow surfaces

Usage:
    python test_models_on_largedatasets.py --test-hpo      # Quick HPO test
    python test_models_on_largedatasets.py --test-train    # Quick training test
    python test_models_on_largedatasets.py --all           # Run both

Author: LaneGameFormer Research Team
Date: January 2026
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

# Dataset paths
HPO_DATASETS = {
    "10Hz": r"C:\Users\Xeron\OneDrive\Documents\LargeDatasets\LargeDatasets\StratifiedProjectPrayagBEVDataset10Hz",
    "30Hz": r"C:\Users\Xeron\OneDrive\Documents\LargeDatasets\LargeDatasets\StratifiedProjectPrayagBEVDataset"
}

TRAIN_DATASETS = {
    "10Hz": r"C:\Users\Xeron\OneDrive\Documents\LargeDatasets\LargeDatasets\StratifiedProjectPrayagBEVDataset10Hz",
    "30Hz": r"C:\Users\Xeron\OneDrive\Documents\LargeDatasets\LargeDatasets\StratifiedProjectPrayagBEVDataset"
}

# Base directory
BASE_DIR = Path(__file__).parent

# Memory settings for RTX 4060 8GB
MEMORY_SETTINGS = {
    "batchSize": 4,      # Reduced from 16
    "maxAgents": 16,     # Reduced from 32
    "numTrials": 3,      # Quick HPO test
    "numEpochs": 3,      # Quick training
    "patience": 2,       # Early stopping
}


def run_command(cmd: list, name: str, cwd: Path = None) -> bool:
    """Run a command and return success status."""
    print("\n" + "=" * 70)
    print(f"Running: {name}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 70 + "\n")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            check=False,
            capture_output=False  # Show output in real-time
        )
        success = result.returncode == 0
        if success:
            print(f"\n✓ {name} completed successfully")
        else:
            print(f"\n✗ {name} failed with return code {result.returncode}")
        return success
    except Exception as e:
        print(f"\n✗ {name} failed with exception: {e}")
        return False


def test_prayag_gameformer_hpo(dataset_type: str = "10Hz") -> bool:
    """Test PrayagGameFormer HPO."""
    dataset_path = HPO_DATASETS[dataset_type]
    
    cmd = [
        sys.executable,
        "tune.py",
        "--datasetType", dataset_type,
        "--datasetPath", dataset_path,
        "--numTrials", str(MEMORY_SETTINGS["numTrials"]),
        "--numEpochs", str(MEMORY_SETTINGS["numEpochs"]),
        "--patience", str(MEMORY_SETTINGS["patience"]),
        "--batchSize", str(MEMORY_SETTINGS["batchSize"]),
        "--maxAgents", str(MEMORY_SETTINGS["maxAgents"]),
        "--studyName", f"prayag_gameformer_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "--outputDir", str(BASE_DIR / "outputs" / "hpo_test" / "prayag_gameformer"),
    ]
    
    return run_command(cmd, "PrayagGameFormer HPO Test", cwd=BASE_DIR / "PrayagGameFormer")


def test_prayag_lanegcn_hpo(dataset_type: str = "10Hz") -> bool:
    """Test PrayagLaneGCN HPO."""
    dataset_path = HPO_DATASETS[dataset_type]
    
    cmd = [
        sys.executable,
        "tune.py",
        "--datasetType", dataset_type,
        "--datasetPath", dataset_path,
        "--numTrials", str(MEMORY_SETTINGS["numTrials"]),
        "--numEpochs", str(MEMORY_SETTINGS["numEpochs"]),
        "--patience", str(MEMORY_SETTINGS["patience"]),
        "--batchSize", str(MEMORY_SETTINGS["batchSize"]),
        "--maxAgents", str(MEMORY_SETTINGS["maxAgents"]),
        "--studyName", f"prayag_lanegcn_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "--outputDir", str(BASE_DIR / "outputs" / "hpo_test" / "prayag_lanegcn"),
    ]
    
    return run_command(cmd, "PrayagLaneGCN HPO Test", cwd=BASE_DIR / "PrayagLaneGCN")


def test_lanegameformer_hpo(dataset_type: str = "10Hz") -> bool:
    """Test LaneGameFormer HPO."""
    # Use the config file with LargeDataset path
    config_path = BASE_DIR / "LaneGameFormer" / "configs" / "config_hpo_largedataset.yaml"
    
    cmd = [
        sys.executable,
        "scripts/tune.py",
        "--config", str(config_path),
        "--n-trials", str(MEMORY_SETTINGS["numTrials"]),
        "--epochs", str(MEMORY_SETTINGS["numEpochs"]),
        "--patience", str(MEMORY_SETTINGS["patience"]),
        "--max-chunks", "2",
        "--study-name", f"lanegameformer_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "--output-dir", str(BASE_DIR / "outputs" / "hpo_test" / "lanegameformer"),
    ]
    
    return run_command(cmd, "LaneGameFormer HPO Test", cwd=BASE_DIR / "LaneGameFormer")


def test_prayag_gameformer_train(dataset_type: str = "10Hz") -> bool:
    """Test PrayagGameFormer training."""
    dataset_path = TRAIN_DATASETS[dataset_type]
    
    cmd = [
        sys.executable,
        "train.py",
        "--datasetType", dataset_type,
        "--batchSize", str(MEMORY_SETTINGS["batchSize"]),
        "--numEpochs", str(MEMORY_SETTINGS["numEpochs"]),
        "--patience", str(MEMORY_SETTINGS["patience"]),
        "--testChunks", "1",
        "--saveDir", str(BASE_DIR / "outputs" / "train_test" / "prayag_gameformer"),
    ]
    
    return run_command(cmd, "PrayagGameFormer Training Test", cwd=BASE_DIR / "PrayagGameFormer")


def test_prayag_lanegcn_train(dataset_type: str = "10Hz") -> bool:
    """Test PrayagLaneGCN training."""
    dataset_path = TRAIN_DATASETS[dataset_type]
    
    cmd = [
        sys.executable,
        "train.py",
        "--datasetType", dataset_type,
        "--batchSize", str(MEMORY_SETTINGS["batchSize"]),
        "--numEpochs", str(MEMORY_SETTINGS["numEpochs"]),
        "--patience", str(MEMORY_SETTINGS["patience"]),
        "--testChunks", "1",
        "--saveDir", str(BASE_DIR / "outputs" / "train_test" / "prayag_lanegcn"),
    ]
    
    return run_command(cmd, "PrayagLaneGCN Training Test", cwd=BASE_DIR / "PrayagLaneGCN")


def test_lanegameformer_train(dataset_type: str = "10Hz") -> bool:
    """Test LaneGameFormer training."""
    # Use the config file with LargeDataset path
    config_path = BASE_DIR / "LaneGameFormer" / "configs" / "config_hpo_largedataset.yaml"
    
    cmd = [
        sys.executable,
        "scripts/train.py",
        "--config", str(config_path),
        "--epochs", str(MEMORY_SETTINGS["numEpochs"]),
        "--patience", str(MEMORY_SETTINGS["patience"]),
        "--test-chunks", "1",
        "--save-dir", str(BASE_DIR / "outputs" / "train_test" / "lanegameformer"),
    ]
    
    return run_command(cmd, "LaneGameFormer Training Test", cwd=BASE_DIR / "LaneGameFormer")


def main():
    parser = argparse.ArgumentParser(
        description="Test HPO and Training on LargeDatasets"
    )
    parser.add_argument(
        "--test-hpo", action="store_true",
        help="Run HPO tests on all models"
    )
    parser.add_argument(
        "--test-train", action="store_true",
        help="Run training tests on all models"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run both HPO and training tests"
    )
    parser.add_argument(
        "--dataset-type", type=str, default="10Hz",
        choices=["10Hz", "30Hz"],
        help="Dataset type to use"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        choices=["gameformer", "lanegcn", "lanegameformer"],
        help="Specific model to test (default: all)"
    )
    
    args = parser.parse_args()
    
    if not (args.test_hpo or args.test_train or args.all):
        parser.print_help()
        print("\nPlease specify --test-hpo, --test-train, or --all")
        return 1
    
    print("=" * 70)
    print("LargeDatasets Test Suite")
    print("=" * 70)
    print(f"Dataset Type: {args.dataset_type}")
    print(f"HPO Dataset: {HPO_DATASETS[args.dataset_type]}")
    print(f"Train Dataset: {TRAIN_DATASETS[args.dataset_type]}")
    print(f"Memory Settings: {MEMORY_SETTINGS}")
    print("=" * 70)
    
    results = {}
    
    # HPO Tests
    if args.test_hpo or args.all:
        print("\n" + "=" * 70)
        print("RUNNING HPO TESTS")
        print("=" * 70)
        
        if args.model is None or args.model == "gameformer":
            results["PrayagGameFormer HPO"] = test_prayag_gameformer_hpo(args.dataset_type)
        
        if args.model is None or args.model == "lanegcn":
            results["PrayagLaneGCN HPO"] = test_prayag_lanegcn_hpo(args.dataset_type)
        
        if args.model is None or args.model == "lanegameformer":
            results["LaneGameFormer HPO"] = test_lanegameformer_hpo(args.dataset_type)
    
    # Training Tests
    if args.test_train or args.all:
        print("\n" + "=" * 70)
        print("RUNNING TRAINING TESTS")
        print("=" * 70)
        
        if args.model is None or args.model == "gameformer":
            results["PrayagGameFormer Train"] = test_prayag_gameformer_train(args.dataset_type)
        
        if args.model is None or args.model == "lanegcn":
            results["PrayagLaneGCN Train"] = test_prayag_lanegcn_train(args.dataset_type)
        
        if args.model is None or args.model == "lanegameformer":
            results["LaneGameFormer Train"] = test_lanegameformer_train(args.dataset_type)
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST RESULTS SUMMARY")
    print("=" * 70)
    
    for name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {name}: {status}")
    
    all_passed = all(results.values())
    print("=" * 70)
    
    if all_passed:
        print("All tests passed!")
        return 0
    else:
        print("Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
