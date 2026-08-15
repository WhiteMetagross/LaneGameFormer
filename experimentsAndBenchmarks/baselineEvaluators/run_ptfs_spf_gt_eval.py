"""
Evaluation orchestrator for PTFS SPF GT (formerly ProjectPrayagV3)
Runs all four baseline variants (A, B, C, D) on 10Hz and 30Hz test splits in metric units.
Saves results to JSON.
"""
import os
import sys
import json
import glob
import numpy as np

# Ensure ProjectPrayagV3 is in python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ProjectPrayagV3"))

import config
from evaluate_system import evaluate_scenario

# Set configuration to use Cartesian meters
config.USE_METRES = True
config.VERBOSE_OUTPUT = False
config.SAVE_INTERMEDIATE_RESULTS = False

datasets = [
    {
        "name": "10Hz",
        "dir": "ChunkedProjectPrayagBEVDataset10Hz",
        "fps": 10.0
    },
    {
        "name": "30Hz",
        "dir": "ChunkedProjectPrayagBEVDataset",
        "fps": 30.0
    }
]

modes = ["A", "B", "C", "D"]
results = {}

for ds in datasets:
    dataset_name = ds["name"]
    dataset_dir = ds["dir"]
    fps = ds["fps"]
    
    results[dataset_name] = {}
    
    base_dir = os.path.join(dataset_dir, "test", "annotations")
    track_files = glob.glob(os.path.join(base_dir, "*_tracks.csv"))
    
    print(f"\n============================================================")
    print(f"Evaluating {dataset_name} Dataset ({len(track_files)} files)")
    print(f"============================================================")
    
    for mode in modes:
        # Override baseline mode in config
        config.BASELINE_MODE = mode
        
        print(f"\nRunning Mode {mode}...")
        
        all_metrics = {
            'minADE@1': [],
            'minADE@4': [],
            'minFDE@1': [],
            'minFDE@4': [],
            'miss_rate_10': [],
            'miss_rate_20': [],
            'norm_fde': [],
            'apd': [],
            'nll': [],
            'collision': [],
            'off_road': []
        }
        
        for track_path in track_files:
            road_mask_path = track_path.replace("_tracks.csv", "_road_annotation.json")
            if not os.path.exists(road_mask_path):
                continue
                
            metrics = evaluate_scenario(track_path, road_mask_path, fps=fps)
            
            for k, v in metrics.items():
                if k in all_metrics:
                    all_metrics[k].extend(v)
        
        # Calculate means
        mode_results = {}
        if all_metrics['minADE@1']:
            count = len(all_metrics['minADE@1'])
            mode_results["samples"] = count
            for k, v in all_metrics.items():
                mode_results[k] = float(np.mean(v))
            
            print(f"Mode {mode} Completed: {count} samples")
            print(f"  minADE@1: {mode_results['minADE@1']:.4f} m")
            print(f"  minADE@4: {mode_results['minADE@4']:.4f} m")
            print(f"  Collision Rate: {mode_results['collision'] * 100:.2f}%")
            print(f"  Off-Road Rate:  {mode_results['off_road'] * 100:.2f}%")
        else:
            print(f"Mode {mode} has no valid samples.")
            mode_results["samples"] = 0
            
        results[dataset_name][mode] = mode_results

# Save results JSON to both workspaces
output_dirs = [
    "experiment_outputs/eval",
    "paper/code/experiment_outputs/eval"
]

for out_dir in output_dirs:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "PTFS_SPF_GT_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_path}")

print("\nEvaluation Orchestration Completed successfully")
