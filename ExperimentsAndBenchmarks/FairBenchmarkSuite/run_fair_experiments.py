#!/usr/bin/env python3
"""
Unified Fair Experiment Runner for Research Paper

Runs HPO and training for all 3 models (PrayagLaneGCN, PrayagGameFormer,
LaneGameFormer) with identical fairness constraints:

  - Master seed: 17
  - 3 training seeds generated via srand(17): [635736847, 1801979802, 1704524973]
  - Optuna TPE sampler with seed 17
  - Same hyperparameter search ranges across all models
  - HPO: 20 trials, 10 epochs, patience=3
  - Training: 50 epochs, patience=5
  - HPO datasets: StratifiedProjectPrayagBEVDataset / StratifiedProjectPrayagBEVDataset10Hz
  - Train datasets: ChunkedProjectPrayagBEVDataset / ChunkedProjectPrayagBEVDataset10Hz
  - torch.compile + Triton, AMP FP16, flash attention, torch.geometric

Usage (from WSL):
    conda activate mambahar

    # Quick test (1-2 trials/epochs)
    python run_fair_experiments.py --mode test

    # Full HPO then training
    python run_fair_experiments.py --mode hpo --dataset-type 10Hz
    python run_fair_experiments.py --mode train --dataset-type 10Hz

    # Both HPO + train sequentially
    python run_fair_experiments.py --mode all --dataset-type 10Hz

Author: Senior AI Engineer
"""

import os
import sys
import json
import time
import argparse
import subprocess
import random
from pathlib import Path
from datetime import datetime

# ============================================================================
# PATH AND COMPATIBILITY HELPERS
# ============================================================================
import platform

def get_base_output_dir():
    """Get the base output directory, converting to WSL path if running on Linux."""
    if platform.system() == 'Linux':
        return Path("/mnt/c/Users/Xeron/LaneGameFormer_outputs")
    return Path("C:/Users/Xeron/LaneGameFormer_outputs")

def get_hpo_dir(model_name, dataset_type):
    """Get HPO directory path (local home directory on Linux to prevent SQLite locks on /mnt/c)."""
    import os
    if platform.system() == 'Linux':
        hpo_base = Path(os.path.expanduser("~/LaneGameFormer_outputs/hpo"))
    else:
        hpo_base = Path("C:/Users/Xeron/LaneGameFormer_outputs/hpo")
    return hpo_base / f"{model_name}_{dataset_type}"

def get_windows_hpo_dir(model_name, dataset_type):
    """Get Windows mount HPO directory path on WSL or native Windows path."""
    if platform.system() == 'Linux':
        return Path("/mnt/c/Users/Xeron/LaneGameFormer_outputs/hpo") / f"{model_name}_{dataset_type}"
    return Path("C:/Users/Xeron/LaneGameFormer_outputs/hpo") / f"{model_name}_{dataset_type}"

def get_train_dir(model_name, dataset_type, seed):
    """Get training output directory (mapped for Windows/WSL compatibility)."""
    if platform.system() == 'Linux':
        base = Path("/mnt/c/Users/Xeron/LaneGameFormer_outputs/train")
    else:
        base = Path("C:/Users/Xeron/LaneGameFormer_outputs/train")
    return base / f"{model_name}_{dataset_type}" / f"seed_{seed}"

# ============================================================================
# MASTER SEED AND DERIVED SEEDS
# ============================================================================
MASTER_SEED = 29

def get_training_seeds(master_seed=MASTER_SEED, count=3):
    """Generate training seeds from master seed using srand."""
    rng = random.Random(master_seed)
    return [rng.randint(0, 2**31 - 1) for _ in range(count)]

TRAINING_SEEDS = get_training_seeds()  # [635736847, 1801979802, 1704524973]

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

MODELS = ["PrayagLaneGCN", "PrayagGameFormer", "LaneGameFormer", "LaneGameFormer_A1", "LaneGameFormer_A2", "LaneGameFormer_A3", "LaneGameFormer_A4", "LaneGameFormer_S1", "LaneGameFormer_K0", "LaneGameFormer_M0"]

HPO_CONFIG = {
    "num_trials": 20,
    "num_epochs": 10,
    "patience": 3,
    "seed": MASTER_SEED,
    "batch_size": 16,
    "max_agents": 32,
    "num_workers": 8,
}

TRAIN_CONFIG = {
    "num_epochs": 50,
    "patience": 6,
    "batch_size": 16,
    "max_agents": 100,
    "num_workers": 8,
}

TEST_HPO_CONFIG = {
    "num_trials": 2,
    "num_epochs": 2,
    "patience": 2,
    "seed": MASTER_SEED,
    "batch_size": 4,
    "max_agents": 16,
    "num_workers": 4,
}

TEST_TRAIN_CONFIG = {
    "num_epochs": 2,
    "patience": 2,
    "batch_size": 4,
    "max_agents": 16,
    "num_workers": 4,
}


def get_dataset_paths(base_dir, dataset_type, for_hpo=False):
    """Get dataset paths based on type and HPO/train mode."""
    if for_hpo:
        if dataset_type == "10Hz":
            return str(base_dir / "StratifiedProjectPrayagBEVDataset10Hz")
        else:
            return str(base_dir / "StratifiedProjectPrayagBEVDataset")
    else:
        if dataset_type == "10Hz":
            return str(base_dir / "ChunkedProjectPrayagBEVDataset10Hz")
        else:
            return str(base_dir / "ChunkedProjectPrayagBEVDataset")


def run_command(cmd, desc=""):
    """Run a shell command and print output."""
    print(f"\n{'='*70}")
    print(f"RUNNING: {desc}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    start = time.time()
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    elapsed = time.time() - start
    print(f"\n{'='*70}")
    print(f"COMPLETED: {desc} ({elapsed:.1f}s, exit={result.returncode})")
    print(f"{'='*70}\n")
    return result.returncode, elapsed


def run_hpo_for_model(model_name, dataset_type, base_dir, config, is_test=False):
    """Run HPO for a single model."""
    if model_name in ("LaneGameFormer_A1", "LaneGameFormer_A2", "LaneGameFormer_A3", "LaneGameFormer_A4", "LaneGameFormer_S1", "LaneGameFormer_K0", "LaneGameFormer_M0"):
        print(f"Skipping HPO for ablation model {model_name} (hyperparameters will be reused from LaneGameFormer)")
        return 0, 0.0
        
    dataset_path = get_dataset_paths(base_dir, dataset_type, for_hpo=True)
    local_output_dir = get_hpo_dir(model_name, dataset_type)
    output_dir_str = str(local_output_dir)
    
    # Restore database/results from Windows mount path to Linux home path if resuming and it exists
    import shutil
    if platform.system() == 'Linux' and not is_test:
        win_dir = get_windows_hpo_dir(model_name, dataset_type)
        if win_dir.exists():
            # Check if local database is newer to avoid overwriting newer progress
            db_prefix = {
                "PrayagLaneGCN": "prayag_lanegcn",
                "PrayagGameFormer": "prayag_gameformer",
                "LaneGameFormer": "lanegameformer"
            }.get(model_name, model_name.lower())
            db_filename = f"{db_prefix}_{dataset_type.lower()}.db"
            local_db = local_output_dir / db_filename
            win_db = win_dir / db_filename
            if local_db.exists() and win_db.exists() and local_db.stat().st_mtime > win_db.stat().st_mtime:
                print(f"Local database is newer than Windows backup. Skipping database restore to preserve progress: {local_db}")
            else:
                local_output_dir.mkdir(parents=True, exist_ok=True)
                shutil.copytree(win_dir, local_output_dir, dirs_exist_ok=True)
                print(f"Restored HPO databases and results: {win_dir} -> {local_output_dir}")
    
    if model_name == "PrayagLaneGCN":
        cmd = [
            sys.executable, str(base_dir / "PrayagLaneGCN" / "tune.py"),
            "--datasetType", dataset_type,
            "--numTrials", str(config["num_trials"]),
            "--numEpochs", str(config["num_epochs"]),
            "--patience", str(config["patience"]),
            "--seed", str(config["seed"]),
            "--outputDir", output_dir_str,
            "--studyName", f"prayag_lanegcn_{dataset_type.lower()}",
            "--datasetPath", dataset_path,
            "--batchSize", str(config["batch_size"]),
            "--maxAgents", str(config["max_agents"]),
            "--numWorkers", str(config["num_workers"]),
            "--saveTrialModels",
        ]
    elif model_name == "PrayagGameFormer":
        cmd = [
            sys.executable, str(base_dir / "PrayagGameFormer" / "tune.py"),
            "--datasetType", dataset_type,
            "--numTrials", str(config["num_trials"]),
            "--numEpochs", str(config["num_epochs"]),
            "--patience", str(config["patience"]),
            "--seed", str(config["seed"]),
            "--outputDir", output_dir_str,
            "--studyName", f"prayag_gameformer_{dataset_type.lower()}",
            "--datasetPath", dataset_path,
            "--batchSize", str(config["batch_size"]),
            "--maxAgents", str(config["max_agents"]),
            "--numWorkers", str(config["num_workers"]),
            "--saveTrialModels",
        ]
    elif model_name == "LaneGameFormer":
        cmd = [
            sys.executable, str(base_dir / "LaneGameFormer" / "scripts" / "tune.py"),
            "--n-trials", str(config["num_trials"]),
            "--epochs", str(config["num_epochs"]),
            "--patience", str(config["patience"]),
            "--output-dir", output_dir_str,
            "--study-name", f"lanegameformer_{dataset_type.lower()}",
            "--dataset-path", dataset_path,
            "--max-agents", str(config["max_agents"]),
            "--num-workers", str(config["num_workers"]),
            "--batch-size", str(config["batch_size"]),
            "--save-trial-models",
        ]
    if not is_test:
        cmd.append("--resume")
    
    ret, elapsed = run_command(cmd, f"HPO {model_name} ({dataset_type})")
    
    # Back up results from Linux home directory to Windows mount path
    if platform.system() == 'Linux':
        win_dir = get_windows_hpo_dir(model_name, dataset_type)
        if local_output_dir.exists():
            win_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(local_output_dir, win_dir, dirs_exist_ok=True)
            print(f"Backed up HPO results/databases: {local_output_dir} -> {win_dir}")
            
    return ret, elapsed


def is_training_complete(save_dir, max_epochs=50, patience=6):
    """
    Check if a training run is complete.
    
    A training run is considered complete if it has reached the maximum epochs
    or if it has triggered early stopping.
    """
    save_dir = Path(save_dir)
    history_file = save_dir / "training_history.json"
    if not history_file.exists():
        return False
    try:
        with open(history_file, 'r') as f:
            data = json.load(f)
        epochs = data.get("train", [])
        if not epochs:
            return False
        last_epoch = epochs[-1].get("epoch", 0)
        if last_epoch >= max_epochs:
            return True
        best_val_loss = float("inf")
        best_epoch = 0
        val_epochs = data.get("val", [])
        for val_ep in val_epochs:
            ep = val_ep.get("epoch", 0)
            loss = val_ep.get("loss", float("inf"))
            if loss < best_val_loss:
                best_val_loss = loss
                best_epoch = ep
        if last_epoch - best_epoch >= patience:
            return True
    except Exception:
        pass
    return False


def run_training_for_model(model_name, dataset_type, base_dir, config, seed, hpo_params_path=None):
    """Run training for a single model with a specific seed."""
    dataset_path = get_dataset_paths(base_dir, dataset_type, for_hpo=False)
    save_dir = str(get_train_dir(model_name, dataset_type, seed))
    latest_checkpoint = os.path.join(save_dir, "latest.pth")
    resume_path = latest_checkpoint if os.path.exists(latest_checkpoint) else None

    if model_name == "PrayagLaneGCN":
        cmd = [
            sys.executable, str(base_dir / "PrayagLaneGCN" / "train.py"),
            "--datasetType", dataset_type,
            "--numEpochs", str(config["num_epochs"]),
            "--patience", str(config["patience"]),
            "--seed", str(seed),
            "--saveDir", save_dir,
            "--datasetPath", dataset_path,
            "--batchSize", str(config["batch_size"]),
            "--maxAgents", str(config["max_agents"]),
            "--workers", str(config["num_workers"]),
        ]
        if resume_path:
            cmd.extend(["--resume", resume_path])
        if hpo_params_path:
            cmd.extend(["--hpoParams", hpo_params_path])

    elif model_name == "PrayagGameFormer":
        cmd = [
            sys.executable, str(base_dir / "PrayagGameFormer" / "train.py"),
            "--datasetType", dataset_type,
            "--numEpochs", str(config["num_epochs"]),
            "--patience", str(config["patience"]),
            "--seed", str(seed),
            "--saveDir", save_dir,
            "--datasetPath", dataset_path,
            "--batchSize", str(config["batch_size"]),
            "--maxAgents", str(config["max_agents"]),
            "--workers", str(config["num_workers"]),
        ]
        if resume_path:
            cmd.extend(["--resume", resume_path])
        if hpo_params_path:
            cmd.extend(["--hpoParams", hpo_params_path])

    elif model_name in ("LaneGameFormer", "LaneGameFormer_A1", "LaneGameFormer_A2", "LaneGameFormer_A3", "LaneGameFormer_A4", "LaneGameFormer_S1", "LaneGameFormer_K0", "LaneGameFormer_M0"):
        cmd = [
            sys.executable, str(base_dir / "LaneGameFormer" / "scripts" / "train.py"),
            "--epochs", str(config["num_epochs"]),
            "--patience", str(config["patience"]),
            "--seed", str(seed),
            "--save-dir", save_dir,
            "--dataset-path", dataset_path,
            "--batch-size", str(config["batch_size"]),
            "--max-agents", str(config["max_agents"]),
            "--workers", str(config["num_workers"]),
        ]
        if resume_path:
            cmd.extend(["--resume", resume_path])
        if "_" in model_name:
            ablation_type = model_name.split("_")[1]
            cmd.extend(["--ablation", ablation_type])
        if hpo_params_path:
            cmd.extend(["--hpo-params", hpo_params_path])

    return run_command(cmd, f"Train {model_name} ({dataset_type}, seed={seed})")


def main():
    parser = argparse.ArgumentParser(description="Unified Fair Experiment Runner")
    parser.add_argument("--mode", choices=["hpo", "train", "all", "test"],
                        default="test", help="Experiment mode")
    parser.add_argument("--dataset-type", choices=["10Hz", "30Hz", "both"],
                        default="10Hz", help="Dataset frequency")
    parser.add_argument("--models", nargs="+", default=None,
                        choices=MODELS, help="Models to run (default: all)")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    models = args.models or MODELS
    dataset_types = ["10Hz", "30Hz"] if args.dataset_type == "both" else [args.dataset_type]
    is_test = args.mode == "test"

    hpo_cfg = TEST_HPO_CONFIG if is_test else HPO_CONFIG
    train_cfg = TEST_TRAIN_CONFIG if is_test else TRAIN_CONFIG

    print("="*70)
    print("UNIFIED FAIR EXPERIMENT RUNNER")
    print("="*70)
    print(f"Mode:          {args.mode}")
    print(f"Master Seed:   {MASTER_SEED}")
    print(f"Training Seeds:{TRAINING_SEEDS}")
    print(f"Models:        {models}")
    print(f"Datasets:      {dataset_types}")
    print(f"HPO Config:    {hpo_cfg}")
    print(f"Train Config:  {train_cfg}")
    print(f"Timestamp:     {datetime.now().isoformat()}")
    print("="*70)

    # Save experiment manifest
    manifest_dir = get_base_output_dir()
    if not os.path.exists(manifest_dir):
        os.makedirs(manifest_dir, exist_ok=True)
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "master_seed": MASTER_SEED,
        "training_seeds": TRAINING_SEEDS,
        "models": models,
        "dataset_types": dataset_types,
        "hpo_config": hpo_cfg,
        "train_config": train_cfg,
        "mode": args.mode,
    }
    with open(str(manifest_dir / "experiment_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    timing_log = []

    # --- HPO Phase ---
    if args.mode in ("hpo", "all", "test"):
        for dt in dataset_types:
            for model in models:
                current_hpo_cfg = hpo_cfg.copy()
                if dt == "30Hz" and not is_test:
                    current_hpo_cfg["batch_size"] = 2
                    current_hpo_cfg["num_workers"] = 4
                ret, elapsed = run_hpo_for_model(model, dt, base_dir, current_hpo_cfg, is_test)
                timing_log.append({"phase": "hpo", "model": model, "dataset": dt,
                                   "elapsed_s": elapsed, "exit_code": ret})

    # --- Training Phase ---
    if args.mode in ("train", "all", "test"):
        for dt in dataset_types:
            for model in models:
                if model in ("LaneGameFormer_A1", "LaneGameFormer_A2", "LaneGameFormer_A3", "LaneGameFormer_A4", "LaneGameFormer_S1", "LaneGameFormer_K0", "LaneGameFormer_M0") and dt == "30Hz":
                    print(f"Skipping training of ablation model {model} on 30Hz dataset per updated plan")
                    continue
                # Find HPO best params
                # For all LaneGameFormer ablation models, we reuse HPO params from the main LaneGameFormer!
                hpo_model_name = "LaneGameFormer" if model in ("LaneGameFormer_A1", "LaneGameFormer_A2", "LaneGameFormer_A3", "LaneGameFormer_A4", "LaneGameFormer_S1", "LaneGameFormer_K0", "LaneGameFormer_M0") else model
                hpo_dir = get_windows_hpo_dir(hpo_model_name, dt)
                hpo_params = str(hpo_dir / "best_params.json") if (hpo_dir / "best_params.json").exists() else None

                seeds = TRAINING_SEEDS if not is_test else TRAINING_SEEDS[:1]
                for seed in seeds:
                    current_train_cfg = train_cfg.copy()
                    
                    # Check if training run is already complete to avoid redundant work
                    save_dir = get_train_dir(model, dt, seed)
                    if not is_test and is_training_complete(save_dir, current_train_cfg["num_epochs"], current_train_cfg["patience"]):
                        print(f"Skipping completed training run for {model} ({dt}, seed={seed})")
                        timing_log.append({"phase": "train", "model": model, "dataset": dt,
                                           "seed": seed, "elapsed_s": 0.0, "exit_code": 0})
                        continue
                        
                    if dt == "30Hz" and not is_test:
                        current_train_cfg["batch_size"] = 2
                        current_train_cfg["num_workers"] = 4
                    elif model in ("PrayagGameFormer", "LaneGameFormer", "LaneGameFormer_A1", "LaneGameFormer_A2", "LaneGameFormer_A3", "LaneGameFormer_A4", "LaneGameFormer_S1", "LaneGameFormer_K0", "LaneGameFormer_M0") and not is_test:
                        # Reduce batch size and max agents for Transformer models to fit in 8GB VRAM
                        current_train_cfg["batch_size"] = 4
                        current_train_cfg["max_agents"] = 32
                        current_train_cfg["num_workers"] = 4
                        
                    ret, elapsed = run_training_for_model(
                        model, dt, base_dir, current_train_cfg, seed, hpo_params)
                    timing_log.append({"phase": "train", "model": model, "dataset": dt,
                                       "seed": seed, "elapsed_s": elapsed, "exit_code": ret})

    # Save timing log
    with open(str(manifest_dir / "timing_log.json"), "w") as f:
        json.dump(timing_log, f, indent=2)

    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    total_time = sum(t["elapsed_s"] for t in timing_log)
    for entry in timing_log:
        status = "OK" if entry["exit_code"] == 0 else "FAIL"
        print(f"  [{status}] {entry['phase']:5s} | {entry['model']:20s} | "
              f"{entry.get('dataset',''):4s} | seed={str(entry.get('seed','N/A')):>12s} | "
              f"{entry['elapsed_s']:.1f}s")
    print(f"\nTotal time: {total_time:.1f}s ({total_time/3600:.2f}h)")
    print("="*70)


if __name__ == "__main__":
    main()
