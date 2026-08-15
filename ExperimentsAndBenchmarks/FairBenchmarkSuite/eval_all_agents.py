"""
All-Agents Evaluation: Compare ALL models when predicting ALL agents per scene.

LGF v6 was originally evaluated on all agents (2230 samples at 10Hz).
This script evaluates PrayagLaneGCN, PrayagGameFormer, and PV3 on all agents too,
for a fair all-agents comparison.

PV3 ("PV4" label in all-agents context) is inherently all-agents: its
evaluate_system.py iterates over every (frame_idx, track_id) pair, predicting
each agent's trajectory in turn. The "ego" in PV3 is whichever agent is the
current prediction target — it already evaluates all agents by design.

Usage:
    python eval_all_agents.py                # 10Hz (default)
    python eval_all_agents.py --freq 30hz    # 30Hz
    python eval_all_agents.py --freq both    # Both frequencies
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

# Reuse metric functions from evaluate_all_models
from evaluate_all_models import (
    compute_min_ade_k, compute_min_fde_k, compute_miss_rate,
    compute_norm_fde, compute_apd, compute_nll, compute_collision_rate,
    compute_all_metrics, gpu, _pad_neighbor_futures
)

SEEDS = [1778837931, 1303193990, 1570340887]
WORKSPACE = Path(__file__).parent

DATASET_PATHS = {
    "10hz": WORKSPACE / "ChunkedProjectPrayagBEVDataset10Hz",
    "30hz": WORKSPACE / "ChunkedProjectPrayagBEVDataset",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_lanegcn_all_agents(checkpoint_path, dataset_path, dataset_type, device, max_agents=100, workers=0):
    sys.path.insert(0, str(WORKSPACE / "PrayagLaneGCN"))
    from config import getConfig
    from model.lanegcn import LaneGCN
    from data.dataset import PrayagDataset, collateFn

    config = getConfig(dataset_type)
    config["datasetPath"] = str(dataset_path)
    config["maxAgents"] = max_agents

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_config = ckpt.get("config", {})
    for k in ["nActor", "nMap", "numMods", "numPreds", "numScales", "predHorizon",
              "obsHorizon", "actor2actorDist", "actor2mapDist", "map2actorDist"]:
        if k in saved_config:
            config[k] = saved_config[k]

    model = LaneGCN(config).to(device)
    model.load_state_dict(ckpt["modelState"])
    model.eval()

    test_ds = PrayagDataset(config, split="test")
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False,
                             num_workers=workers, collate_fn=collateFn, pin_memory=True)

    all_preds, all_gt, all_scores = [], [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="LaneGCN all-agents"):
            out = model(batch)
            B = len(out["reg"])
            for i in range(B):
                reg_i = out["reg"][i].detach().cpu().numpy()  # (N, M, T, 2)
                cls_i = out["cls"][i].detach().cpu().numpy()  # (N, M)
                gt_all_i = batch["gtPreds"][i]
                if torch.is_tensor(gt_all_i):
                    gt_all_i = gt_all_i.numpy()
                else:
                    gt_all_i = np.array(gt_all_i)

                N_agents = reg_i.shape[0]
                for n in range(N_agents):
                    g = gt_all_i[n]  # (T, 2)
                    # Skip zero-padded agents
                    if np.linalg.norm(g) < 1e-6:
                        continue
                    all_preds.append(reg_i[n])
                    all_scores.append(cls_i[n])
                    all_gt.append(g)

    sys.path.remove(str(WORKSPACE / "PrayagLaneGCN"))
    for m in [k for k in sys.modules if k in ('config', 'model', 'data', 'utils')
              or k.startswith(('model.', 'data.', 'config.', 'utils.'))]:
        del sys.modules[m]

    preds = np.array(all_preds)
    gt = np.array(all_gt)
    scores = np.array(all_scores)

    M = preds.shape[1]
    return {
        "Samples": len(preds),
        "minADE@1": compute_min_ade_k(preds, gt, k=1),
        "minADE@4": compute_min_ade_k(preds, gt, k=min(4, M)),
        "minFDE@1": compute_min_fde_k(preds, gt, k=1),
        "minFDE@4": compute_min_fde_k(preds, gt, k=min(4, M)),
        "MR@10px": compute_miss_rate(preds, gt, threshold=10.0),
        "MR@20px": compute_miss_rate(preds, gt, threshold=20.0),
        "APD": compute_apd(preds, k=M),
        "NLL": compute_nll(preds, gt, scores=scores),
        "CR": 0.0,  # Can't compute per-agent CR without restructuring neighbors
        "ORR": 0.0,
    }


def eval_gameformer_all_agents(checkpoint_path, dataset_path, dataset_type, device, max_agents=100, workers=0):
    sys.path.insert(0, str(WORKSPACE / "PrayagGameFormer"))
    from config import getConfig
    from model.gameformer import GameFormer
    from data.dataset import PrayagDataset, collateFn

    config = getConfig(dataset_type)
    config["datasetDir"] = str(dataset_path)
    config["maxAgents"] = max_agents
    config["testAnnotations"] = str(Path(dataset_path) / "test" / "annotations")
    config["testVideos"] = str(Path(dataset_path) / "test" / "videos")
    config["chunkList"]["test"] = str(Path(dataset_path) / "test_chunks.txt")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_config = ckpt.get("config", {})
    for k in ["dim", "heads", "encoderLayers", "decoderLevels", "numModes",
              "neighborsToPredict", "predHorizon", "obsHorizon"]:
        if k in saved_config:
            config[k] = saved_config[k]

    model = GameFormer(config).to(device)
    model.load_state_dict(ckpt["modelState"])
    model.eval()

    test_ds = PrayagDataset(config, split="test")
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False,
                             num_workers=workers, collate_fn=collateFn, pin_memory=True)

    decoder_levels = config.get("decoderLevels", 3)
    all_preds, all_gt, all_scores = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="GameFormer all-agents"):
            batch = gpu(batch, device)
            out = model(batch)
            final_key = f"level_{decoder_levels}_interactions"
            scores_key = f"level_{decoder_levels}_scores"
            interactions = out[final_key]  # (B, N, M, T, 4)
            sc = out[scores_key]           # (B, N, M)
            B = interactions.shape[0]
            N = interactions.shape[1]

            gt_ego = batch["ego_future"].cpu().numpy()         # (B, T, 2)
            neigh_fut = batch.get("neighbors_future", None)
            if neigh_fut is not None:
                neigh_fut = neigh_fut.cpu().numpy()  # (B, N_neigh, T, 2)

            for i in range(B):
                # Agent 0 = ego
                p0 = interactions[i, 0, :, :, :2].detach().cpu().numpy()  # (M, T, 2)
                s0 = sc[i, 0].detach().cpu().numpy()  # (M,)
                g0 = gt_ego[i]  # (T, 2)
                if np.linalg.norm(g0) > 1e-6:
                    all_preds.append(p0)
                    all_scores.append(s0)
                    all_gt.append(g0)

                # Remaining agents (neighbors)
                if neigh_fut is not None and N > 1:
                    for n in range(1, min(N, 1 + neigh_fut.shape[1])):
                        g_n = neigh_fut[i, n - 1]  # (T, 2)
                        if np.linalg.norm(g_n) < 1e-6:
                            continue
                        p_n = interactions[i, n, :, :, :2].detach().cpu().numpy()
                        s_n = sc[i, n].detach().cpu().numpy()
                        all_preds.append(p_n)
                        all_scores.append(s_n)
                        all_gt.append(g_n)

    sys.path.remove(str(WORKSPACE / "PrayagGameFormer"))
    for m in [k for k in sys.modules if k in ('config', 'model', 'data', 'utils')
              or k.startswith(('model.', 'data.', 'config.', 'utils.'))]:
        del sys.modules[m]

    preds = np.array(all_preds)
    gt = np.array(all_gt)
    scores = np.array(all_scores)

    M = preds.shape[1]
    return {
        "Samples": len(preds),
        "minADE@1": compute_min_ade_k(preds, gt, k=1),
        "minADE@4": compute_min_ade_k(preds, gt, k=min(4, M)),
        "minFDE@1": compute_min_fde_k(preds, gt, k=1),
        "minFDE@4": compute_min_fde_k(preds, gt, k=min(4, M)),
        "MR@10px": compute_miss_rate(preds, gt, threshold=10.0),
        "MR@20px": compute_miss_rate(preds, gt, threshold=20.0),
        "APD": compute_apd(preds, k=M),
        "NLL": compute_nll(preds, gt, scores=scores),
        "CR": 0.0,
        "ORR": 0.0,
    }


def eval_pv3_all_agents(dataset_path, fps, split='test'):
    """Evaluate PV3 on all agents (PV3 is inherently all-agents).

    PV3's evaluate_on_dataset() already iterates over every (frame_idx, track_id) pair,
    so the returned sample count IS the all-agents count. No modification needed.
    """
    sys.path.insert(0, str(WORKSPACE / "PrayagProjectv3"))
    import evaluate_system as pv3_eval
    import config as pv3_config

    metrics_agg = pv3_eval.evaluate_on_dataset(
        dataset_dir=str(dataset_path),
        fps=fps,
        split=split,
    )

    sys.path.remove(str(WORKSPACE / "PrayagProjectv3"))
    for m in [k for k in sys.modules
              if k.startswith(('evaluate_system', 'evaluation_metrics', 'road_mask',
                               'trajectory_predict', 'track_data', 'lane_',
                               'social_potential', 'game_theory', 'emerging_lane',
                               'config'))]:
        del sys.modules[m]

    n_samples = len(metrics_agg.get("minADE@1", []))
    if n_samples == 0:
        return None

    def _mean(key):
        v = metrics_agg.get(key, [])
        return float(np.mean(v)) if v else 0.0

    return {
        "Samples": n_samples,
        "minADE@1": _mean("minADE@1"),
        "minADE@4": _mean("minADE@4"),
        "minFDE@1": _mean("minFDE@1"),
        "minFDE@4": _mean("minFDE@4"),
        "MR@10px": _mean("miss_rate_10"),
        "MR@20px": _mean("miss_rate_20"),
        "APD": _mean("apd"),
        "NLL": _mean("nll"),
        "CR": _mean("collision"),
        "ORR": _mean("off_road"),
    }


def run_eval_for_freq(freq_label, dataset_path, dataset_type):
    """Run all-agents evaluation for a given frequency."""
    freq_hz = 10.0 if dataset_type == "10Hz" else 30.0

    print("=" * 70)
    print(f"ALL-AGENTS EVALUATION ({freq_label})")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Dataset: {dataset_path}")
    print()

    results = defaultdict(list)

    # --- PV3 / PV4 (inherently all-agents, deterministic) ---
    print("\n--- PV3 (PV4: All-Agents) ---")
    pv3_r = eval_pv3_all_agents(str(dataset_path), fps=freq_hz)
    if pv3_r:
        results["PV3 (PV4)"].append(pv3_r)
        print(f"  Samples={pv3_r['Samples']}, minADE@1={pv3_r['minADE@1']:.2f}, "
              f"minADE@4={pv3_r['minADE@4']:.2f}, minFDE@4={pv3_r['minFDE@4']:.2f}, "
              f"MR@10={pv3_r['MR@10px']*100:.1f}%, APD={pv3_r['APD']:.2f}, "
              f"CR={pv3_r['CR']*100:.2f}%, ORR={pv3_r['ORR']*100:.2f}%")
    else:
        print("  No valid PV3 samples.")

    # --- PrayagLaneGCN ---
    train_dir = f"experiment_outputs/train/PrayagLaneGCN_{dataset_type}"
    for seed in SEEDS:
        ckpt = WORKSPACE / train_dir / f"seed_{seed}" / "best.pth"
        if not ckpt.exists():
            print(f"  [SKIP] PrayagLaneGCN seed {seed}: no checkpoint at {ckpt}")
            continue
        print(f"\n--- PrayagLaneGCN seed {seed} ---")
        r = eval_lanegcn_all_agents(str(ckpt), str(dataset_path), dataset_type, DEVICE)
        results["PrayagLaneGCN"].append(r)
        print(f"  Samples={r['Samples']}, minADE@1={r['minADE@1']:.2f}, minADE@4={r['minADE@4']:.2f}, "
              f"minFDE@4={r['minFDE@4']:.2f}, MR@10={r['MR@10px']*100:.1f}%, APD={r['APD']:.2f}")

    # --- PrayagGameFormer ---
    train_dir = f"experiment_outputs/train/PrayagGameFormer_{dataset_type}"
    for seed in SEEDS:
        ckpt = WORKSPACE / train_dir / f"seed_{seed}" / "best.pth"
        if not ckpt.exists():
            print(f"  [SKIP] PrayagGameFormer seed {seed}: no checkpoint at {ckpt}")
            continue
        print(f"\n--- PrayagGameFormer seed {seed} ---")
        r = eval_gameformer_all_agents(str(ckpt), str(dataset_path), dataset_type, DEVICE)
        results["PrayagGameFormer"].append(r)
        print(f"  Samples={r['Samples']}, minADE@1={r['minADE@1']:.2f}, minADE@4={r['minADE@4']:.2f}, "
              f"minFDE@4={r['minFDE@4']:.2f}, MR@10={r['MR@10px']*100:.1f}%, APD={r['APD']:.2f}")

    # --- LGF v6 from saved eval files ---
    lgf_results = {}
    # Use path relative to workspace so it works on both Windows and WSL
    docs_dir = WORKSPACE.parent.parent  # .../Documents/Programs/LaneGameFormer -> .../Documents
    if dataset_type == "10Hz":
        ckpt_base = docs_dir / "checkpoints_v6"
    else:
        ckpt_base = docs_dir / "checkpoints_v6_30hz"

    for seed in SEEDS:
        json_path = ckpt_base / f"seed_{seed}" / "eval_test_best.json"
        if json_path.exists():
            with open(json_path) as f:
                lgf_results[seed] = json.load(f)

    # --- Summary ---
    print("\n" + "=" * 70)
    print(f"ALL-AGENTS COMPARISON ({freq_label}, mean ± std)")
    print("=" * 70)

    metrics_keys = ["Samples", "minADE@1", "minADE@4", "minFDE@1", "minFDE@4",
                    "MR@10px", "MR@20px", "APD", "NLL", "CR", "ORR"]

    print(f"{'Model':<22} {'Samples':>8} {'minADE@1':>12} {'minADE@4':>12} "
          f"{'minFDE@1':>12} {'minFDE@4':>12} {'MR@10px':>10} {'MR@20px':>10} "
          f"{'APD':>10} {'NLL':>12} {'CR':>8} {'ORR':>8}")
    print("-" * 150)

    for model_name in ["PV3 (PV4)", "PrayagLaneGCN", "PrayagGameFormer"]:
        if not results[model_name]:
            continue
        vals = {k: np.array([r[k] for r in results[model_name]]) for k in metrics_keys}
        line = f"{model_name:<22} {vals['Samples'].mean():>8.0f}"
        for k in metrics_keys[1:]:
            v = vals[k]
            if k.startswith("MR") or k in ("CR", "ORR"):
                if v.std() > 0:
                    line += f" {v.mean()*100:>7.2f}±{v.std()*100:.2f}%"
                else:
                    line += f" {v.mean()*100:>10.2f}%"
            else:
                if v.std() > 0:
                    line += f" {v.mean():>8.2f}±{v.std():>4.2f}"
                else:
                    line += f" {v.mean():>12.2f}"
        print(line)

    # LGF v6 from saved files
    if lgf_results:
        lgf_list = list(lgf_results.values())
        key_map = {"minADE@1": "minADE_1", "minADE@4": "minADE_4",
                   "minFDE@1": "minFDE_1", "minFDE@4": "minFDE_4",
                   "MR@10px": "miss_rate_10px", "MR@20px": "miss_rate_20px",
                   "APD": "apd", "NLL": "nll", "Samples": "num_samples",
                   "CR": "collision_rate", "ORR": "off_road_rate"}
        line = f"{'LGF v6':<22}"
        for k in metrics_keys:
            mapped = key_map.get(k, k)
            vals_list = []
            for r in lgf_list:
                if mapped in r:
                    vals_list.append(r[mapped])
            if not vals_list:
                line += f" {'N/A':>12}"
                continue
            v = np.array(vals_list)
            if k == "Samples":
                line += f" {v.mean():>8.0f}"
            elif k.startswith("MR") or k in ("CR", "ORR"):
                if v.std() > 0:
                    line += f" {v.mean()*100:>7.2f}±{v.std()*100:.2f}%"
                else:
                    line += f" {v.mean()*100:>10.2f}%"
            else:
                if v.std() > 0:
                    line += f" {v.mean():>8.2f}±{v.std():>4.2f}"
                else:
                    line += f" {v.mean():>12.2f}"
        print(line)

    # Save results JSON
    output = {}
    for model_name, runs in results.items():
        output[model_name] = runs
    if lgf_results:
        output["LGF_v6"] = list(lgf_results.values())
    out_path = WORKSPACE / "experiment_outputs" / "eval" / f"all_agents_comparison_{dataset_type.lower()}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nResults saved to {out_path}")

    return output


def main():
    parser = argparse.ArgumentParser(description='All-Agents Evaluation')
    parser.add_argument('--freq', type=str, default='10hz',
                        choices=['10hz', '30hz', 'both'],
                        help='Frequency to evaluate: 10hz, 30hz, or both')
    args = parser.parse_args()

    freqs = ['10hz', '30hz'] if args.freq == 'both' else [args.freq]

    for freq in freqs:
        dataset_path = DATASET_PATHS[freq]
        dataset_type = "10Hz" if freq == "10hz" else "30Hz"
        freq_label = f"{dataset_type} Dataset"

        if not dataset_path.exists():
            print(f"WARNING: Dataset not found at {dataset_path}. Skipping {freq_label}.")
            continue

        run_eval_for_freq(freq_label, dataset_path, dataset_type)
        print()


if __name__ == "__main__":
    main()
