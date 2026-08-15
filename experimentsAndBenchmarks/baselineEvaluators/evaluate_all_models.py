"""
Unified Evaluation Script for All Trajectory Prediction Models.

Evaluates PrayagLaneGCN, PrayagGameFormer, LaneGameFormer, and optionally
PrayagProjectV3 (classical baseline) on test sets across 3 seeds, computing
comprehensive metrics with mean ± std.

Metrics computed:
  - minADE@1, minADE@4, minFDE@1, minFDE@4
  - Miss Rate @10px, @20px
  - Norm FDE
  - APD (Diversity)
  - NLL (Probabilistic)
  - Collision Rate
  - Off-Road Rate

Usage:
    python evaluate_all_models.py --output-dir ./experiment_outputs/eval
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Enable speed optimizations
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.set_float32_matmul_precision('high')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ============================================================================
# Metrics (self-contained, no external dependency)
# ============================================================================

def compute_min_ade_k(preds, gt, k=1):
    """minADE@k. preds: (N, M, T, 2), gt: (N, T, 2)."""
    preds = preds[:, :k]
    err = np.sqrt(((preds - gt[:, None]) ** 2).sum(-1)).mean(-1)  # (N, k)
    return err.min(1).mean()


def compute_min_fde_k(preds, gt, k=1):
    """minFDE@k. preds: (N, M, T, 2), gt: (N, T, 2)."""
    preds = preds[:, :k]
    err = np.sqrt(((preds[:, :, -1] - gt[:, None, -1]) ** 2).sum(-1))  # (N, k)
    return err.min(1).mean()


def compute_miss_rate(preds, gt, threshold=10.0):
    """Miss rate: fraction where best-mode FDE > threshold."""
    M = preds.shape[1]
    fde = np.sqrt(((preds[:, :, -1] - gt[:, None, -1]) ** 2).sum(-1))  # (N, M)
    return (fde.min(1) > threshold).mean()


def compute_norm_fde(preds, gt):
    """Normalized FDE = FDE / trajectory_length."""
    if preds.ndim == 4:
        fde_per_mode = np.sqrt(((preds[:, :, -1] - gt[:, None, -1]) ** 2).sum(-1))
        best = fde_per_mode.argmin(1)
        preds_best = preds[np.arange(len(preds)), best]
    else:
        preds_best = preds
    fde = np.sqrt(((preds_best[:, -1] - gt[:, -1]) ** 2).sum(-1))
    traj_len = np.sqrt((np.diff(gt, axis=1) ** 2).sum(-1)).sum(-1)
    traj_len = np.maximum(traj_len, 1e-6)
    return (fde / traj_len).mean()


def compute_apd(preds, k=6):
    """Average Pairwise Distance (diversity). preds: (N, M, T, 2)."""
    preds = preds[:, :k]
    M = preds.shape[1]
    if M < 2:
        return 0.0
    final = preds[:, :, -1]  # (N, M, 2)
    total, cnt = 0.0, 0
    for i in range(M):
        for j in range(i + 1, M):
            total += np.sqrt(((final[:, i] - final[:, j]) ** 2).sum(-1)).mean()
            cnt += 1
    return total / cnt


def compute_nll(preds, gt, scores=None, sigma=1.0):
    """Negative log-likelihood under Gaussian mixture (per-timestep mean).
    
    Computes per-timestep mixture NLL then averages over timesteps,
    matching PrayagProjectV3's formulation for fair comparison.
    """
    N, M, T, _ = preds.shape
    if scores is None:
        scores = np.ones((N, M)) / M
    else:
        scores = np.exp(scores - scores.max(1, keepdims=True))
        scores = scores / scores.sum(1, keepdims=True)
    log_scores = np.log(scores + 1e-10)  # (N, M)
    log_norm = np.log(2 * np.pi * sigma ** 2)
    nll_per_sample = np.zeros(N)
    for n in range(N):
        nll_per_t = np.zeros(T)
        for t in range(T):
            # Per-mode log-Gaussian at timestep t: (M,)
            diff = preds[n, :, t] - gt[n, t]  # (M, 2)
            sq = (diff ** 2).sum(-1)  # (M,)
            log_gauss = -0.5 * sq / sigma ** 2 - log_norm  # (M,)
            weighted = log_scores[n] + log_gauss  # (M,)
            mx = weighted.max()
            nll_per_t[t] = -(mx + np.log(np.exp(weighted - mx).sum()))
        nll_per_sample[n] = nll_per_t.mean()
    return nll_per_sample.mean()


def compute_collision_rate(ego_preds, neighbor_futures, threshold=30.0):
    """Collision rate: fraction of samples with min dist < threshold.

    Threshold is in BEV pixels. Default 30px ~ one vehicle length (median
    BEV vehicle is ~36px long, ~22px wide). Two agents whose centers are
    within 30px are considered colliding.
    zero-padded timesteps at (0,0), which would incorrectly register as
    collisions if not filtered. This version skips individual timesteps where
    the neighbor position is exactly (0,0).
    """
    if neighbor_futures is None or len(neighbor_futures) == 0:
        return 0.0
    N = len(ego_preds)
    if ego_preds.ndim == 4:
        ego_preds = ego_preds[:, 0]  # Best mode
    collisions = 0
    for i in range(N):
        for n in range(neighbor_futures.shape[1]):
            if np.all(neighbor_futures[i, n] == 0):
                continue
            # Per-timestep padding filter: skip timesteps where neighbor is
            # at exactly (0,0), which indicates missing/padded data
            valid_t = ~np.all(neighbor_futures[i, n] == 0, axis=-1)  # (T,)
            if not np.any(valid_t):
                continue
            d = np.sqrt(((ego_preds[i] - neighbor_futures[i, n]) ** 2).sum(-1))
            d[~valid_t] = np.inf  # Ignore padded timesteps
            if d.min() < threshold:
                collisions += 1
                break
    return collisions / N


def _load_road_mask(annotation_path, width=1920, height=1080):
    """Load binary road mask from JSON annotation file. Returns (H, W) uint8 or None."""
    try:
        import cv2
    except ImportError:
        return None

    mask = np.zeros((height, width), dtype=np.uint8)
    try:
        with open(annotation_path, 'r') as f:
            import json as _json
            data = _json.load(f)
        for poly in data.get('polygons', []):
            verts = poly.get('vertices', [])
            ptype = poly.get('type', 'additive')
            if len(verts) >= 6:
                pts = []
                for j in range(0, len(verts), 2):
                    if j + 1 < len(verts):
                        x = int(np.clip(verts[j], 0, width - 1))
                        y = int(np.clip(verts[j + 1], 0, height - 1))
                        pts.append((x, y))
                if len(pts) >= 3:
                    arr = np.array(pts, dtype=np.int32)
                    if ptype == 'additive':
                        cv2.fillPoly(mask, [arr], 255)
                    elif ptype == 'subtractive':
                        cv2.fillPoly(mask, [arr], 0)
    except Exception:
        return None
    return mask


def compute_off_road_rate(ego_preds, origins, rotations, chunk_ids,
                          dataset_path, split='test'):
    """Compute off-road rate by converting ego-centric predictions to absolute
    pixel coordinates and checking against per-chunk road masks.

    Returns fraction of prediction points that fall outside the road polygon.
    """
    if origins is None or chunk_ids is None or dataset_path is None:
        return 0.0

    N, T, _ = ego_preds.shape

    # Load road masks per unique chunk (cache)
    road_masks = {}
    for cid in set(chunk_ids):
        ann = os.path.join(dataset_path, split, "annotations",
                           f"{cid}_road_annotation.json")
        if os.path.exists(ann):
            m = _load_road_mask(ann)
            if m is not None:
                road_masks[cid] = m

    if not road_masks:
        return 0.0

    off_road = 0
    total = 0
    for i in range(N):
        cid = chunk_ids[i]
        if cid not in road_masks:
            continue
        mask = road_masks[cid]
        H, W = mask.shape
        orig = np.asarray(origins[i], dtype=np.float64)
        rot = np.asarray(rotations[i], dtype=np.float64)
        rot_inv = np.linalg.inv(rot)
        for t in range(T):
            pos_ego = ego_preds[i, t].astype(np.float64)
            pos_abs = rot_inv @ pos_ego + orig
            xi = int(np.clip(np.round(pos_abs[0]), 0, W - 1))
            yi = int(np.clip(np.round(pos_abs[1]), 0, H - 1))
            total += 1
            if mask[yi, xi] == 0:
                off_road += 1

    return off_road / total if total > 0 else 0.0


def _pad_neighbor_futures(neighbor_list, T_pred):
    """Pad variable-length neighbor futures to a uniform (N, max_neigh, T, 2) array."""
    if not neighbor_list:
        return None
    max_n = max(nf.shape[0] for nf in neighbor_list)
    if max_n == 0:
        return None
    padded = np.zeros((len(neighbor_list), max_n, T_pred, 2))
    for i, nf in enumerate(neighbor_list):
        if nf.shape[0] > 0:
            padded[i, :nf.shape[0]] = nf
    return padded


def compute_all_metrics(preds, gt, scores=None, neighbor_futures=None,
                        origins=None, rotations=None, chunk_ids=None,
                        dataset_path=None, split='test'):
    """Compute all metrics. preds: (N, M, T, 2), gt: (N, T, 2)."""
    M = preds.shape[1]

    # Off-road rate: convert best-mode predictions to absolute pixel coords
    orr = 0.0
    if origins is not None and chunk_ids is not None and dataset_path is not None:
        best_mode = preds[:, 0]  # (N, T, 2)
        orr = compute_off_road_rate(best_mode, origins, rotations, chunk_ids,
                                     dataset_path, split)

    results = {
        "Samples": len(preds),
        "minADE@1": compute_min_ade_k(preds, gt, k=1),
        "minADE@4": compute_min_ade_k(preds, gt, k=min(4, M)),
        "minFDE@1": compute_min_fde_k(preds, gt, k=1),
        "minFDE@4": compute_min_fde_k(preds, gt, k=min(4, M)),
        "MR@10px": compute_miss_rate(preds, gt, threshold=10.0),
        "MR@20px": compute_miss_rate(preds, gt, threshold=20.0),
        "NormFDE": compute_norm_fde(preds, gt),
        "APD": compute_apd(preds, k=M),
        "NLL": compute_nll(preds, gt, scores=scores),
        "CR": compute_collision_rate(preds, neighbor_futures),
        "ORR": orr,
    }
    return results


# ============================================================================
# Model-specific loaders
# ============================================================================

def gpu(data, device):
    """Recursively move data to device."""
    if isinstance(data, list):
        return [gpu(x, device) for x in data]
    if isinstance(data, dict):
        return {k: gpu(v, device) for k, v in data.items()}
    if torch.is_tensor(data):
        return data.to(device)
    return data


def eval_prayag_lanegcn(checkpoint_path, dataset_path, dataset_type, device, max_agents=100, workers=4):
    """Evaluate PrayagLaneGCN on test set."""
    sys.path.insert(0, str(Path(__file__).parent / "PrayagLaneGCN"))
    try:
        from config import getConfig
        from model.lanegcn import LaneGCN
        from data.dataset import PrayagDataset, collateFn

        config = getConfig(dataset_type)
        config["datasetPath"] = str(dataset_path)
        config["maxAgents"] = max_agents

        # Load checkpoint and restore config if available
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
        all_neighbor_futures = []
        all_origins, all_rotations, all_chunk_ids = [], [], []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="PrayagLaneGCN eval"):
                # PrayagLaneGCN model internally moves data to GPU via _gpu()
                out = model(batch)
                B = len(out["reg"])
                for i in range(B):
                    # Ego agent (index 0)
                    p = out["reg"][i][0].detach().cpu().numpy()   # (M, T, 2)
                    s = out["cls"][i][0].detach().cpu().numpy()   # (M,)
                    gt_all_i = batch["gtPreds"][i]
                    if torch.is_tensor(gt_all_i):
                        gt_all_i = gt_all_i.numpy()
                    else:
                        gt_all_i = np.array(gt_all_i)
                    g = gt_all_i[0]  # (T, 2)
                    all_preds.append(p)
                    all_scores.append(s)
                    all_gt.append(g)

                    # Neighbor GT futures for collision rate
                    if gt_all_i.shape[0] > 1:
                        all_neighbor_futures.append(gt_all_i[1:])  # (N-1, T, 2)
                    else:
                        all_neighbor_futures.append(np.zeros((0, g.shape[0], 2)))

                    # Origin, rotation, chunk for ORR
                    orig_i = batch["orig"][i]
                    all_origins.append(orig_i.numpy() if torch.is_tensor(orig_i) else np.array(orig_i))
                    rot_i = batch["rot"][i]
                    all_rotations.append(rot_i.numpy() if torch.is_tensor(rot_i) else np.array(rot_i))
                    all_chunk_ids.append(batch["chunkId"][i])

        preds = np.array(all_preds)   # (N, M, T, 2)
        gt = np.array(all_gt)         # (N, T, 2)
        scores = np.array(all_scores) # (N, M)
        neigh = _pad_neighbor_futures(all_neighbor_futures, gt.shape[1])

        return compute_all_metrics(preds, gt, scores=scores, neighbor_futures=neigh,
                                   origins=all_origins, rotations=all_rotations,
                                   chunk_ids=all_chunk_ids,
                                   dataset_path=str(dataset_path), split='test')
    finally:
        sys.path.remove(str(Path(__file__).parent / "PrayagLaneGCN"))
        mods_to_remove = [
            m for m in sys.modules
            if m in ('config', 'model', 'data', 'utils')
            or m.startswith(('model.', 'data.', 'config.', 'utils.'))
        ]
        for m in mods_to_remove:
            del sys.modules[m]


def eval_prayag_gameformer(checkpoint_path, dataset_path, dataset_type, device, max_agents=100, workers=4):
    """Evaluate PrayagGameFormer on test set."""
    sys.path.insert(0, str(Path(__file__).parent / "PrayagGameFormer"))
    try:
        from config import getConfig
        from model.gameformer import GameFormer
        from data.dataset import PrayagDataset, collateFn

        config = getConfig(dataset_type)
        config["datasetDir"] = str(dataset_path)
        config["maxAgents"] = max_agents
        # Update all paths to point to custom dataset_path
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
        all_preds, all_gt, all_scores, all_neighbor_futures = [], [], [], []
        all_origins, all_chunk_ids = [], []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="PrayagGameFormer eval"):
                batch = gpu(batch, device)
                out = model(batch)
                final_key = f"level_{decoder_levels}_interactions"
                scores_key = f"level_{decoder_levels}_scores"
                interactions = out[final_key]  # (B, N, M, T, 4)
                sc = out[scores_key]           # (B, N, M)
                B = interactions.shape[0]
                # Ego = agent 0, xy = :2
                ego_preds = interactions[:, 0, :, :, :2].detach().cpu().numpy()  # (B, M, T, 2)
                ego_scores = sc[:, 0].detach().cpu().numpy()                     # (B, M)
                ego_gt = batch["ego_future"].cpu().numpy()                       # (B, T, 2)
                # Neighbor futures for collision rate
                neigh_fut = batch.get("neighbors_future", None)
                if neigh_fut is not None:
                    neigh_fut = neigh_fut.cpu().numpy()  # (B, N_neigh, T, 2)
                # Origin for ORR (added to dataset)
                origin_batch = batch.get("origin", None)
                if origin_batch is not None:
                    origin_batch = origin_batch.cpu().numpy()  # (B, 2)
                chunk_batch = batch.get("chunk", None)

                for i in range(B):
                    all_preds.append(ego_preds[i])
                    all_scores.append(ego_scores[i])
                    all_gt.append(ego_gt[i])
                    if neigh_fut is not None:
                        all_neighbor_futures.append(neigh_fut[i])
                    if origin_batch is not None:
                        all_origins.append(origin_batch[i])
                        all_chunk_ids.append(chunk_batch[i] if chunk_batch else None)

        preds = np.array(all_preds)
        gt = np.array(all_gt)
        scores = np.array(all_scores)
        neigh = np.array(all_neighbor_futures) if all_neighbor_futures else None

        # Origins and chunks for ORR
        origins = all_origins if all_origins else None
        rotations = [np.eye(2)] * len(all_preds) if origins else None  # No rotation in PrayagGameFormer
        chunk_ids = all_chunk_ids if all_chunk_ids else None

        return compute_all_metrics(preds, gt, scores=scores, neighbor_futures=neigh,
                                   origins=origins, rotations=rotations,
                                   chunk_ids=chunk_ids,
                                   dataset_path=str(dataset_path), split='test')
    finally:
        sys.path.remove(str(Path(__file__).parent / "PrayagGameFormer"))
        mods_to_remove = [
            m for m in sys.modules
            if m in ('config', 'model', 'data', 'utils')
            or m.startswith(('model.', 'data.', 'config.', 'utils.'))
        ]
        for m in mods_to_remove:
            del sys.modules[m]


def eval_lanegameformer(checkpoint_path, dataset_path, config_path, device, max_agents=100, batch_size=4, workers=4):
    """Evaluate LaneGameFormer on test set."""
    import yaml
    sys.path.insert(0, str(Path(__file__).parent / "LaneGameFormer"))
    try:
        from model.lane_game_former import LaneGameFormer
        from data.dataset import LaneGameFormerDataset, collate_fn

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        config['data']['dataset_path'] = str(dataset_path)
        if 'max_agents' in config.get('data', {}):
            config['data']['max_agents'] = max_agents
        # Ensure test horizon matches training horizon; disable long test override.
        if 'test_consecutive_frames' in config.get('data', {}):
            config['data']['test_consecutive_frames'] = None

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        saved_config = ckpt.get("config", None)
        if saved_config:
            # Restore model architecture params
            if 'model' in saved_config:
                config['model'] = saved_config['model']
            if 'data' in saved_config and 'max_agents' in saved_config['data']:
                config['data']['max_agents'] = saved_config['data']['max_agents']
            if 'ablation' in saved_config:
                config['ablation'] = saved_config['ablation']

        model = LaneGameFormer(config).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        test_ds = LaneGameFormerDataset(config, split="test")
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=workers, collate_fn=collate_fn, pin_memory=True)

        if config.get('ablation', {}).get('bypass_interaction_decoder', False):
            final_level_key = 'level_0'
        else:
            num_levels = config.get('model', {}).get('decoder', {}).get('num_decoder_layers', 3)
            final_level_key = f'level_{num_levels}'

        all_preds, all_gt, all_scores = [], [], []
        all_neighbor_futures = []
        all_origins, all_rotations, all_chunk_ids = [], [], []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="LaneGameFormer eval"):
                batch = gpu(batch, device)
                outputs = model(batch)
                # Use final decoder level (after all game-theoretic iterations)
                traj = outputs[final_level_key]['traj']       # (B, N_max, K, T, 2)
                sc = outputs[final_level_key]['scores']        # (B, N_max, K)
                B = len(batch['feats'])
                for i in range(B):
                    # Ego = agent 0
                    p = traj[i, 0].detach().cpu().numpy()   # (K, T, 2)
                    s = sc[i, 0].detach().cpu().numpy()      # (K,)
                    gt_all_i = batch['gt_preds'][i]
                    if torch.is_tensor(gt_all_i):
                        gt_all_i = gt_all_i.cpu().numpy()
                    else:
                        gt_all_i = np.array(gt_all_i)
                    g = gt_all_i[0]  # (T, 2)
                    all_preds.append(p)
                    all_scores.append(s)
                    all_gt.append(g)

                    # Neighbor GT futures for collision rate
                    if gt_all_i.shape[0] > 1:
                        all_neighbor_futures.append(gt_all_i[1:])  # (N-1, T, 2)
                    else:
                        all_neighbor_futures.append(np.zeros((0, g.shape[0], 2)))

                    # Origin, rotation, chunk for ORR
                    orig_i = batch['orig'][i]
                    all_origins.append(orig_i.cpu().numpy() if torch.is_tensor(orig_i) else np.array(orig_i))
                    rot_i = batch['rot'][i]
                    all_rotations.append(rot_i.cpu().numpy() if torch.is_tensor(rot_i) else np.array(rot_i))
                    all_chunk_ids.append(batch['chunk_id'][i])

        preds = np.array(all_preds)
        gt = np.array(all_gt)
        scores = np.array(all_scores)
        neigh = _pad_neighbor_futures(all_neighbor_futures, gt.shape[1])

        return compute_all_metrics(preds, gt, scores=scores, neighbor_futures=neigh,
                                   origins=all_origins, rotations=all_rotations,
                                   chunk_ids=all_chunk_ids,
                                   dataset_path=str(dataset_path), split='test')
    finally:
        sys.path.remove(str(Path(__file__).parent / "LaneGameFormer"))
        mods_to_remove = [
            m for m in sys.modules
            if m in ('config', 'model', 'data', 'utils')
            or m.startswith(('model.', 'data.', 'config.', 'utils.'))
        ]
        for m in mods_to_remove:
            del sys.modules[m]


def eval_prayag_projectv3(dataset_path, device, workers=4):
    """Evaluate PrayagProjectV3 (classical/mathematical baseline).
    
    PrayagProjectV3 is deterministic (no seeds, no checkpoints).
    Uses its own evaluation pipeline with ego-centric conversion.
    """
    sys.path.insert(0, str(Path(__file__).parent / "PrayagProjectv3"))
    try:
        import evaluate_system as pv3_eval
        import config as pv3_config
        
        # Run PrayagProjectV3 evaluation
        # It handles its own data loading, prediction, and metric computation
        metrics_agg = pv3_eval.evaluate_on_dataset(
            dataset_dir=str(dataset_path),
            fps=pv3_config.DEFAULT_FPS
        )
        
        # Reformat to unified metrics dict
        results = {}
        for key in ["minADE@1", "minADE@4", "minFDE@1", "minFDE@4",
                    "miss_rate_10", "miss_rate_20", "norm_fde", "apd", "nll",
                    "collision", "off_road"]:
            vals = metrics_agg.get(key, [])
            if vals:
                results[key] = float(np.mean(vals))
            else:
                results[key] = 0.0
        
        # Map to unified key names
        unified = {
            "Samples": len(metrics_agg.get("minADE@1", [])),
            "minADE@1": results.get("minADE@1", 0.0),
            "minADE@4": results.get("minADE@4", 0.0),
            "minFDE@1": results.get("minFDE@1", 0.0),
            "minFDE@4": results.get("minFDE@4", 0.0),
            "MR@10px": results.get("miss_rate_10", 0.0),
            "MR@20px": results.get("miss_rate_20", 0.0),
            "NormFDE": results.get("norm_fde", 0.0),
            "APD": results.get("apd", 0.0),
            "NLL": results.get("nll", 0.0),
            "CR": results.get("collision", 0.0),
            "ORR": results.get("off_road", 0.0),
        }
        return unified
    finally:
        sys.path.remove(str(Path(__file__).parent / "PrayagProjectv3"))
        mods_to_remove = [
            m for m in sys.modules
            if m.startswith(('evaluate_system', 'evaluation_metrics', 'road_mask',
                             'trajectory_prediction', 'track_loader', 'lane_extraction'))
        ]
        for m in mods_to_remove:
            del sys.modules[m]
# ============================================================================
# Main evaluation pipeline
# ============================================================================

SEEDS = [326915759, 1485336377, 1239516575]

EVAL_CONFIGS = {
    "PrayagLaneGCN_10Hz": {
        "model_type": "lanegcn",
        "dataset_path": "ChunkedProjectPrayagBEVDataset10Hz",
        "dataset_type": "10Hz",
    },
    "PrayagGameFormer_10Hz": {
        "model_type": "gameformer",
        "dataset_path": "ChunkedProjectPrayagBEVDataset10Hz",
        "dataset_type": "10Hz",
    },
    "LaneGameFormer_10Hz": {
        "model_type": "lanegameformer",
        "dataset_path": "ChunkedProjectPrayagBEVDataset10Hz",
        "config_path": "LaneGameFormer/configs/config.yaml",
        "max_agents": 100,
    },
    "LaneGameFormer_A1_10Hz": {
        "model_type": "lanegameformer",
        "dataset_path": "ChunkedProjectPrayagBEVDataset10Hz",
        "config_path": "LaneGameFormer/configs/config.yaml",
        "max_agents": 100,
    },
    "LaneGameFormer_A2_10Hz": {
        "model_type": "lanegameformer",
        "dataset_path": "ChunkedProjectPrayagBEVDataset10Hz",
        "config_path": "LaneGameFormer/configs/config.yaml",
        "max_agents": 100,
    },
    "LaneGameFormer_A3_10Hz": {
        "model_type": "lanegameformer",
        "dataset_path": "ChunkedProjectPrayagBEVDataset10Hz",
        "config_path": "LaneGameFormer/configs/config.yaml",
        "max_agents": 100,
    },
    "LaneGameFormer_A4_10Hz": {
        "model_type": "lanegameformer",
        "dataset_path": "ChunkedProjectPrayagBEVDataset10Hz",
        "config_path": "LaneGameFormer/configs/config.yaml",
        "max_agents": 100,
    },
    "LaneGameFormer_S1_10Hz": {
        "model_type": "lanegameformer",
        "dataset_path": "ChunkedProjectPrayagBEVDataset10Hz",
        "config_path": "LaneGameFormer/configs/config.yaml",
        "max_agents": 100,
    },
    "LaneGameFormer_K0_10Hz": {
        "model_type": "lanegameformer",
        "dataset_path": "ChunkedProjectPrayagBEVDataset10Hz",
        "config_path": "LaneGameFormer/configs/config.yaml",
        "max_agents": 100,
    },
    "LaneGameFormer_M0_10Hz": {
        "model_type": "lanegameformer",
        "dataset_path": "ChunkedProjectPrayagBEVDataset10Hz",
        "config_path": "LaneGameFormer/configs/config.yaml",
        "max_agents": 100,
    },
}

METRIC_KEYS = [
    "Samples", "minADE@1", "minADE@4", "minFDE@1", "minFDE@4",
    "MR@10px", "MR@20px", "NormFDE", "APD", "NLL", "CR", "ORR"
]


def json_default(obj):
    """JSON serializer for NumPy/Torch scalar and array types."""
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def evaluate_model_seed(name, cfg, seed, root, device, workers):
    """Evaluate one model + one seed."""
    import platform
    if platform.system() == 'Linux':
        base_output = Path("/mnt/c/Users/Xeron/LaneGameFormer_outputs/train")
    else:
        base_output = Path("C:/Users/Xeron/LaneGameFormer_outputs/train")
        
    ckpt_path = base_output / name / f"seed_{seed}" / "best.pth"
    if not ckpt_path.exists():
        print(f"  WARNING: {ckpt_path} not found, skipping")
        return None

    dataset_path = Path(root) / cfg["dataset_path"]
    max_agents = cfg.get("max_agents", 100)

    if cfg["model_type"] == "lanegcn":
        return eval_prayag_lanegcn(str(ckpt_path), str(dataset_path),
                                    cfg["dataset_type"], device, max_agents=max_agents, workers=workers)
    elif cfg["model_type"] == "gameformer":
        return eval_prayag_gameformer(str(ckpt_path), str(dataset_path),
                                      cfg["dataset_type"], device, max_agents=max_agents, workers=workers)
    elif cfg["model_type"] == "lanegameformer":
        config_path = Path(root) / cfg["config_path"]
        batch_size = cfg.get("batch_size", 4)
        return eval_lanegameformer(str(ckpt_path), str(dataset_path),
                                   str(config_path), device, max_agents=max_agents,
                                   batch_size=batch_size, workers=workers)


def evaluate_prayagv3(name, cfg, root, device, workers):
    """Evaluate PrayagProjectV3 (no seeds, no checkpoints, deterministic)."""
    dataset_path = Path(root) / cfg["dataset_path"]
    return eval_prayag_projectv3(str(dataset_path), device, workers=workers)


def aggregate_seeds(seed_results):
    """Compute mean ± std across seeds."""
    agg = {}
    for key in METRIC_KEYS:
        vals = [r[key] for r in seed_results if r is not None and key in r]
        if vals:
            agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        else:
            agg[key] = {"mean": 0.0, "std": 0.0}
    return agg


def format_val(mean, std, key):
    """Format metric value for display."""
    if key == "Samples":
        return str(int(mean))
    if "MR" in key or key in ("CR", "ORR"):
        return f"{mean*100:.2f}% ± {std*100:.2f}%"
    return f"{mean:.4f} ± {std:.4f}"


def generate_results_md(all_results, output_path):
    """Generate Results.md with tables and insights."""
    lines = []
    lines.append("# Trajectory Prediction Model Comparison Results\n")
    lines.append(f"**Evaluation Date**: {time.strftime('%Y-%m-%d %H:%M')}\n")
    lines.append(f"**Seeds**: {SEEDS} (from master seed 17)\n")
    lines.append("**Dataset**: ChunkedProjectPrayagBEVDataset (Indian traffic, BEV)\n")
    lines.append("**Test Split**: Held-out test chunks\n\n")

    # Per-dataset tables
    for hz in ["10Hz", "30Hz"]:
        lines.append(f"## {hz} Dataset Results\n")
        models_for_hz = {k: v for k, v in all_results.items() if hz in k}
        if not models_for_hz:
            lines.append("*No results available.*\n\n")
            continue

        # Table header
        display_metrics = [k for k in METRIC_KEYS if k != "Samples"]
        header = "| Model | " + " | ".join(display_metrics) + " |"
        sep = "|" + "|".join(["---"] * (len(display_metrics) + 1)) + "|"
        lines.append(header)
        lines.append(sep)

        for name, agg in models_for_hz.items():
            short_name = name.replace(f"_{hz}", "")
            row = f"| **{short_name}** |"
            for mk in display_metrics:
                m, s = agg[mk]["mean"], agg[mk]["std"]
                row += f" {format_val(m, s, mk)} |"
            lines.append(row)
        lines.append("")

    # Sample counts
    lines.append("## Sample Counts\n")
    lines.append("| Model | Test Samples |")
    lines.append("|---|---|")
    for name, agg in all_results.items():
        lines.append(f"| {name} | {int(agg['Samples']['mean'])} |")
    lines.append("")

    # Insights
    lines.append("## Key Insights\n")

    # Best model per metric per Hz
    for hz in ["10Hz", "30Hz"]:
        models_hz = {k: v for k, v in all_results.items() if hz in k}
        if not models_hz:
            continue
        lines.append(f"### {hz} Analysis\n")

        # Best minADE@1
        best_ade = min(models_hz.items(), key=lambda x: x[1]["minADE@1"]["mean"])
        lines.append(f"- **Best minADE@1**: {best_ade[0]} ({best_ade[1]['minADE@1']['mean']:.4f})")

        best_fde = min(models_hz.items(), key=lambda x: x[1]["minFDE@1"]["mean"])
        lines.append(f"- **Best minFDE@1**: {best_fde[0]} ({best_fde[1]['minFDE@1']['mean']:.4f})")

        best_mr = min(models_hz.items(), key=lambda x: x[1]["MR@10px"]["mean"])
        lines.append(f"- **Lowest MR@10px**: {best_mr[0]} ({best_mr[1]['MR@10px']['mean']*100:.2f}%)")

        most_diverse = max(models_hz.items(), key=lambda x: x[1]["APD"]["mean"])
        lines.append(f"- **Most Diverse (APD)**: {most_diverse[0]} ({most_diverse[1]['APD']['mean']:.4f})")

        best_nll = min(models_hz.items(), key=lambda x: x[1]["NLL"]["mean"])
        lines.append(f"- **Best NLL**: {best_nll[0]} ({best_nll[1]['NLL']['mean']:.4f})")
        lines.append("")

    # Cross-Hz comparison
    lines.append("### 10Hz vs 30Hz Comparison\n")
    for model_base in ["PrayagLaneGCN", "PrayagGameFormer", "LaneGameFormer"]:
        k10 = f"{model_base}_10Hz"
        k30 = f"{model_base}_30Hz"
        if k10 in all_results and k30 in all_results:
            ade10 = all_results[k10]["minADE@1"]["mean"]
            ade30 = all_results[k30]["minADE@1"]["mean"]
            diff = ((ade30 - ade10) / ade10) * 100
            direction = "worse" if diff > 0 else "better"
            lines.append(f"- **{model_base}**: 30Hz is {abs(diff):.1f}% {direction} "
                         f"(minADE@1: {ade10:.4f} → {ade30:.4f})")
    lines.append("")

    # Model complexity
    lines.append("### Model Characteristics\n")
    lines.append("| Model | Architecture | Multi-modal | Game-theoretic | Lane-aware |")
    lines.append("|---|---|---|---|---|")
    lines.append("| PrayagLaneGCN | LaneGCN (GNN) | 6 modes | No | Yes (graph) |")
    lines.append("| PrayagGameFormer | GameFormer (Transformer) | 6 modes | Yes (Level-k) | Yes (lanes) |")
    lines.append("| LaneGameFormer | LaneGCN + GameFormer | 6 modes | Yes (Level-k) | Yes (GNN + graph) |")
    lines.append("| PrayagProjectV3 | Mathematical (classical) | 6 modes | Yes (game theory) | Yes (extracted) |")
    lines.append("")

    # Methodology notes
    lines.append("## Methodology Notes\n")
    lines.append("- **Collision Rate (CR)**: Computed with per-timestep zero-padding filter. "
                 "In ego-centric coordinates, zero-padded neighbor timesteps sit at the origin "
                 "(ego position) and are excluded from distance checks to avoid false positives.")
    lines.append("- **Off-Road Rate (ORR)**: Predictions are converted from ego-centric to "
                 "absolute pixel coordinates using per-sample origin and rotation, then checked "
                 "against per-chunk road polygon masks (with extracted lanes embedded).")
    lines.append("- **Multi-modal Metrics (APD, minADE@K)**: APD measures average pairwise "
                 "distance between mode endpoints. A value near zero indicates mode collapse.")
    lines.append("")

    lines.append("---\n")
    lines.append("*Results generated by `evaluate_all_models.py`*\n")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\nResults written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate all models on test sets")
    parser.add_argument("--output-dir", type=str, default="./experiment_outputs/eval",
                        help="Directory to save evaluation results")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--models", type=str, nargs="*", default=None,
                        help="Specific models to evaluate (default: all)")
    parser.add_argument("--include-prayagv3", action="store_true",
                        help="Include PrayagProjectV3 (classical baseline) in evaluation")
    args = parser.parse_args()

    root = Path(__file__).parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("UNIFIED MODEL EVALUATION")
    print("=" * 70)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Seeds: {SEEDS}")
    print(f"Output: {args.output_dir}")
    print("=" * 70)

    configs_to_eval = EVAL_CONFIGS
    if args.models:
        configs_to_eval = {k: v for k, v in EVAL_CONFIGS.items() if k in args.models}

    all_results = {}

    for name, cfg in configs_to_eval.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"{'='*60}")

        seed_results = []
        for i, seed in enumerate(SEEDS):
            print(f"  Seed {i+1}/{len(SEEDS)}: {seed}")
            try:
                result = evaluate_model_seed(name, cfg, seed, root, device, args.workers)
                if result:
                    seed_results.append(result)
                    print(f"    minADE@1={result['minADE@1']:.4f}, "
                          f"minFDE@1={result['minFDE@1']:.4f}, "
                          f"MR@10={result['MR@10px']*100:.1f}%")
                # Clear GPU memory between seeds
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()

        if seed_results:
            agg = aggregate_seeds(seed_results)
            all_results[name] = agg
            print(f"\n  {name} Summary (mean ± std):")
            print(f"    minADE@1: {agg['minADE@1']['mean']:.4f} ± {agg['minADE@1']['std']:.4f}")
            print(f"    minFDE@1: {agg['minFDE@1']['mean']:.4f} ± {agg['minFDE@1']['std']:.4f}")

            # Save per-model results
            model_result_path = Path(args.output_dir) / f"{name}_eval.json"
            with open(model_result_path, 'w') as f:
                json.dump(
                    {"seeds": SEEDS, "per_seed": seed_results, "aggregate": agg},
                    f,
                    indent=2,
                    default=json_default,
                )

    # Optionally evaluate PrayagProjectV3 (classical baseline)
    if args.include_prayagv3:
        pv3_configs = {
            "PrayagProjectV3_30Hz": {
                "model_type": "prayagv3",
                "dataset_path": "ChunkedProjectPrayagBEVDataset",
            },
        }
        for name, cfg in pv3_configs.items():
            print(f"\n{'='*60}")
            print(f"Evaluating: {name} (Classical Baseline — no seeds)")
            print(f"{'='*60}")
            try:
                result = evaluate_prayagv3(name, cfg, root, device, args.workers)
                if result:
                    # Wrap in aggregate format (deterministic: std=0)
                    agg = {}
                    for key in METRIC_KEYS:
                        agg[key] = {"mean": float(result.get(key, 0.0)), "std": 0.0}
                    all_results[name] = agg
                    print(f"    minADE@1={result['minADE@1']:.4f}, "
                          f"minFDE@1={result['minFDE@1']:.4f}")
                    # Save per-model results
                    model_result_path = Path(args.output_dir) / f"{name}_eval.json"
                    with open(model_result_path, 'w') as f:
                        json.dump(
                            {"seeds": ["deterministic"], "per_seed": [result], "aggregate": agg},
                            f, indent=2, default=json_default
                        )
            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()

    # Save combined results
    combined_path = Path(args.output_dir) / "all_results.json"
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=json_default)

    # Generate Results.md
    results_md_path = Path(args.output_dir) / "Results.md"
    generate_results_md(all_results, str(results_md_path))

    # Also copy to workspace root
    root_results = root / "Results_raw.md"
    generate_results_md(all_results, str(root_results))

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
