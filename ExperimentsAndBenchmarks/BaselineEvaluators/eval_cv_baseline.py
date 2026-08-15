"""Constant-Velocity (CV) baseline evaluation.

For each agent, predicts future positions by extrapolating the velocity
estimated from the last two observed positions:
    v = pos[obs-1] - pos[obs-2]
    pred[t] = pos[obs-1] + v * (t+1)

Single-mode baseline: minADE@1 == minADE@4, APD = 0.
No probabilistic output, so NLL is reported as N/A.

Evaluates on both 10Hz and 30Hz datasets, all-agents.
Uses the fixed center-distance CR metric (threshold=10px).
"""

import sys, os, json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

WORKSPACE = Path(__file__).parent

# Dataset configs: (dataset_path, obs_horizon, pred_horizon, fps)
DATASET_CONFIGS = {
    "10Hz": {
        "path": WORKSPACE / "ChunkedProjectPrayagBEVDataset10Hz",
        "obs": 10,   # 1s observation (matches LaneGCN/GameFormer)
        "pred": 30,  # 3s prediction
        "fps": 10,
    },
    "30Hz": {
        "path": WORKSPACE / "ChunkedProjectPrayagBEVDataset",
        "obs": 30,   # 1s observation (matches LaneGCN/GameFormer)
        "pred": 90,  # 3s prediction
        "fps": 30,
    },
}


def load_test_sequences(dataset_path, split, obs_horizon, pred_horizon, fps):
    """Load test sequences from chunk CSVs, matching LaneGCN windowing."""
    seq_len = obs_horizon + pred_horizon
    stride = max(1, fps // 2)

    chunk_file = dataset_path / f"{split}_chunks.txt"
    with open(chunk_file) as f:
        chunk_ids = [line.strip() for line in f if line.strip()]

    sequences = []
    for chunk_id in chunk_ids:
        tracks_path = dataset_path / split / "annotations" / f"{chunk_id}_tracks.csv"
        if not tracks_path.exists():
            print(f"  Warning: {tracks_path} not found, skipping")
            continue

        df = pd.read_csv(tracks_path)
        frame_ids = sorted(df['frame_id'].unique())

        # Find consecutive frame ranges
        ranges = []
        if len(frame_ids) > 0:
            start = prev = frame_ids[0]
            for fid in frame_ids[1:]:
                if fid != prev + 1:
                    ranges.append((start, prev))
                    start = fid
                prev = fid
            ranges.append((start, prev))

        for r_start, r_end in ranges:
            r_len = r_end - r_start + 1
            if r_len < seq_len:
                continue
            for s in range(r_start, r_end - seq_len + 2, stride):
                frames = list(range(s, s + seq_len))
                sequences.append({
                    "chunk_id": chunk_id,
                    "frames": frames,
                    "tracks_path": str(tracks_path),
                })

    return sequences


def evaluate_cv_baseline(freq_label):
    """Run CV baseline on one dataset frequency."""
    cfg = DATASET_CONFIGS[freq_label]
    dataset_path = cfg["path"]
    obs_h = cfg["obs"]
    pred_h = cfg["pred"]
    fps = cfg["fps"]

    print(f"\n{'='*60}")
    print(f"CV Baseline — {freq_label} ({dataset_path.name})")
    print(f"  obs={obs_h} frames ({obs_h/fps:.1f}s), pred={pred_h} frames ({pred_h/fps:.1f}s)")
    print(f"{'='*60}")

    sequences = load_test_sequences(dataset_path, "test", obs_h, pred_h, fps)
    print(f"  {len(sequences)} test sequences from {len(set(s['chunk_id'] for s in sequences))} chunks")

    # Import collision metric
    sys.path.insert(0, str(WORKSPACE / "LaneGameFormer"))
    from utils.metrics import check_collision_scene
    sys.path.pop(0)

    import torch

    all_ade = []
    all_fde = []
    all_fde_for_mr10 = []
    all_fde_for_mr20 = []
    scene_cr = []
    max_agents = 100

    df_cache = {}

    for seq_idx, seq in enumerate(sequences):
        tracks_path = seq["tracks_path"]
        if tracks_path not in df_cache:
            df = pd.read_csv(tracks_path)
            # Pre-index by (track_id, frame_id) for fast lookups
            df_indexed = df.set_index(['track_id', 'frame_id'])[['center_x', 'center_y']]
            df_cache[tracks_path] = df_indexed
        df_indexed = df_cache[tracks_path]

        frames = seq["frames"]
        obs_frames = frames[:obs_h]
        fut_frames = frames[obs_h:]
        current_frame = obs_frames[-1]
        prev_frame = obs_frames[-2]

        # Agents present at current frame
        try:
            current_agents = df_indexed.xs(current_frame, level='frame_id').index.unique().tolist()
        except KeyError:
            continue
        if len(current_agents) > max_agents:
            current_agents = current_agents[:max_agents]

        cv_preds_scene = []   # list of (T, 2) for CR
        has_preds_scene = []  # list of (T,) bool

        for agent_id in current_agents:
            # Get position at last two obs frames via index
            try:
                pos_cur = df_indexed.loc[(agent_id, current_frame)].values
                if pos_cur.ndim > 1:
                    pos_cur = pos_cur[0]
            except KeyError:
                continue

            try:
                pos_prev = df_indexed.loc[(agent_id, prev_frame)].values
                if pos_prev.ndim > 1:
                    pos_prev = pos_prev[0]
                vel = pos_cur - pos_prev
            except KeyError:
                vel = np.zeros(2)

            # CV prediction: extrapolate
            t_range = np.arange(1, pred_h + 1, dtype=np.float32)
            pred = pos_cur[np.newaxis] + vel[np.newaxis] * t_range[:, np.newaxis]

            # Ground truth future — vectorized lookup
            gt = np.zeros((pred_h, 2), dtype=np.float32)
            hp = np.zeros(pred_h, dtype=bool)
            try:
                agent_data = df_indexed.loc[agent_id]
                for t, fid in enumerate(fut_frames):
                    if fid in agent_data.index:
                        row_vals = agent_data.loc[fid]
                        if hasattr(row_vals, 'values'):
                            gt[t] = row_vals.values[:2] if row_vals.values.ndim == 1 else row_vals.values[0][:2]
                        else:
                            gt[t] = [row_vals.iloc[0], row_vals.iloc[1]] if hasattr(row_vals, 'iloc') else row_vals
                        hp[t] = True
            except KeyError:
                pass

            # Need last frame valid for FDE (matching other models)
            if not hp[-1]:
                # Still include for CR if any valid frames
                if hp.any():
                    cv_preds_scene.append(pred)
                    has_preds_scene.append(hp)
                continue

            cv_preds_scene.append(pred)
            has_preds_scene.append(hp)

            # Per-agent displacement metrics (only on valid timesteps)
            errors = np.linalg.norm(pred[hp] - gt[hp], axis=-1)
            ade = errors.mean()
            fde = np.linalg.norm(pred[-1] - gt[-1])
            all_ade.append(ade)
            all_fde.append(fde)
            all_fde_for_mr10.append(fde > 10.0)
            all_fde_for_mr20.append(fde > 20.0)

        # Scene-level CR
        if len(cv_preds_scene) >= 2:
            preds_t = torch.tensor(np.array(cv_preds_scene), dtype=torch.float32)
            hp_t = torch.tensor(np.array(has_preds_scene), dtype=torch.bool)
            cr = check_collision_scene(preds_t, has_preds=hp_t)
            scene_cr.append(cr)

    n_samples = len(all_ade)
    results = {
        "Samples": n_samples,
        "minADE@1": float(np.mean(all_ade)),
        "minADE@4": float(np.mean(all_ade)),  # same (1 mode)
        "minFDE@1": float(np.mean(all_fde)),
        "minFDE@4": float(np.mean(all_fde)),
        "MR@10px": float(np.mean(all_fde_for_mr10)),
        "MR@20px": float(np.mean(all_fde_for_mr20)),
        "APD": 0.0,
        "NLL": None,  # Not applicable for point predictions
        "CR": float(np.mean(scene_cr)) if scene_cr else 0.0,
        "ORR": 0.0,
    }

    print(f"\n  Results ({freq_label}):")
    print(f"    Samples:   {results['Samples']}")
    print(f"    minADE@1:  {results['minADE@1']:.2f} px")
    print(f"    minFDE@1:  {results['minFDE@1']:.2f} px")
    print(f"    MR@10px:   {results['MR@10px']*100:.2f}%")
    print(f"    MR@20px:   {results['MR@20px']*100:.2f}%")
    print(f"    CR:        {results['CR']*100:.2f}%")

    return results


if __name__ == "__main__":
    all_results = {}
    for freq in ["10Hz", "30Hz"]:
        all_results[freq] = evaluate_cv_baseline(freq)

    # Save results
    out_path = WORKSPACE / "experiment_outputs" / "eval" / "cv_baseline_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert None to string for JSON
    save_results = {}
    for k, v in all_results.items():
        save_results[k] = {mk: mv if mv is not None else "N/A" for mk, mv in v.items()}

    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {out_path}")
