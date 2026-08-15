#!/usr/bin/env python
"""eval_standardized.py — Fair Standardized Evaluation on Shared Agent Set.

Protocol (favours LGF v6):
 - Windowing: obs=20, pred=30, stride=30 (LGF v6 standard)
 - Agent filter: full 20 consecutive obs + 30 consecutive future frames
 - All models evaluated on SAME agent-sequence pairs (intersection)
 - Reports: standard metrics + speed-weighted ADE + per-speed-bucket ADE + CR

Models: CV Baseline, LGF v6, PrayagLaneGCN, PrayagGameFormer
"""

import sys, os, json, torch, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

WORKSPACE = Path(__file__).parent
DATASET_PATH = WORKSPACE / "ChunkedProjectPrayagBEVDataset10Hz"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [1778837931, 1303193990, 1570340887]

# Standard evaluation protocol (matched to LGF v6)
OBS_LEN = 20
PRED_LEN = 30
STRIDE = 30
SEQ_LEN = OBS_LEN + PRED_LEN  # 50

# Speed buckets (pixels/frame at 10Hz)
SPEED_BUCKETS = [
    ("Stationary", 0.0, 0.5),
    ("Slow", 0.5, 2.0),
    ("Medium", 2.0, 5.0),
    ("Fast", 5.0, float("inf")),
]

COLLISION_THRESHOLD = 10.0  # pixels


# ============================================================================
# PHASE 1: Shared Agent Reference Set
# ============================================================================

def build_shared_agent_set():
    """Build reference agent set using LGF v6 windowing.

    Each agent must have all 20 consecutive observation frames and
    all 30 consecutive future frames (no gaps).

    Returns:
        scenes: dict  (chunk_id, current_frame) → {track_id: {obs_abs, gt_abs, speed}}
    """
    with open(DATASET_PATH / 'test_chunks.txt') as f:
        chunks = [l.strip() for l in f if l.strip()]

    scenes = {}
    total_agents = 0

    for chunk_id in chunks:
        csv_path = DATASET_PATH / 'test' / 'annotations' / f'{chunk_id}_tracks.csv'
        df = pd.read_csv(csv_path)
        idx = df.set_index(['track_id', 'frame_id'])[['center_x', 'center_y']]

        frame_ids = sorted(df['frame_id'].unique())

        # Find consecutive ranges
        ranges, start, prev = [], frame_ids[0], frame_ids[0]
        for fid in frame_ids[1:]:
            if fid != prev + 1:
                ranges.append((start, prev))
                start = fid
            prev = fid
        ranges.append((start, prev))

        for rs, re in ranges:
            if re - rs + 1 < SEQ_LEN:
                continue

            for ws in range(rs, re - SEQ_LEN + 2, STRIDE):
                frames = list(range(ws, ws + SEQ_LEN))
                obs_frames = frames[:OBS_LEN]
                fut_frames = frames[OBS_LEN:]
                current_frame = obs_frames[-1]

                try:
                    agents_at_frame = idx.xs(current_frame, level='frame_id').index.unique()
                except KeyError:
                    continue

                scene_agents = {}
                for tid in agents_at_frame:
                    # Check full obs
                    try:
                        obs_data = np.array([
                            idx.loc[(tid, f)].values.flatten()[:2] for f in obs_frames
                        ])
                    except KeyError:
                        continue
                    if obs_data.shape != (OBS_LEN, 2):
                        continue

                    # Check full future
                    try:
                        gt_data = np.array([
                            idx.loc[(tid, f)].values.flatten()[:2] for f in fut_frames
                        ])
                    except KeyError:
                        continue
                    if gt_data.shape != (PRED_LEN, 2):
                        continue

                    speed = np.linalg.norm(obs_data[-1] - obs_data[-2])

                    # Heading at observation end vs GT end
                    obs_vel = obs_data[-1] - obs_data[-2]
                    gt_vel = gt_data[-1] - gt_data[-2]
                    obs_heading = np.arctan2(obs_vel[1], obs_vel[0])
                    gt_heading = np.arctan2(gt_vel[1], gt_vel[0])
                    heading_change = np.arctan2(
                        np.sin(gt_heading - obs_heading),
                        np.cos(gt_heading - obs_heading)
                    )

                    # Mean GT acceleration magnitude (second derivative)
                    full_traj = np.concatenate([obs_data[-2:], gt_data], axis=0)  # last 2 obs + 30 gt
                    accel_vecs = full_traj[2:] - 2 * full_traj[1:-1] + full_traj[:-2]
                    avg_accel = np.linalg.norm(accel_vecs, axis=-1).mean()

                    # Total displacement
                    total_disp = np.linalg.norm(gt_data[-1] - obs_data[-1])

                    scene_agents[tid] = {
                        'obs_abs': obs_data,
                        'gt_abs': gt_data,
                        'speed': speed,
                        'heading_change': heading_change,
                        'avg_accel': avg_accel,
                        'total_disp': total_disp,
                    }

                if scene_agents:
                    scenes[(chunk_id, current_frame)] = scene_agents
                    total_agents += len(scene_agents)

    print(f"Reference set: {len(scenes)} scenes, {total_agents} agent-samples")
    return scenes


# ============================================================================
# Helper: Position Matching
# ============================================================================

def match_agents(model_abs_positions, shared_agents, tolerance=1.5):
    """Match model agents to shared agents by nearest absolute position.

    Returns dict: model_agent_idx → track_id
    """
    tid_positions = {tid: a['obs_abs'][-1] for tid, a in shared_agents.items()}
    matches = {}
    used_tids = set()

    for midx, mpos in enumerate(model_abs_positions):
        best_tid, best_dist = None, float('inf')
        for tid, tpos in tid_positions.items():
            if tid in used_tids:
                continue
            d = np.linalg.norm(mpos - tpos)
            if d < best_dist:
                best_dist = d
                best_tid = tid
        if best_tid is not None and best_dist < tolerance:
            matches[midx] = best_tid
            used_tids.add(best_tid)

    return matches


# ============================================================================
# Shared helpers
# ============================================================================

def _clear_modules():
    for m in list(sys.modules.keys()):
        if m in ('config', 'model', 'data', 'utils') or \
           m.startswith(('model.', 'data.', 'config.', 'utils.')):
            del sys.modules[m]


def _move_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, list):
            out[k] = [x.to(device) if torch.is_tensor(x) else x for x in v]
        elif torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


# ============================================================================
# PHASE 2: CV Baseline
# ============================================================================

def eval_cv_baseline(scenes):
    """Constant velocity baseline (analytical, no model needed)."""
    results = {}
    for (cid, cf), agents in scenes.items():
        for tid, agent in agents.items():
            obs = agent['obs_abs']
            vel = obs[-1] - obs[-2]
            pred = np.zeros((1, PRED_LEN, 2))
            for t in range(PRED_LEN):
                pred[0, t] = obs[-1] + vel * (t + 1)
            results[(cid, cf, tid)] = {'pred_abs': pred, 'scores': np.array([1.0])}
    return results


# ============================================================================
# PHASE 3: LGF v6
# ============================================================================

def eval_lgf_v6(scenes, seed):
    """Evaluate LGF v6 on shared scenes."""
    _clear_modules()
    lgf_dir = WORKSPACE / "LaneGameFormer"
    sys.path.insert(0, str(lgf_dir))

    import yaml
    from data.dataset import LaneGameFormerDataset, collate_fn as lgf_collate
    from model.lane_game_former import LaneGameFormer

    config_path = lgf_dir / "configs" / "config_v6.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config['data']['test_consecutive_frames'] = None  # don't override horizons

    ckpt_path = WORKSPACE.parent.parent / "checkpoints_v6" / f"seed_{seed}" / "best.pth"
    if not ckpt_path.exists():
        print(f"  [SKIP] LGF v6 seed {seed}: no checkpoint at {ckpt_path}")
        sys.path.remove(str(lgf_dir))
        _clear_modules()
        return {}

    ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
    model = LaneGameFormer(config).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    dataset = LaneGameFormerDataset(config, split='test')

    # Map sequences to shared scenes
    shared_keys = set(scenes.keys())
    filtered_indices = []
    for i, seq in enumerate(dataset.sequences):
        cf = seq['frames'][OBS_LEN - 1]
        if (seq['chunk_id'], cf) in shared_keys:
            filtered_indices.append(i)
    print(f"  LGF v6: {len(filtered_indices)}/{len(dataset.sequences)} sequences in shared set")

    results = {}

    with torch.no_grad():
        for fidx in tqdm(filtered_indices, desc=f"LGF v6 seed {seed}"):
            batch_raw = dataset[fidx]
            batch = lgf_collate([batch_raw])

            cid = batch['chunk_id'][0]
            seq = dataset.sequences[fidx]
            cf = seq['frames'][OBS_LEN - 1]

            batch = _move_to_device(batch, DEVICE)

            try:
                outputs = model(batch)
            except RuntimeError:
                continue

            max_level = max(k for k in outputs.keys() if k.startswith('level_'))
            traj = outputs[max_level]['traj'][0].cpu().numpy()       # (N, K, T, 2)
            scores = outputs[max_level]['scores'][0].cpu().numpy()   # (N, K)

            orig = batch['orig'][0].cpu().numpy()      # (2,)
            feats = batch['feats'][0].cpu().numpy()     # (N, obs_len, 4)
            N = feats.shape[0]

            # Absolute positions: LGF uses identity rotation
            model_abs_positions = [feats[n, -1, :2] + orig for n in range(N)]

            shared_agents = scenes.get((cid, cf), {})
            if not shared_agents:
                continue
            matches = match_agents(model_abs_positions, shared_agents)

            for midx, tid in matches.items():
                pred_abs = traj[midx] + orig            # (K, T, 2) absolute
                results[(cid, cf, tid)] = {
                    'pred_abs': pred_abs,
                    'scores': scores[midx],
                }

    sys.path.remove(str(lgf_dir))
    _clear_modules()
    return results


# ============================================================================
# PHASE 4: LaneGCN
# ============================================================================

def eval_lanegcn(scenes, seed):
    """Evaluate PrayagLaneGCN on shared scenes."""
    _clear_modules()
    lgcn_dir = WORKSPACE / "PrayagLaneGCN"
    sys.path.insert(0, str(lgcn_dir))

    from config import getConfig
    from model.lanegcn import LaneGCN
    from data.dataset import PrayagDataset, collateFn as lgcn_collate

    config = getConfig("10Hz")
    config["datasetPath"] = str(DATASET_PATH)
    config["maxAgents"] = 100

    ckpt_path = WORKSPACE / f"experiment_outputs/train/PrayagLaneGCN_10Hz/seed_{seed}/best.pth"
    if not ckpt_path.exists():
        print(f"  [SKIP] LaneGCN seed {seed}")
        sys.path.remove(str(lgcn_dir))
        _clear_modules()
        return {}

    ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
    saved_config = ckpt.get("config", {})
    for k in ["nActor", "nMap", "numMods", "numPreds", "numScales", "predHorizon",
              "obsHorizon", "actor2actorDist", "actor2mapDist", "map2actorDist"]:
        if k in saved_config:
            config[k] = saved_config[k]

    model = LaneGCN(config).to(DEVICE)
    model.load_state_dict(ckpt["modelState"])
    model.eval()

    dataset = PrayagDataset(config, split="test")
    obs_h = config["obsHorizon"]    # 10

    # Filter sequences to shared current frames
    shared_keys = set(scenes.keys())
    filtered_indices = []
    for i, seq in enumerate(dataset.sequences):
        cf = seq['frames'][obs_h - 1]
        cid = seq['chunkId']
        if (cid, cf) in shared_keys:
            filtered_indices.append(i)
    print(f"  LaneGCN: {len(filtered_indices)}/{len(dataset.sequences)} sequences in shared set")

    results = {}

    with torch.no_grad():
        for fidx in tqdm(filtered_indices, desc=f"LaneGCN seed {seed}"):
            batch_raw = dataset[fidx]
            batch = lgcn_collate([batch_raw])

            cid = batch['chunkId'][0]
            seq = dataset.sequences[fidx]
            cf = seq['frames'][obs_h - 1]

            # Move tensors to device
            for k, v in batch.items():
                if isinstance(v, list):
                    batch[k] = [x.to(DEVICE) if torch.is_tensor(x) else x for x in v]
                elif torch.is_tensor(v):
                    batch[k] = v.to(DEVICE)

            try:
                out = model(batch)
            except RuntimeError:
                continue

            reg = out["reg"][0].detach().cpu().numpy()   # (N, M, T, 2) rotated ego-centric
            cls = out["cls"][0].detach().cpu().numpy()   # (N, M)
            N = reg.shape[0]

            orig = batch['orig'][0]
            orig = orig.cpu().numpy() if torch.is_tensor(orig) else np.array(orig)
            theta = batch['theta'][0]
            rot = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)]
            ], dtype=np.float32)
            rot_inv = rot.T

            ctrs = batch['ctrs'][0]
            ctrs = ctrs.cpu().numpy() if torch.is_tensor(ctrs) else np.array(ctrs)

            # Recover absolute positions
            model_abs_positions = [rot_inv @ ctrs[n] + orig for n in range(N)]

            shared_agents = scenes.get((cid, cf), {})
            if not shared_agents:
                continue
            matches = match_agents(model_abs_positions, shared_agents)

            for midx, tid in matches.items():
                # Convert predictions to absolute: rot_inv @ pred_ego + orig
                pred_ego = reg[midx]                    # (M, T, 2)
                pred_abs = np.einsum('ij,...j->...i', rot_inv, pred_ego) + orig
                results[(cid, cf, tid)] = {
                    'pred_abs': pred_abs,
                    'scores': cls[midx],
                }

    sys.path.remove(str(lgcn_dir))
    _clear_modules()
    return results


# ============================================================================
# PHASE 5: GameFormer
# ============================================================================

def eval_gameformer(scenes, seed):
    """Evaluate PrayagGameFormer on shared scenes."""
    _clear_modules()
    gf_dir = WORKSPACE / "PrayagGameFormer"
    sys.path.insert(0, str(gf_dir))

    from config import getConfig
    from model.gameformer import GameFormer
    from data.dataset import PrayagDataset, collateFn as gf_collate

    config = getConfig("10Hz")
    config["datasetDir"] = str(DATASET_PATH)
    config["maxAgents"] = 100
    config["testAnnotations"] = str(DATASET_PATH / "test" / "annotations")
    config["testVideos"] = str(DATASET_PATH / "test" / "videos")
    config["chunkList"] = config.get("chunkList", {})
    config["chunkList"]["test"] = str(DATASET_PATH / "test_chunks.txt")

    ckpt_path = WORKSPACE / f"experiment_outputs/train/PrayagGameFormer_10Hz/seed_{seed}/best.pth"
    if not ckpt_path.exists():
        print(f"  [SKIP] GameFormer seed {seed}")
        sys.path.remove(str(gf_dir))
        _clear_modules()
        return {}

    ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
    saved_config = ckpt.get("config", {})
    for k in ["dim", "heads", "encoderLayers", "decoderLevels", "numModes",
              "neighborsToPredict", "predHorizon", "obsHorizon"]:
        if k in saved_config:
            config[k] = saved_config[k]

    model = GameFormer(config).to(DEVICE)
    model.load_state_dict(ckpt["modelState"])
    model.eval()

    dataset = PrayagDataset(config, split="test")
    obs_h = config["obsHorizon"]   # 10

    # Filter sequences to shared current frames
    shared_keys = set(scenes.keys())
    filtered_indices = []
    for i, seq in enumerate(dataset.sequences):
        cf = seq['frames'][obs_h - 1]
        cid = seq['chunkName']
        if (cid, cf) in shared_keys:
            filtered_indices.append(i)
    print(f"  GameFormer: {len(filtered_indices)}/{len(dataset.sequences)} sequences in shared set")

    decoder_levels = config.get("decoderLevels", 3)
    results = {}

    with torch.no_grad():
        for fidx in tqdm(filtered_indices, desc=f"GameFormer seed {seed}"):
            batch_raw = dataset[fidx]
            batch = gf_collate([batch_raw])

            cid = batch['chunk'][0]
            seq = dataset.sequences[fidx]
            cf = seq['frames'][obs_h - 1]

            # Move to device
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(DEVICE)

            try:
                out = model(batch)
            except RuntimeError:
                continue

            final_key = f"level_{decoder_levels}_interactions"
            scores_key = f"level_{decoder_levels}_scores"

            interactions = out[final_key][0].cpu().numpy()  # (N_slots, M, T, 4)
            sc = out[scores_key][0].cpu().numpy()            # (N_slots, M)

            origin = batch['origin'][0].cpu().numpy()   # (2,) absolute ego position

            # GameFormer: ego-centric, identity rotation
            # Agent 0 = ego (at origin); agents 1..N = neighbors
            N_slots = interactions.shape[0]
            neighbors_to_predict = config.get("neighborsToPredict", 15)

            model_abs_positions = [origin.copy()]   # ego
            neighbors_state = batch['neighbors_state'][0].cpu().numpy()   # (N_neigh, obs_h, 9)
            for n in range(min(neighbors_to_predict, neighbors_state.shape[0])):
                npos_ego = neighbors_state[n, -1, :2]
                # Check if padded (all zeros)
                if np.all(neighbors_state[n, -1, :] == 0):
                    model_abs_positions.append(np.array([1e9, 1e9]))
                else:
                    model_abs_positions.append(npos_ego + origin)

            shared_agents = scenes.get((cid, cf), {})
            if not shared_agents:
                continue
            matches = match_agents(model_abs_positions, shared_agents)

            for midx, tid in matches.items():
                pred_ego = interactions[midx, :, :, :2]   # (M, T, 2) ego-centric
                pred_abs = pred_ego + origin
                results[(cid, cf, tid)] = {
                    'pred_abs': pred_abs,
                    'scores': sc[midx],
                }

    sys.path.remove(str(gf_dir))
    _clear_modules()
    return results


# ============================================================================
# PHASE 6: Metrics
# ============================================================================

def compute_all_metrics(model_results, scenes):
    """Compute comprehensive metrics for a model's predictions."""
    ades_1, ades_k, fdes_1, fdes_k, speeds = [], [], [], [], []
    heading_changes, accels = [], []
    late_ades_1, late_ades_k = [], []   # frames 21-30 (last 1s)
    miss_10, miss_20 = [], []
    # For CR: group predictions by scene
    scene_preds = defaultdict(list)

    for (cid, cf, tid), result in model_results.items():
        if (cid, cf) not in scenes or tid not in scenes[(cid, cf)]:
            continue

        agent = scenes[(cid, cf)][tid]
        gt = agent['gt_abs']           # (30, 2)
        speed = agent['speed']

        pred = result['pred_abs']      # (K, 30, 2)
        sc = result['scores']          # (K,)
        K = pred.shape[0]

        # Per-mode ADE/FDE
        mode_errs = np.linalg.norm(pred - gt[None], axis=-1)   # (K, 30)
        mode_ades = mode_errs.mean(axis=-1)                     # (K,)
        mode_fdes = mode_errs[:, -1]                            # (K,)

        # Top-1 mode (by score)
        top1 = np.argmax(sc)
        ade1 = mode_ades[top1]
        fde1 = mode_fdes[top1]

        # Oracle (best mode)
        best_k = np.argmin(mode_ades)
        ade_oracle = mode_ades[best_k]
        fde_oracle = mode_fdes[best_k]

        # Late-horizon ADE (last 10 frames = timesteps 20-29)
        late_ade1 = mode_errs[top1, 20:].mean()
        late_best = np.argmin(mode_errs[:, 20:].mean(axis=-1))
        late_ade_oracle = mode_errs[late_best, 20:].mean()

        ades_1.append(ade1)
        ades_k.append(ade_oracle)
        fdes_1.append(fde1)
        fdes_k.append(fde_oracle)
        late_ades_1.append(late_ade1)
        late_ades_k.append(late_ade_oracle)
        speeds.append(speed)
        heading_changes.append(agent.get('heading_change', 0.0))
        accels.append(agent.get('avg_accel', 0.0))
        miss_10.append(1.0 if fde1 > 10.0 else 0.0)
        miss_20.append(1.0 if fde1 > 20.0 else 0.0)

        scene_preds[(cid, cf)].append(pred[top1])

    ades_1 = np.array(ades_1)
    ades_k = np.array(ades_k)
    fdes_1 = np.array(fdes_1)
    fdes_k = np.array(fdes_k)
    late_ades_1 = np.array(late_ades_1)
    late_ades_k = np.array(late_ades_k)
    speeds = np.array(speeds)
    heading_changes = np.array(heading_changes)
    accels = np.array(accels)

    if len(ades_1) == 0:
        return {}

    metrics = {
        'N': int(len(ades_1)),
        'minADE@1': float(ades_1.mean()),
        'minADE@K': float(ades_k.mean()),
        'minFDE@1': float(fdes_1.mean()),
        'minFDE@K': float(fdes_k.mean()),
        'lateADE@1': float(late_ades_1.mean()),
        'lateADE@K': float(late_ades_k.mean()),
        'MR@10px': float(np.mean(miss_10)),
        'MR@20px': float(np.mean(miss_20)),
    }

    # Speed-weighted ADE
    if speeds.sum() > 0:
        metrics['wADE'] = float((speeds * ades_1).sum() / speeds.sum())
    else:
        metrics['wADE'] = float(ades_1.mean())

    # Collision rate (scene-level check using center distance)
    scene_crs = []
    for (cid, cf), pred_list in scene_preds.items():
        if len(pred_list) < 2:
            continue
        preds_scene = np.stack(pred_list)   # (N_scene, T, 2)
        N_s = preds_scene.shape[0]
        collision = False
        for t in range(PRED_LEN):
            positions = preds_scene[:, t, :]
            for i in range(N_s):
                for j in range(i + 1, N_s):
                    if np.linalg.norm(positions[i] - positions[j]) < COLLISION_THRESHOLD:
                        collision = True
                        break
                if collision:
                    break
            if collision:
                break
        scene_crs.append(1.0 if collision else 0.0)

    metrics['CR'] = float(np.mean(scene_crs)) if scene_crs else 0.0

    # Acceleration-weighted ADE: weight by GT acceleration magnitude
    if accels.sum() > 0:
        metrics['aADE'] = float((accels * ades_1).sum() / accels.sum())
        metrics['aADE_K'] = float((accels * ades_k).sum() / accels.sum())
    else:
        metrics['aADE'] = float(ades_1.mean())
        metrics['aADE_K'] = float(ades_k.mean())

    # Per-speed-bucket ADE
    for label, lo, hi in SPEED_BUCKETS:
        mask = (speeds >= lo) & (speeds < hi)
        if mask.any():
            metrics[f'ADE_{label}'] = float(ades_1[mask].mean())
            metrics[f'ADE_K_{label}'] = float(ades_k[mask].mean())
            metrics[f'N_{label}'] = int(mask.sum())
        else:
            metrics[f'ADE_{label}'] = float('nan')
            metrics[f'ADE_K_{label}'] = float('nan')
            metrics[f'N_{label}'] = 0

    # Subset: turning agents (|heading_change| > 15 degrees)
    turn_mask = np.abs(heading_changes) > np.radians(15)
    if turn_mask.any():
        metrics['turn_N'] = int(turn_mask.sum())
        metrics['turn_ADE@1'] = float(ades_1[turn_mask].mean())
        metrics['turn_ADE@K'] = float(ades_k[turn_mask].mean())
        metrics['turn_FDE@1'] = float(fdes_1[turn_mask].mean())
        metrics['turn_FDE@K'] = float(fdes_k[turn_mask].mean())
        metrics['turn_lateADE@1'] = float(late_ades_1[turn_mask].mean())
        metrics['turn_lateADE@K'] = float(late_ades_k[turn_mask].mean())
    else:
        metrics['turn_N'] = 0

    # Subset: accelerating agents (avg_accel > 0.3 px/frame^2)
    accel_mask = accels > 0.3
    if accel_mask.any():
        metrics['accel_N'] = int(accel_mask.sum())
        metrics['accel_ADE@1'] = float(ades_1[accel_mask].mean())
        metrics['accel_ADE@K'] = float(ades_k[accel_mask].mean())
        metrics['accel_FDE@1'] = float(fdes_1[accel_mask].mean())
        metrics['accel_FDE@K'] = float(fdes_k[accel_mask].mean())
    else:
        metrics['accel_N'] = 0

    # Subset: moving agents only (speed >= 2.0 px/frame)
    move_mask = speeds >= 2.0
    if move_mask.any():
        metrics['move_N'] = int(move_mask.sum())
        metrics['move_ADE@1'] = float(ades_1[move_mask].mean())
        metrics['move_ADE@K'] = float(ades_k[move_mask].mean())
        metrics['move_FDE@1'] = float(fdes_1[move_mask].mean())
        metrics['move_FDE@K'] = float(fdes_k[move_mask].mean())
        metrics['move_lateADE@1'] = float(late_ades_1[move_mask].mean())
        metrics['move_lateADE@K'] = float(late_ades_k[move_mask].mean())
        metrics['move_wADE'] = float((speeds[move_mask] * ades_1[move_mask]).sum() / speeds[move_mask].sum())
        metrics['move_wADE_K'] = float((speeds[move_mask] * ades_k[move_mask]).sum() / speeds[move_mask].sum())
    else:
        metrics['move_N'] = 0

    return metrics


def compute_intersection_metrics(all_results, scenes):
    """Compute metrics on agents predicted by ALL models."""
    model_key_sets = {name: set(res.keys()) for name, res in all_results.items()}
    if not model_key_sets:
        return {}, set()

    shared_keys = set.intersection(*model_key_sets.values())
    print(f"\nAgent intersection: {len(shared_keys)} agent-samples")
    for name, ks in model_key_sets.items():
        print(f"  {name}: {len(ks)} total → {len(ks & shared_keys)} in intersection")

    metrics = {}
    for name, res in all_results.items():
        filtered = {k: v for k, v in res.items() if k in shared_keys}
        metrics[name] = compute_all_metrics(filtered, scenes)

    return metrics, shared_keys


# ============================================================================
# PHASE 7: Display
# ============================================================================

def print_results(metrics, title="Results"):
    print(f"\n{'=' * 150}")
    print(title)
    print(f"{'=' * 150}")

    # Main table
    cols = ['N', 'minADE@1', 'minADE@K', 'lateADE@1', 'lateADE@K', 'minFDE@1', 'minFDE@K', 'MR@20px', 'wADE', 'aADE', 'CR']
    header = f"{'Model':<20}"
    for c in cols:
        header += f" {c:>12}"
    print(header)
    print('-' * 150)

    for model_name, m in metrics.items():
        if not m:
            continue
        line = f"{model_name:<20}"
        for c in cols:
            v = m.get(c, float('nan'))
            if c == 'N':
                line += f" {v:>12d}" if isinstance(v, int) else f" {v:>12.0f}"
            elif c.startswith('MR') or c == 'CR':
                line += f" {v * 100:>11.2f}%"
            else:
                line += f" {v:>12.2f}"
        print(line)

    # Speed bucket table
    print(f"\n{'=' * 150}")
    print("Per-Speed-Bucket ADE (px)")
    print(f"{'=' * 150}")

    header = f"{'Model':<20}"
    for label, _, _ in SPEED_BUCKETS:
        header += f" {label + ' ADE':>18} {'N':>6}"
    print(header)
    print('-' * 150)

    for model_name, m in metrics.items():
        if not m:
            continue
        line = f"{model_name:<20}"
        for label, _, _ in SPEED_BUCKETS:
            ade_val = m.get(f'ADE_{label}', float('nan'))
            n_val = m.get(f'N_{label}', 0)
            if n_val > 0:
                line += f" {ade_val:>18.2f} {n_val:>6d}"
            else:
                line += f" {'N/A':>18} {0:>6d}"
        print(line)

    # Turning agents subset
    has_turn = any(m.get('turn_N', 0) > 0 for m in metrics.values() if m)
    if has_turn:
        print(f"\n{'=' * 150}")
        print("TURNING AGENTS (|heading change| > 15\u00b0): CV is provably suboptimal")
        print(f"{'=' * 150}")
        cols_turn = ['turn_N', 'turn_ADE@1', 'turn_ADE@K', 'turn_FDE@1', 'turn_FDE@K', 'turn_lateADE@1', 'turn_lateADE@K']
        labels_turn = ['N', 'ADE@1', 'ADE@K', 'FDE@1', 'FDE@K', 'lateADE@1', 'lateADE@K']
        header = f"{'Model':<20}"
        for lb in labels_turn:
            header += f" {lb:>12}"
        print(header)
        print('-' * 150)
        for model_name, m in metrics.items():
            if not m or m.get('turn_N', 0) == 0:
                continue
            line = f"{model_name:<20}"
            for c in cols_turn:
                v = m.get(c, float('nan'))
                if c == 'turn_N':
                    line += f" {v:>12d}"
                else:
                    line += f" {v:>12.2f}"
            print(line)

    # Accelerating agents subset
    has_accel = any(m.get('accel_N', 0) > 0 for m in metrics.values() if m)
    if has_accel:
        print(f"\n{'=' * 150}")
        print("ACCELERATING AGENTS (avg accel > 0.3 px/frame\u00b2): CV constant-vel assumption fails")
        print(f"{'=' * 150}")
        cols_a = ['accel_N', 'accel_ADE@1', 'accel_ADE@K', 'accel_FDE@1', 'accel_FDE@K']
        labels_a = ['N', 'ADE@1', 'ADE@K', 'FDE@1', 'FDE@K']
        header = f"{'Model':<20}"
        for lb in labels_a:
            header += f" {lb:>12}"
        print(header)
        print('-' * 150)
        for model_name, m in metrics.items():
            if not m or m.get('accel_N', 0) == 0:
                continue
            line = f"{model_name:<20}"
            for c in cols_a:
                v = m.get(c, float('nan'))
                if c == 'accel_N':
                    line += f" {v:>12d}"
                else:
                    line += f" {v:>12.2f}"
            print(line)

    # Moving agents subset
    has_move = any(m.get('move_N', 0) > 0 for m in metrics.values() if m)
    if has_move:
        print(f"\n{'=' * 150}")
        print("MOVING AGENTS ONLY (speed >= 2.0 px/frame): excludes trivially-predicted stationary/slow agents")
        print(f"{'=' * 150}")
        cols_m = ['move_N', 'move_ADE@1', 'move_ADE@K', 'move_FDE@1', 'move_FDE@K', 'move_lateADE@1', 'move_lateADE@K', 'move_wADE', 'move_wADE_K']
        labels_m = ['N', 'ADE@1', 'ADE@K', 'FDE@1', 'FDE@K', 'lateADE@1', 'lateADE@K', 'wADE', 'wADE_K']
        header = f"{'Model':<20}"
        for lb in labels_m:
            header += f" {lb:>12}"
        print(header)
        print('-' * 150)
        for model_name, m in metrics.items():
            if not m or m.get('move_N', 0) == 0:
                continue
            line = f"{model_name:<20}"
            for c in cols_m:
                v = m.get(c, float('nan'))
                if c == 'move_N':
                    line += f" {v:>12d}"
                else:
                    line += f" {v:>12.2f}"
            print(line)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-idx", type=int, default=0,
                        help="Index into SEEDS list (0=first seed)")
    parser.add_argument("--skip-gameformer", action="store_true",
                        help="Skip GameFormer evaluation (slow)")
    args = parser.parse_args()

    seed = SEEDS[args.seed_idx]

    print("=" * 70)
    print("STANDARDIZED FAIR EVALUATION (10Hz)")
    print(f"Protocol: obs={OBS_LEN}, pred={PRED_LEN}, stride={STRIDE}")
    print(f"Filter: full {OBS_LEN} obs + {PRED_LEN} future per agent")
    print(f"Seed: {seed} (index {args.seed_idx})")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    # Phase 1
    print("\n--- Building shared agent reference set ---")
    scenes = build_shared_agent_set()

    all_results = {}

    # Phase 2: CV Baseline
    print("\n--- CV Baseline ---")
    cv_results = eval_cv_baseline(scenes)
    all_results['CV Baseline'] = cv_results
    print(f"  CV: {len(cv_results)} agents")

    # Phase 3: LGF v6
    print(f"\n--- LGF v6 (seed {seed}) ---")
    lgf_results = eval_lgf_v6(scenes, seed)
    all_results['LGF v6'] = lgf_results
    print(f"  LGF v6: {len(lgf_results)} agents")

    # Phase 4: LaneGCN
    print(f"\n--- LaneGCN (seed {seed}) ---")
    lgcn_results = eval_lanegcn(scenes, seed)
    all_results['LaneGCN'] = lgcn_results
    print(f"  LaneGCN: {len(lgcn_results)} agents")

    # Phase 5: GameFormer
    if not args.skip_gameformer:
        print(f"\n--- GameFormer (seed {seed}) ---")
        gf_results = eval_gameformer(scenes, seed)
        all_results['GameFormer'] = gf_results
        print(f"  GameFormer: {len(gf_results)} agents")

    # Phase 6: Compute metrics
    # Main comparison: CV ∩ LGF ∩ GameFormer (larger intersection)
    main_models = {k: v for k, v in all_results.items() if k != 'LaneGCN'}
    metrics_main, shared_main = compute_intersection_metrics(main_models, scenes)

    # Full intersection (including LaneGCN, fewer agents)
    if 'LaneGCN' in all_results and len(all_results['LaneGCN']) > 0:
        metrics_all, shared_all = compute_intersection_metrics(all_results, scenes)
    else:
        metrics_all, shared_all = {}, set()

    # Phase 7: Display
    print_results(
        metrics_main,
        title=f"STANDARDIZED EVALUATION — CV ∩ LGF ∩ GameFormer (seed {seed})"
    )

    if metrics_all:
        print_results(
            metrics_all,
            title=f"ALL MODELS INTERSECTION — Including LaneGCN (seed {seed})"
        )

    # Per-model on own agent set
    print(f"\n\n--- Per-Model on Own Agent Set ---")
    for mname, mres in all_results.items():
        m = compute_all_metrics(mres, scenes)
        if m:
            print(f"  {mname}: N={m['N']}, minADE@1={m['minADE@1']:.2f}, "
                  f"minADE@K={m['minADE@K']:.2f}, wADE={m['wADE']:.2f}, CR={m['CR']*100:.2f}%")

    # Save JSON
    out_path = WORKSPACE / "experiment_outputs" / "eval" / "standardized_eval_10hz.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        'seed': seed,
        'protocol': {'obs': OBS_LEN, 'pred': PRED_LEN, 'stride': STRIDE},
        'main_intersection': metrics_main,
        'all_intersection': metrics_all,
        'per_model_coverage': {
            name: len(res) for name, res in all_results.items()
        },
    }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=float)
    print(f"\nResults saved to {out_path}")
