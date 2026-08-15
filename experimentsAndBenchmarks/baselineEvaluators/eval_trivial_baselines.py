"""
Evaluation of Constant-Velocity and Static baselines on LGF test protocol.
Computes metrics in both pixels and meters, ego-only.
"""
import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

# Insert LaneGameFormer path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "LaneGameFormer"))

from data.dataset import LaneGameFormerDataset, collate_fn
from evaluate_all_models import (
    compute_min_ade_k, compute_min_fde_k, compute_miss_rate,
    compute_norm_fde, compute_collision_rate, compute_off_road_rate,
    _pad_neighbor_futures
)

# Resolution and scaling parameters
BEV_PIXELS_PER_METER = 8.734845
DATASET_PATH = "ChunkedProjectPrayagBEVDataset10Hz"

config = {
    'data': {
        'dataset_path': DATASET_PATH,
        'obs_horizon': 20,
        'pred_horizon': 30,
        'max_agents': 32,
    },
    'flow_surface': {
        'resolution': [1080, 1920],
        'sigma': 30.0,
        'min_confidence': 0.1,
    },
    'model': {
        'encoder': {
            'in_dim': 7,
        }
    }
}

def evaluate_trivial():
    print(f"Loading LaneGameFormer dataset for split: test...")
    dataset = LaneGameFormerDataset(config, split="test")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    cv_preds_all = []
    static_preds_all = []
    gt_all = []
    neighbor_futures_all = []
    origins_all = []
    rotations_all = []
    chunks_all = []
    
    for idx, batch in enumerate(loader):
        # Ego is index 0
        feats = batch['feats'][0]  # (N, obs, dim)
        gt_preds = batch['gt_preds'][0]  # (N, pred, 2)
        has_preds = batch['has_preds'][0]  # (N, pred)
        orig = batch['orig'][0]
        rot = batch['rot'][0]
        chunk_id = batch['chunk_id'][0]
        
        if len(feats) == 0:
            continue
            
        # Ego history coordinates
        ego_hist = feats[0, :, :2].cpu().numpy()  # (obs, 2)
        ego_gt = gt_preds[0].cpu().numpy()  # (pred, 2)
        ego_has_pred = has_preds[0].cpu().numpy()  # (pred)
        
        # Last two observed frames
        pos_cur = ego_hist[-1]
        pos_prev = ego_hist[-2]
        
        # 1. Constant Velocity Baseline
        vel = pos_cur - pos_prev
        t_range = np.arange(1, 31, dtype=np.float32)
        pred_cv = pos_cur[np.newaxis] + vel[np.newaxis] * t_range[:, np.newaxis]  # (30, 2)
        
        # 2. Static Baseline
        pred_static = np.tile(pos_cur, (30, 1))  # (30, 2)
        
        # Add mode dimension for GMM interface compatibility: (1, 30, 2)
        cv_preds_all.append(pred_cv[np.newaxis])
        static_preds_all.append(pred_static[np.newaxis])
        gt_all.append(ego_gt)
        
        # Collated neighbor ground truth futures for collision rate
        if gt_preds.shape[0] > 1:
            neighbor_futures_all.append(gt_preds[1:].cpu().numpy())
        else:
            neighbor_futures_all.append(np.zeros((0, 30, 2)))
            
        # Context metadata
        origins_all.append(orig.cpu().numpy() if torch.is_tensor(orig) else np.array(orig))
        rotations_all.append(rot.cpu().numpy() if torch.is_tensor(rot) else np.array(rot))
        chunks_all.append(chunk_id)
        
    cv_preds = np.array(cv_preds_all)      # (N_samples, 1, 30, 2)
    static_preds = np.array(static_preds_all)  # (N_samples, 1, 30, 2)
    gt = np.array(gt_all)                  # (N_samples, 30, 2)
    
    # Pad neighbor futures
    neigh = _pad_neighbor_futures(neighbor_futures_all, 30)
    
    baselines = {
        "Constant Velocity (CV)": cv_preds,
        "Static (Zero Velocity)": static_preds
    }
    
    final_results = {}
    
    for name, preds in baselines.items():
        print(f"\nEvaluating baseline: {name}")
        
        # Pixel-scale metrics
        ade = compute_min_ade_k(preds, gt, k=1)
        fde = compute_min_fde_k(preds, gt, k=1)
        mr10 = compute_miss_rate(preds, gt, threshold=10.0)
        mr20 = compute_miss_rate(preds, gt, threshold=20.0)
        norm_fde = compute_norm_fde(preds, gt)
        
        cr = compute_collision_rate(preds, neigh, threshold=10.0)
        orr = compute_off_road_rate(preds[:, 0], origins_all, rotations_all, chunks_all,
                                     dataset_path=DATASET_PATH, split='test')
        
        # Metric scale (meters) projection
        ade_m = ade / BEV_PIXELS_PER_METER
        fde_m = fde / BEV_PIXELS_PER_METER
        
        results_px = {
            "minADE@1": float(ade),
            "minADE@4": float(ade),
            "minFDE@1": float(fde),
            "minFDE@4": float(fde),
            "MR@10px": float(mr10),
            "MR@20px": float(mr20),
            "NormFDE": float(norm_fde),
            "APD": 0.0,
            "NLL": "N/A",
            "CR": float(cr),
            "ORR": float(orr)
        }
        
        results_m = {
            "minADE@1": float(ade_m),
            "minADE@4": float(ade_m),
            "minFDE@1": float(fde_m),
            "minFDE@4": float(fde_m),
            "MR@1.14m": float(mr10),
            "MR@2.29m": float(mr20),
            "NormFDE": float(norm_fde),
            "APD": 0.0,
            "NLL": "N/A",
            "CR": float(cr),
            "ORR": float(orr)
        }
        
        final_results[name] = {
            "pixels": results_px,
            "meters": results_m,
            "samples": len(preds)
        }
        
        print(f"  Samples: {len(preds)}")
        print(f"  minADE@1: {ade:.4f} px ({ade_m:.4f} m)")
        print(f"  minFDE@1: {fde:.4f} px ({fde_m:.4f} m)")
        print(f"  Collision Rate: {cr * 100:.2f}%")
        print(f"  Off-Road Rate:  {orr * 100:.2f}%")

    out_path = "experiment_outputs/eval/trivial_baselines_results.json"
    with open(out_path, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nSaved trivial baselines results to {out_path}")
    
    # Also save to paper folder
    paper_out_path = "paper/code/experiment_outputs/eval/trivial_baselines_results.json"
    os.makedirs(os.path.dirname(paper_out_path), exist_ok=True)
    with open(paper_out_path, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"Saved copy to {paper_out_path}")

if __name__ == "__main__":
    evaluate_trivial()
