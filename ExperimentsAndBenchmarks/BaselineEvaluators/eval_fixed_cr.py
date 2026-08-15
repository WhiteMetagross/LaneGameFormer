"""Evaluate ALL models with the FIXED collision rate metric.

Runs:
1. LGF v6 (3 seeds) via evaluate.py
2. PrayagLaneGCN (3 seeds) via eval_all_agents.py approach
3. PrayagGameFormer (3 seeds) via eval_all_agents.py approach
4. PV3 (deterministic)

All use the corrected center-distance CR (threshold=10px, GT=0%).
Results saved per-model and a unified comparison printed.
"""
import sys, os, json, torch, numpy as np
from pathlib import Path
from collections import defaultdict

WORKSPACE = Path(__file__).parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [1778837931, 1303193990, 1570340887]
DATASET_PATH = WORKSPACE / "ChunkedProjectPrayagBEVDataset10Hz"


def eval_lgf_v6():
    """Evaluate LGF v6 for all 3 seeds on test set (all agents)."""
    lgf_dir = WORKSPACE / "LaneGameFormer"
    sys.path.insert(0, str(lgf_dir))
    os.chdir(str(lgf_dir))

    from scripts.evaluate import evaluate

    docs_dir = WORKSPACE.parent.parent  # .../Documents
    ckpt_base = docs_dir / "checkpoints_v6"
    config_path = str(lgf_dir / "configs" / "config_v6.yaml")

    results = []
    for seed in SEEDS:
        ckpt = ckpt_base / f"seed_{seed}" / "best.pth"
        if not ckpt.exists():
            print(f"  [SKIP] LGF v6 seed {seed}: no checkpoint")
            continue
        print(f"\n--- LGF v6 seed {seed} ---")
        out_dir = str(ckpt.parent)
        r = evaluate(
            config_path=config_path,
            checkpoint_path=str(ckpt),
            split='test',
            output_dir=out_dir,
            ego_only=False,
        )
        if r:
            results.append(r)
            print(f"  minADE@1={r['minADE_1']:.2f}, CR={r['collision_rate']*100:.2f}%")

    os.chdir(str(WORKSPACE))
    sys.path.remove(str(lgf_dir))
    return results


def _clear_modules():
    for m in [k for k in sys.modules if k in ('config', 'model', 'data', 'utils')
              or k.startswith(('model.', 'data.', 'config.', 'utils.'))]:
        del sys.modules[m]


def eval_lanegcn():
    """Evaluate PrayagLaneGCN for all 3 seeds, compute CR with center-distance."""
    _clear_modules()
    sys.path.insert(0, str(WORKSPACE / "LaneGameFormer"))
    from utils.metrics import check_collision_scene
    sys.path.remove(str(WORKSPACE / "LaneGameFormer"))
    _clear_modules()

    sys.path.insert(0, str(WORKSPACE / "PrayagLaneGCN"))
    from config import getConfig
    from model.lanegcn import LaneGCN
    from data.dataset import PrayagDataset, collateFn
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    config = getConfig("10Hz")
    config["datasetPath"] = str(DATASET_PATH)
    config["maxAgents"] = 100

    results = []
    for seed in SEEDS:
        ckpt_path = WORKSPACE / f"experiment_outputs/train/PrayagLaneGCN_10Hz/seed_{seed}/best.pth"
        if not ckpt_path.exists():
            print(f"  [SKIP] LaneGCN seed {seed}")
            continue
        print(f"\n--- PrayagLaneGCN seed {seed} ---")

        ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
        saved_config = ckpt.get("config", {})
        for k in ["nActor", "nMap", "numMods", "numPreds", "numScales", "predHorizon",
                   "obsHorizon", "actor2actorDist", "actor2mapDist", "map2actorDist"]:
            if k in saved_config:
                config[k] = saved_config[k]

        model = LaneGCN(config).to(DEVICE)
        model.load_state_dict(ckpt["modelState"])
        model.eval()

        test_ds = PrayagDataset(config, split="test")
        test_loader = DataLoader(test_ds, batch_size=16, shuffle=False,
                                 num_workers=0, collate_fn=collateFn, pin_memory=True)

        all_preds, all_gt, all_scores = [], [], []
        scene_collision_fracs = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"LaneGCN seed {seed}"):
                out = model(batch)
                B = len(out["reg"])
                for i in range(B):
                    reg_i = out["reg"][i]    # (N, M, T, 2)
                    cls_i = out["cls"][i]    # (N, M)
                    gt_all_i = batch["gtPreds"][i]
                    if torch.is_tensor(gt_all_i):
                        gt_all_i = gt_all_i
                    else:
                        gt_all_i = torch.tensor(gt_all_i, dtype=torch.float32)

                    has_preds_i = batch.get("hasPreds", None)
                    if has_preds_i is not None:
                        hp_i = has_preds_i[i]
                        if not torch.is_tensor(hp_i):
                            hp_i = torch.tensor(hp_i, dtype=torch.bool)
                    else:
                        hp_i = None

                    N_agents = reg_i.shape[0]
                    # Per-agent metrics
                    for n in range(N_agents):
                        g = gt_all_i[n].numpy() if torch.is_tensor(gt_all_i[n]) else np.array(gt_all_i[n])
                        if np.linalg.norm(g) < 1e-6:
                            continue
                        r_n = reg_i[n].detach().cpu().numpy()
                        s_n = cls_i[n].detach().cpu().numpy()
                        all_preds.append(r_n)
                        all_scores.append(s_n)
                        all_gt.append(g)

                    # Scene-level collision (top-1 mode)
                    if N_agents >= 2:
                        top1_idx = cls_i.argmax(dim=-1)  # (N,)
                        T = reg_i.shape[2]
                        top1 = torch.gather(
                            reg_i, 1,
                            top1_idx.view(N_agents, 1, 1, 1).expand(N_agents, 1, T, 2)
                        ).squeeze(1).cpu()  # (N, T, 2)
                        cr = check_collision_scene(top1, has_preds=hp_i)
                        scene_collision_fracs.append(cr)

        preds = np.array(all_preds)
        gt = np.array(all_gt)
        scores = np.array(all_scores)
        M = preds.shape[1]

        from evaluate_all_models import (
            compute_min_ade_k, compute_min_fde_k, compute_miss_rate,
            compute_apd, compute_nll
        )

        r = {
            "Samples": len(preds),
            "minADE@1": float(compute_min_ade_k(preds, gt, k=1)),
            "minADE@4": float(compute_min_ade_k(preds, gt, k=min(4, M))),
            "minFDE@1": float(compute_min_fde_k(preds, gt, k=1)),
            "minFDE@4": float(compute_min_fde_k(preds, gt, k=min(4, M))),
            "MR@10px": float(compute_miss_rate(preds, gt, threshold=10.0)),
            "MR@20px": float(compute_miss_rate(preds, gt, threshold=20.0)),
            "APD": float(compute_apd(preds, k=M)),
            "NLL": float(compute_nll(preds, gt, scores=scores)),
            "CR": float(np.mean(scene_collision_fracs)) if scene_collision_fracs else 0.0,
            "ORR": 0.0,
        }
        results.append(r)
        print(f"  Samples={r['Samples']}, minADE@1={r['minADE@1']:.2f}, CR={r['CR']*100:.2f}%")

    sys.path.remove(str(WORKSPACE / "PrayagLaneGCN"))
    _clear_modules()
    return results


def eval_gameformer():
    """Evaluate PrayagGameFormer for all 3 seeds, compute CR with center-distance."""
    _clear_modules()
    sys.path.insert(0, str(WORKSPACE / "LaneGameFormer"))
    from utils.metrics import check_collision_scene
    sys.path.remove(str(WORKSPACE / "LaneGameFormer"))
    _clear_modules()

    sys.path.insert(0, str(WORKSPACE / "PrayagGameFormer"))
    from config import getConfig
    from model.gameformer import GameFormer
    from data.dataset import PrayagDataset, collateFn
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    config = getConfig("10Hz")
    config["datasetDir"] = str(DATASET_PATH)
    config["maxAgents"] = 100
    config["testAnnotations"] = str(DATASET_PATH / "test" / "annotations")
    config["testVideos"] = str(DATASET_PATH / "test" / "videos")
    config["chunkList"] = config.get("chunkList", {})
    config["chunkList"]["test"] = str(DATASET_PATH / "test_chunks.txt")

    results = []
    for seed in SEEDS:
        ckpt_path = WORKSPACE / f"experiment_outputs/train/PrayagGameFormer_10Hz/seed_{seed}/best.pth"
        if not ckpt_path.exists():
            print(f"  [SKIP] GameFormer seed {seed}")
            continue
        print(f"\n--- PrayagGameFormer seed {seed} ---")

        ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
        saved_config = ckpt.get("config", {})
        for k in ["dim", "heads", "encoderLayers", "decoderLevels", "numModes",
                   "neighborsToPredict", "predHorizon", "obsHorizon"]:
            if k in saved_config:
                config[k] = saved_config[k]

        model = GameFormer(config).to(DEVICE)
        model.load_state_dict(ckpt["modelState"])
        model.eval()

        test_ds = PrayagDataset(config, split="test")
        test_loader = DataLoader(test_ds, batch_size=2, shuffle=False,
                                 num_workers=0, collate_fn=collateFn, pin_memory=True)

        decoder_levels = config.get("decoderLevels", 3)
        all_preds, all_gt, all_scores = [], [], []
        scene_collision_fracs = []

        def gpu(batch):
            out = {}
            for k, v in batch.items():
                if torch.is_tensor(v):
                    out[k] = v.to(DEVICE)
                else:
                    out[k] = v
            return out

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"GameFormer seed {seed}"):
                batch_g = gpu(batch)
                out = model(batch_g)
                final_key = f"level_{decoder_levels}_interactions"
                scores_key = f"level_{decoder_levels}_scores"
                interactions = out[final_key]   # (B, N, M, T, 4)
                sc = out[scores_key]            # (B, N, M)
                B = interactions.shape[0]
                N = interactions.shape[1]
                T = interactions.shape[3]

                gt_ego = batch["ego_future"].cpu()       # (B, T, 2)
                neigh_fut = batch.get("neighbors_future", None)
                if neigh_fut is not None:
                    neigh_fut = neigh_fut.cpu()  # (B, N_neigh, T, 2)

                for i in range(B):
                    # Scene-level collision
                    pred_all = interactions[i, :, :, :, :2]  # (N, M, T, 2)
                    top1_idx = sc[i].argmax(dim=-1)          # (N,)
                    top1 = torch.gather(
                        pred_all, 1,
                        top1_idx.view(N, 1, 1, 1).expand(N, 1, T, 2)
                    ).squeeze(1).cpu()  # (N, T, 2)

                    # has_preds: construct from gt (non-zero)
                    gt_scene = torch.zeros(N, T, 2)
                    gt_scene[0] = gt_ego[i]
                    if neigh_fut is not None and N > 1:
                        n_neigh = min(N - 1, neigh_fut.shape[1])
                        gt_scene[1:1+n_neigh] = neigh_fut[i, :n_neigh]
                    hp = (gt_scene.norm(dim=-1) > 1e-6)

                    if N >= 2:
                        cr = check_collision_scene(top1, has_preds=hp)
                        scene_collision_fracs.append(cr)

                    # Per-agent displacement metrics
                    p0 = interactions[i, 0, :, :, :2].detach().cpu().numpy()
                    s0 = sc[i, 0].detach().cpu().numpy()
                    g0 = gt_ego[i].numpy()
                    if np.linalg.norm(g0) > 1e-6:
                        all_preds.append(p0)
                        all_scores.append(s0)
                        all_gt.append(g0)

                    if neigh_fut is not None and N > 1:
                        for n in range(1, min(N, 1 + neigh_fut.shape[1])):
                            g_n = neigh_fut[i, n - 1].numpy()
                            if np.linalg.norm(g_n) < 1e-6:
                                continue
                            p_n = interactions[i, n, :, :, :2].detach().cpu().numpy()
                            s_n = sc[i, n].detach().cpu().numpy()
                            all_preds.append(p_n)
                            all_scores.append(s_n)
                            all_gt.append(g_n)

        preds = np.array(all_preds)
        gt = np.array(all_gt)
        scores = np.array(all_scores)
        M = preds.shape[1]

        from evaluate_all_models import (
            compute_min_ade_k, compute_min_fde_k, compute_miss_rate,
            compute_apd, compute_nll
        )

        r = {
            "Samples": len(preds),
            "minADE@1": float(compute_min_ade_k(preds, gt, k=1)),
            "minADE@4": float(compute_min_ade_k(preds, gt, k=min(4, M))),
            "minFDE@1": float(compute_min_fde_k(preds, gt, k=1)),
            "minFDE@4": float(compute_min_fde_k(preds, gt, k=min(4, M))),
            "MR@10px": float(compute_miss_rate(preds, gt, threshold=10.0)),
            "MR@20px": float(compute_miss_rate(preds, gt, threshold=20.0)),
            "APD": float(compute_apd(preds, k=M)),
            "NLL": float(compute_nll(preds, gt, scores=scores)),
            "CR": float(np.mean(scene_collision_fracs)) if scene_collision_fracs else 0.0,
            "ORR": 0.0,
        }
        results.append(r)
        print(f"  Samples={r['Samples']}, minADE@1={r['minADE@1']:.2f}, CR={r['CR']*100:.2f}%")

    sys.path.remove(str(WORKSPACE / "PrayagGameFormer"))
    _clear_modules()
    return results


def eval_pv3():
    """Evaluate PV3 (inherently all-agents)."""
    _clear_modules()
    sys.path.insert(0, str(WORKSPACE / "PrayagProjectv3"))
    import evaluate_system as pv3_eval

    metrics_agg = pv3_eval.evaluate_on_dataset(
        dataset_dir=str(DATASET_PATH),
        fps=10.0,
        split='test',
    )

    sys.path.remove(str(WORKSPACE / "PrayagProjectv3"))
    for m in [k for k in sys.modules
              if k.startswith(('evaluate_system', 'evaluation_metrics', 'road_mask',
                               'trajectory_predict', 'track_data', 'lane_',
                               'social_potential', 'game_theory', 'emerging_lane',
                               'config'))]:
        del sys.modules[m]

    n = len(metrics_agg.get("minADE@1", []))
    if n == 0:
        return None

    def _mean(key):
        v = metrics_agg.get(key, [])
        return float(np.mean(v)) if v else 0.0

    return {
        "Samples": n,
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


def print_comparison(all_results):
    """Print comparison table."""
    print("\n" + "=" * 120)
    print("ALL-AGENTS COMPARISON (10Hz, Fixed CR metric, threshold=10px)")
    print("=" * 120)

    metrics = ["Samples", "minADE@1", "minADE@4", "minFDE@1", "minFDE@4",
               "MR@10px", "MR@20px", "APD", "NLL", "CR", "ORR"]
    header = f"{'Model':<22}"
    for m in metrics:
        header += f" {m:>12}"
    print(header)
    print("-" * 120)

    for model_name, runs in all_results.items():
        if not runs:
            continue
        line = f"{model_name:<22}"
        for k in metrics:
            vals = np.array([r[k] for r in runs if k in r])
            if len(vals) == 0:
                line += f" {'N/A':>12}"
            elif k == "Samples":
                line += f" {vals.mean():>12.0f}"
            elif k.startswith("MR") or k in ("CR", "ORR"):
                if vals.std() > 0:
                    line += f" {vals.mean()*100:>6.2f}±{vals.std()*100:.2f}%"
                else:
                    line += f" {vals.mean()*100:>10.2f}%"
            else:
                if vals.std() > 0:
                    line += f" {vals.mean():>8.2f}±{vals.std():.2f}"
                else:
                    line += f" {vals.mean():>12.2f}"
        print(line)

    # Save JSON
    out_path = WORKSPACE / "experiment_outputs" / "eval" / "all_agents_fixed_cr_10hz.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({k: v for k, v in all_results.items()}, f, indent=2, default=float)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    all_results = {}

    print("=" * 70)
    print("FULL EVALUATION WITH FIXED COLLISION RATE (10Hz, all-agents)")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    # 1. PV3
    print("\n\n=== PV3 (PV4: All-Agents) ===")
    pv3_r = eval_pv3()
    if pv3_r:
        all_results["PV3 (PV4)"] = [pv3_r]
        print(f"  Samples={pv3_r['Samples']}, minADE@1={pv3_r['minADE@1']:.2f}, CR={pv3_r['CR']*100:.2f}%")

    # 2. LaneGCN
    print("\n\n=== PrayagLaneGCN (3 seeds) ===")
    lgcn_results = eval_lanegcn()
    if lgcn_results:
        all_results["PrayagLaneGCN"] = lgcn_results

    # 3. GameFormer
    print("\n\n=== PrayagGameFormer (3 seeds) ===")
    gf_results = eval_gameformer()
    if gf_results:
        all_results["PrayagGameFormer"] = gf_results

    # 4. LGF v6
    print("\n\n=== LGF v6 (3 seeds) ===")
    lgf_results = eval_lgf_v6()
    if lgf_results:
        all_results["LGF v6"] = lgf_results

    # Print comparison
    print_comparison(all_results)
