"""
System Evaluation Script
Runs the trajectory prediction system on a test set and computes quantitative metrics.
"""
import os
import argparse
import glob
import numpy as np
from tqdm import tqdm
import config
from track_data_loader import TrackDataLoader
from road_mask_manager import RoadMaskManager
from lane_manager import LaneManager
from social_potential_field import SocialPotentialField
from game_theory_predictor import GameTheoryPredictor
from trajectory_predictor import TrajectoryPredictor
from evaluation_metrics import TrajectoryEvaluator
from emerging_lane_extractor import EmergingLaneExtractor

def evaluate_scenario(track_path, road_mask_path, fps=config.DEFAULT_FPS, lane_track_path=None, lane_fps=None):
    """
    Evaluate a scenario.
    
    Args:
        track_path: Path to tracking CSV for evaluation
        road_mask_path: Path to road mask JSON
        fps: FPS of the evaluation dataset
        lane_track_path: Optional separate track path for lane extraction (for cross-dataset evaluation)
        lane_fps: Optional FPS for lane extraction dataset
    """
    print(f"Evaluating: {os.path.basename(track_path)}")
    
    # 1. Load Data
    track_loader = TrackDataLoader(track_path)
    
    # Hardcoded resolution for DJI dataset
    FRAME_WIDTH = 1920
    FRAME_HEIGHT = 1080
    road_mask = RoadMaskManager(road_mask_path, FRAME_WIDTH, FRAME_HEIGHT)
    
    # 2. Initialize Components
    lane_manager = LaneManager()
    
    # Extract lanes - use separate dataset if provided
    if lane_track_path and lane_fps:
        print(f"  Using lanes from: {os.path.basename(lane_track_path)} at {lane_fps}Hz")
        lane_track_loader = TrackDataLoader(lane_track_path)
        lane_extractor = EmergingLaneExtractor(lane_track_loader, lane_fps)
    else:
        lane_extractor = EmergingLaneExtractor(track_loader, fps)
    
    trajectories = track_loader.extract_lane_trajectories()
    lanes_data = lane_extractor.extract_lanes()
    lanes_dicts = lanes_data['lanes']
    connections = lanes_data['connections']
    
    # Extract just the points for lane manager
    lane_points_list = [l['points'] for l in lanes_dicts]
    lane_manager.add_lanes_from_extractor(lane_points_list, connections)
    
    social_field = SocialPotentialField(FRAME_WIDTH, FRAME_HEIGHT)
    game_theory = GameTheoryPredictor()
    
    # Initialize Predictor
    predictor = TrajectoryPredictor(
        road_mask, lane_manager, social_field, game_theory, fps
    )
    
    evaluator = TrajectoryEvaluator(road_mask)

    # 3. Define Test Set
    history_frames = int(config.PAST_HISTORY_SECONDS * fps)
    future_frames = int(config.FUTURE_PREDICTION_SECONDS * fps)
    
    test_cases = []
    
    # Sample every 50th frame (adjusted for FPS if needed, but keeping simple step for now)
    # If FPS is lower (10 vs 30), 50 frames is 5s vs 1.6s. 
    # Let's adjust step to be roughly 1.5-2 seconds.
    frame_step = int(1.5 * fps)
    
    start_frame = track_loader.metadata['min_frame'] + history_frames
    end_frame = track_loader.metadata['max_frame'] - future_frames
    
    if start_frame >= end_frame:
        print("  Skipping: Not enough frames for history+future window.")
        return {}

    for frame_idx in range(start_frame, end_frame, frame_step):
        active_tracks = track_loader.get_active_tracks_at_frame(frame_idx)
        
        for track_id in active_tracks:
            track_data = track_loader.get_track(track_id)
            
            curr_idx = -1
            for i, d in enumerate(track_data):
                if d['frame_id'] == frame_idx:
                    curr_idx = i
                    break
            
            if curr_idx == -1: continue
            if curr_idx < history_frames: continue
            if curr_idx + future_frames >= len(track_data): continue
            
            test_cases.append({
                'frame_id': frame_idx,
                'track_id': track_id,
                'curr_idx': curr_idx
            })
            
    print(f"  Found {len(test_cases)} valid test cases.")
    
    # 4. Run Evaluation Loop
    metrics_agg = {
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
    
    for case in tqdm(test_cases, desc="Evaluating frames"):
        frame_id = case['frame_id']
        ego_id = case['track_id']
        curr_idx = case['curr_idx']
        
        full_track = track_loader.get_track(ego_id)
        history_data = full_track[curr_idx - history_frames : curr_idx + 1]
        ego_history = [(d['center_x'], d['center_y']) for d in history_data]
        
        future_data = full_track[curr_idx + 1 : curr_idx + 1 + future_frames]
        gt_future = [(d['center_x'], d['center_y']) for d in future_data]
        
        all_vehicles_history = {}
        other_agents_futures = []
        
        active_others = track_loader.get_active_tracks_at_frame(frame_id)
        for other_id in active_others:
            if other_id == ego_id: continue
            
            other_track = track_loader.get_track(other_id)
            
            o_curr_idx = -1
            for i, d in enumerate(other_track):
                if d['frame_id'] == frame_id:
                    o_curr_idx = i
                    break
            
            if o_curr_idx != -1:
                start_idx = max(0, o_curr_idx - history_frames)
                o_hist = other_track[start_idx : o_curr_idx + 1]
                all_vehicles_history[other_id] = [(d['center_x'], d['center_y']) for d in o_hist]
                
                o_fut = other_track[o_curr_idx + 1 : o_curr_idx + 1 + future_frames]
                if o_fut:
                    other_agents_futures.append([(d['center_x'], d['center_y']) for d in o_fut])

        all_vehicles_history[ego_id] = ego_history
        
        predictions = predictor.predict_trajectories(
            ego_id=ego_id,
            ego_history=ego_history,
            all_vehicles=all_vehicles_history,
            time_horizon=config.FUTURE_PREDICTION_SECONDS
        )
        
        res = evaluator.evaluate_batch(predictions, gt_future, other_agents_futures, k_list=[1, 4])
        
        if res:
            for k in metrics_agg.keys():
                if k in res:
                    metrics_agg[k].append(res[k])

    return metrics_agg

def main():
    parser = argparse.ArgumentParser(description='Evaluate Trajectory Prediction System')
    parser.add_argument('--split', type=str, choices=['val', 'test'], help='Dataset split to evaluate')
    parser.add_argument('--dataset', type=str, default="ChunkedProjectPrayagBEVDataset", help='Dataset directory name')
    parser.add_argument('--fps', type=float, default=config.DEFAULT_FPS, help='Frames per second of the dataset')
    parser.add_argument('--lane-dataset', type=str, default=None, help='Dataset to use for lane extraction (cross-dataset mode)')
    parser.add_argument('--lane-fps', type=float, default=None, help='FPS of the lane extraction dataset')
    args = parser.parse_args()

    if args.split:
        base_dir = os.path.join(args.dataset, args.split, "annotations")
        track_files = glob.glob(os.path.join(base_dir, "*_tracks.csv"))
        
        # If using separate lane dataset, prepare mapping
        lane_base_dir = None
        if args.lane_dataset:
            lane_base_dir = os.path.join(args.lane_dataset, args.split, "annotations")
            print(f"Cross-dataset mode: Lanes from {args.lane_dataset} at {args.lane_fps}Hz")
        
        print(f"Found {len(track_files)} track files in {base_dir}")
        
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
            # Find corresponding road mask
            # Assuming naming convention: DJI_0910_chunk_10_tracks.csv -> DJI_0910_chunk_10_road_annotation.json
            road_mask_path = track_path.replace("_tracks.csv", "_road_annotation.json")
            
            if not os.path.exists(road_mask_path):
                print(f"Warning: Road mask not found for {track_path}. Skipping.")
                continue
            
            # Prepare lane extraction paths if cross-dataset mode
            lane_track_path = None
            lane_fps = None
            if lane_base_dir:
                # Map the track file to the corresponding lane dataset file
                track_basename = os.path.basename(track_path)
                lane_track_path = os.path.join(lane_base_dir, track_basename)
                if os.path.exists(lane_track_path):
                    lane_fps = args.lane_fps
                else:
                    print(f"  Warning: Lane track not found: {lane_track_path}. Using same dataset.")
                    lane_track_path = None
                
            metrics = evaluate_scenario(track_path, road_mask_path, fps=args.fps, 
                                        lane_track_path=lane_track_path, lane_fps=lane_fps)
            
            for k, v in metrics.items():
                if k in all_metrics:
                    all_metrics[k].extend(v)
        
        print("\n" + "="*60)
        mode_str = ""
        if args.lane_dataset:
            mode_str = f" [Lanes from {args.lane_dataset}@{args.lane_fps}Hz]"
        print(f"FINAL RESULTS FOR {args.split.upper()} SPLIT ({args.dataset}){mode_str}")
        print("="*60)
        if all_metrics['minADE@4']:
            minADE4 = np.mean(all_metrics['minADE@4'])
            minFDE4 = np.mean(all_metrics['minFDE@4'])
            collision_rate = np.mean(all_metrics['collision'])
            off_road_rate = np.mean(all_metrics['off_road'])
            # Model Selection Score: MSS = minADE + minFDE + 10.0 * CR + 10.0 * ORR
            mss = minADE4 + minFDE4 + 10.0 * collision_rate + 10.0 * off_road_rate
            
            print(f"Total Samples: {len(all_metrics['minADE@4'])}")
            print(f"minADE@1:       {np.mean(all_metrics['minADE@1']):.2f} pixels")
            print(f"minADE@4:       {minADE4:.2f} pixels")
            print(f"minFDE@1:       {np.mean(all_metrics['minFDE@1']):.2f} pixels")
            print(f"minFDE@4:       {minFDE4:.2f} pixels")
            print(f"Miss Rate @10px: {np.mean(all_metrics['miss_rate_10']) * 100:.2f}%")
            print(f"Miss Rate @20px: {np.mean(all_metrics['miss_rate_20']) * 100:.2f}%")
            print(f"Norm FDE:       {np.mean(all_metrics['norm_fde']):.4f}")
            print(f"APD (Diversity):{np.mean(all_metrics['apd']):.2f} pixels")
            print(f"NLL:            {np.mean(all_metrics['nll']):.4f}")
            print(f"Collision Rate: {collision_rate * 100:.2f}%")
            print(f"Off-Road Rate:  {off_road_rate * 100:.2f}%")
            print(f"MSS (Model Selection Score): {mss:.2f}")
        else:
            print("No valid samples evaluated.")
        print("="*60)
        
    else:
        # Default behavior
        metrics = evaluate_scenario(config.TRACKING_DATA_PATH, config.ROAD_MASK_PATH, fps=args.fps)
        
        print("\n[5/5] Results:")
        print("------------------------------------------------------------")
        if metrics['minADE@4']:
            minADE4 = np.mean(metrics['minADE@4'])
            minFDE4 = np.mean(metrics['minFDE@4'])
            collision_rate = np.mean(metrics['collision'])
            off_road_rate = np.mean(metrics['off_road'])
            # Model Selection Score: MSS = minADE + minFDE + 10.0 * CR + 10.0 * ORR
            mss = minADE4 + minFDE4 + 10.0 * collision_rate + 10.0 * off_road_rate
            
            print(f"Samples: {len(metrics['minADE@4'])}")
            print(f"minADE@1:       {np.mean(metrics['minADE@1']):.2f} pixels")
            print(f"minADE@4:       {minADE4:.2f} pixels")
            print(f"minFDE@1:       {np.mean(metrics['minFDE@1']):.2f} pixels")
            print(f"minFDE@4:       {minFDE4:.2f} pixels")
            print(f"Miss Rate @10px: {np.mean(metrics['miss_rate_10']) * 100:.2f}%")
            print(f"Miss Rate @20px: {np.mean(metrics['miss_rate_20']) * 100:.2f}%")
            print(f"Norm FDE:       {np.mean(metrics['norm_fde']):.4f}")
            print(f"APD (Diversity):{np.mean(metrics['apd']):.2f} pixels")
            print(f"NLL:            {np.mean(metrics['nll']):.4f}")
            print(f"Collision Rate: {collision_rate * 100:.2f}%")
            print(f"Off-Road Rate:  {off_road_rate * 100:.2f}%")
            print(f"MSS (Model Selection Score): {mss:.2f}")
        else:
            print("No valid samples evaluated.")
        print("------------------------------------------------------------")

if __name__ == "__main__":
    main()
