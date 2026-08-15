import os
import cv2
import numpy as np
import config
from track_data_loader import TrackDataLoader
from road_mask_manager import RoadMaskManager
from emerging_lane_extractor import EmergingLaneExtractor
from lane_manager import LaneManager
from social_potential_field import SocialPotentialField
from game_theory_predictor import GameTheoryPredictor
from trajectory_predictor import TrajectoryPredictor
from visualizer import Visualizer
from collections import deque


def create_flow_field_visualization(road_mask, lanes, frame_width, frame_height):
    """
    Create a flow field visualization with:
    - Dark blue background
    - Light blue/cyan road mask
    - White lane polylines
    - Heatmap for emerging lane bands
    """
    # Create dark blue background
    dark_blue = np.array([128, 0, 0], dtype=np.uint8)  # BGR: dark blue
    result = np.full((frame_height, frame_width, 3), dark_blue, dtype=np.uint8)
    
    # Normalize road mask to 0-255
    mask_norm = (np.clip(road_mask, 0, 1.0) * 255).astype(np.uint8)
    
    # Apply JET colormap for the heatmap effect on the road/lane regions
    heatmap = cv2.applyColorMap(mask_norm, cv2.COLORMAP_JET)
    
    # Create alpha mask based on road mask intensity
    # Higher intensity = more visible heatmap
    alpha_mask = np.clip(road_mask, 0, 1.0).astype(np.float32)
    alpha_3ch = np.dstack([alpha_mask, alpha_mask, alpha_mask])
    
    # Blend heatmap onto dark blue background
    result = (result * (1 - alpha_3ch) + heatmap * alpha_3ch).astype(np.uint8)
    
    # Draw lane polylines in white
    for lane in lanes:
        coords = lane.get('coords', [])
        if len(coords) < 2:
            continue
        
        # Filter valid points
        valid_pts = []
        for x, y in coords:
            ix, iy = int(round(x)), int(round(y))
            if 0 <= ix < frame_width and 0 <= iy < frame_height:
                valid_pts.append((ix, iy))
        
        if len(valid_pts) < 2:
            continue
        
        pts_array = np.array(valid_pts, dtype=np.int32)
        
        # Draw in white with thin lines
        cv2.polylines(result, [pts_array], False, (255, 255, 255), 1, cv2.LINE_AA)
    
    return result


def generate_visualizations():
    # Paths for 10Hz dataset
    base_dir = "ChunkedProjectPrayagBEVDataset10Hz"
    video_path = os.path.join(base_dir, "train", "videos", "DJI_0912_chunk_0.mp4")
    track_path = os.path.join(base_dir, "train", "annotations", "DJI_0912_chunk_0_tracks.csv")
    road_mask_path = os.path.join(base_dir, "train", "annotations", "DJI_0912_chunk_0_road_annotation.json")
    
    output_dir = "visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    fps = 10.0
    target_frame = 100  # 10 seconds at 10Hz
    
    print(f"Processing {video_path}...")
    
    # 1. Load Video Info
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 2. Load Tracking Data
    print("Loading tracking data...")
    track_loader = TrackDataLoader(track_path)
    
    # 3. Load Road Mask
    print("Loading road mask...")
    road_mask_manager = RoadMaskManager(road_mask_path, frame_width, frame_height)
    
    # 4. Extract Lanes
    print("Extracting lanes...")
    lane_extractor = EmergingLaneExtractor(track_loader, fps)
    lane_data = lane_extractor.extract_lanes()
    
    lane_manager = LaneManager()
    raw_lanes = [l['points'] if isinstance(l, dict) else l for l in lane_data['lanes']]
    lane_manager.add_lanes_from_extractor(raw_lanes, lane_data['connections'])
    
    # Embed lanes into road mask
    for lane_obj in lane_data['lanes']:
        if isinstance(lane_obj, dict):
            points = lane_obj['points']
            score = lane_obj.get('score', 1.0)
        else:
            points = lane_obj
            score = 1.0
        road_mask_manager.embed_lane_spline(points, sigma=20.0, weight=0.6, confidence_score=score)
        
    # 5. Initialize Predictors
    social_field = SocialPotentialField(frame_width, frame_height)
    game_theory = GameTheoryPredictor()
    trajectory_predictor = TrajectoryPredictor(
        road_mask_manager, lane_manager, social_field, game_theory, fps
    )
    visualizer = Visualizer(frame_width, frame_height)
    
    # 6. Process up to target frame
    vehicle_histories = {}
    vehicle_predictions = {}
    history_length = int(config.PAST_HISTORY_SECONDS * fps)
    
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        
        # Update tracking state
        detections = track_loader.get_frame_detections(frame_num)
        current_ids = set()
        
        for det in detections:
            track_id = det['track_id']
            pos = (det['center_x'], det['center_y'])
            current_ids.add(track_id)
            
            if track_id not in vehicle_histories:
                vehicle_histories[track_id] = deque(maxlen=history_length)
            vehicle_histories[track_id].append(pos)
            
        # Cleanup old
        for vid in list(vehicle_histories.keys()):
            if vid not in current_ids:
                del vehicle_histories[vid]
                if vid in vehicle_predictions:
                    del vehicle_predictions[vid]
        
        # Generate predictions at target frame
        if frame_num == target_frame:
            print(f"Generating visualizations at frame {frame_num}...")
            
            # Generate predictions
            all_histories = {vid: list(h) for vid, h in vehicle_histories.items() if len(h) >= 2}
            
            # Extract OBBs
            all_obbs = []
            for det in detections:
                if det.get('obb_corners') and len(det.get('obb_corners', [])) == 4:
                    all_obbs.append({
                        'track_id': det['track_id'],
                        'obb_corners': det['obb_corners']
                    })

            for vid, history in all_histories.items():
                try:
                    preds = trajectory_predictor.predict_trajectories(
                        vid, history, all_histories, 
                        config.FUTURE_PREDICTION_SECONDS,
                        all_obbs=all_obbs
                    )
                    if preds:
                        vehicle_predictions[vid] = preds
                except Exception:
                    pass
            
            # Prepare vehicle states for visualization
            vehicle_states = []
            for vid, history in vehicle_histories.items():
                if len(history) >= 2:
                    kinematics = trajectory_predictor.compute_kinematics(list(history))
                    if kinematics:
                        vehicle_states.append({
                            'id': vid,
                            'position': kinematics['position'],
                            'velocity': kinematics['velocity'],
                            'direction': kinematics['direction'],
                            'speed': kinematics['speed'],
                            'history': list(history)
                        })
            
            # --- Generate Image 1: Traffic Flow Potential Surface ---
            # Create visualization with dark blue background, light blue road mask, 
            # white lane polylines, and heatmap for lane bands (matching 30Hz style)
            flow_viz = create_flow_field_visualization(
                road_mask_manager.road_mask, 
                lane_manager.lanes, 
                frame_width, 
                frame_height
            )
            cv2.imwrite(os.path.join(output_dir, "VisualizationTraffixFlowRegion10Hz.jpg"), flow_viz)
            print("Saved VisualizationTraffixFlowRegion10Hz.jpg")
            
            # --- Generate Image 2: Trajectory Prediction Output ---
            pred_viz = visualizer.create_composite_visualization(
                frame, road_mask_manager.road_mask,
                vehicle_states, vehicle_predictions,
                lane_manager.lanes, frame_num, detections=detections
            )
            cv2.imwrite(os.path.join(output_dir, "TrajectoryPredtionsImage10Hz.jpg"), pred_viz)
            print("Saved TrajectoryPredtionsImage10Hz.jpg")
            
            break
            
    cap.release()

if __name__ == "__main__":
    generate_visualizations()
