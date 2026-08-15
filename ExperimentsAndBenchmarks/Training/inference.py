#!/usr/bin/env python3
"""
inference.py - Video-based trajectory prediction using LaneGCN

This script processes video files to predict vehicle trajectories using a trained LaneGCN model.
It uses lane information from a GeoJSON file and detects/tracks vehicles in the video to make
lane-aware trajectory predictions.

Usage:
    python inference.py --video_path video.mp4 --geojson_path lanes.geojson --checkpoint model.ckpt --output_dir results
"""

import argparse
import cv2
import numpy as np
import os
import json
import sys
import torch
import torch.nn.functional as F
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tqdm import tqdm
import math
import colorsys

# Import model components
from src.model.lanegcn import LaneGCN
from utils import gpu, to_long, Optimizer, StepLR

try:
    import geojson
except ImportError:
    print("Installing geojson...")
    os.system("pip install geojson")
    import geojson

def parse_args():
    parser = argparse.ArgumentParser(description='Video-based trajectory prediction using LaneGCN')
    parser.add_argument('--video_path', type=str, required=True,
                        help='Path to input video file')
    parser.add_argument('--geojson_path', type=str, required=True,
                        help='Path to lane GeoJSON file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to LaneGCN checkpoint')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save output files')
    parser.add_argument('--yolo_model', type=str,
                        default=r"C:\Users\Xeron\Videos\PrayagIntersection\yolo11m-obb.pt",
                        help='Path to YOLO model weights')
    parser.add_argument('--history_length', type=int, default=20,
                        help='Number of history frames to use for prediction')
    parser.add_argument('--prediction_horizon', type=int, default=30,
                        help='Number of future frames to predict')
    parser.add_argument('--conf_threshold', type=float, default=0.3,
                        help='Confidence threshold for vehicle detection')
    parser.add_argument('--save_video', action='store_true',
                        help='Save output video with visualizations')
    parser.add_argument('--fps', type=int, default=30,
                        help='Output video FPS')
    return parser.parse_args()

class LaneManager:
    """Manages lane information from GeoJSON"""
    def __init__(self, geojson_path):
        self.lanes = []
        self.load_lanes(geojson_path)
    
    def load_lanes(self, geojson_path):
        """Load lanes from GeoJSON file"""
        try:
            with open(geojson_path, 'r') as f:
                data = json.load(f)
            
            if 'features' in data:
                for feature in data['features']:
                    if feature['geometry']['type'] == 'LineString':
                        coords = np.array(feature['geometry']['coordinates'])
                        self.lanes.append({
                            'coords': coords,
                            'id': feature.get('properties', {}).get('lane_id', len(self.lanes))
                        })
            
            print(f"Loaded {len(self.lanes)} lanes from {geojson_path}")
        except Exception as e:
            print(f"Error loading lanes: {e}")
            self.lanes = []
    
    def get_nearest_lane(self, point, max_distance=50):
        """Find the nearest lane to a given point"""
        if not self.lanes:
            return None
        
        min_distance = float('inf')
        nearest_lane = None
        
        for lane in self.lanes:
            coords = lane['coords']
            for i in range(len(coords) - 1):
                # Calculate distance from point to line segment
                dist = self.point_to_line_distance(point, coords[i], coords[i+1])
                if dist < min_distance:
                    min_distance = dist
                    nearest_lane = lane
        
        return nearest_lane if min_distance <= max_distance else None
    
    def point_to_line_distance(self, point, line_start, line_end):
        """Calculate distance from point to line segment"""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Vector from line_start to line_end
        dx = x2 - x1
        dy = y2 - y1
        
        # If the line segment is a point
        if dx == 0 and dy == 0:
            return np.sqrt((x0 - x1)**2 + (y0 - y1)**2)
        
        # Parameter t for the projection
        t = max(0, min(1, ((x0 - x1) * dx + (y0 - y1) * dy) / (dx**2 + dy**2)))
        
        # Projection point
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        
        # Distance from point to projection
        return np.sqrt((x0 - proj_x)**2 + (y0 - proj_y)**2)

class VehicleTracker:
    """Tracks vehicles and maintains their trajectories"""
    def __init__(self, model_path, history_length=20):
        self.model = YOLO(model_path)
        self.history_length = history_length
        self.tracks = defaultdict(lambda: deque(maxlen=history_length))
        self.track_colors = {}
        self.last_positions = {}
        
    def detect_and_track(self, frame):
        """Detect and track vehicles in frame"""
        results = self.model.track(
            frame,
            persist=True,
            conf=0.3,
            iou=0.5,
            classes=[2, 3, 5, 7],  # car, motorcycle, bus, truck
            tracker="bytetrack.yaml"
        )
        
        current_tracks = {}
        
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    if hasattr(box, 'id') and box.id is not None:
                        track_id = int(box.id[0])
                        
                        # Extract center point
                        if hasattr(box, 'xyxy'):
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            center = [(x1 + x2) / 2, (y1 + y2) / 2]
                        else:
                            continue
                        
                        # Update track history
                        self.tracks[track_id].append(center)
                        self.last_positions[track_id] = center
                        current_tracks[track_id] = {
                            'center': center,
                            'box': box,
                            'history': list(self.tracks[track_id])
                        }
                        
                        # Assign color if new track
                        if track_id not in self.track_colors:
                            self.track_colors[track_id] = self.generate_color(track_id)
        
        return current_tracks
    
    def generate_color(self, track_id):
        """Generate unique color for track"""
        hue = (track_id * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
        return (int(r * 255), int(g * 255), int(b * 255))

class LaneGCNInference:
    """LaneGCN model inference wrapper"""
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device
        self.model = self.load_model(checkpoint_path)
        
    def load_model(self, checkpoint_path):
        """Load LaneGCN model from checkpoint"""
        # Model configuration (should match training config)
        config = {
            'n_actor': 64,  # Reduced from 128 for memory
            'n_map': 64,    # Reduced from 128 for memory
            'actor_heads': 6,
            'map_heads': 6,
            'num_mods': 6,
            'num_preds': 1,
            'pred_size': 30,
            'pred_step': 1,
            'num_scales': 6,
            'n_actor_layers': 3,
            'n_map_layers': 3,
            'sync_dist': True,
            'dist_th': 100.0,
            'cross_dist': 6,
            'cross_angle': 0.5 * np.pi,
            'rot': True,
            'norm': "GN",
            'ngpus': 1,
            'batch_size': 8
        }
        
        # Initialize model
        model = LaneGCN(config)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        print(f"Loaded LaneGCN model from {checkpoint_path}")
        return model
    
    def prepare_input(self, track_history, lanes, video_width, video_height):
        """Prepare input data for LaneGCN"""
        if len(track_history) < 2:
            return None
        
        # Convert to numpy array and normalize
        trajectory = np.array(track_history)
        
        # Normalize coordinates to [-1, 1]
        trajectory[:, 0] = (trajectory[:, 0] / video_width) * 2 - 1
        trajectory[:, 1] = (trajectory[:, 1] / video_height) * 2 - 1
        
        # Pad or truncate to fixed length (20 frames)
        target_length = 20
        if len(trajectory) > target_length:
            trajectory = trajectory[-target_length:]
        else:
            # Pad with the first point
            padding_length = target_length - len(trajectory)
            first_point = trajectory[0:1]
            padding = np.repeat(first_point, padding_length, axis=0)
            trajectory = np.vstack([padding, trajectory])
        
        # Create actor features
        actors = torch.zeros((1, 1, target_length, 3))  # batch, num_actors, time, features
        actors[0, 0, :, :2] = torch.from_numpy(trajectory).float()
        actors[0, 0, :, 2] = 1.0  # valid mask
        
        # Create simple lane graph (simplified)
        lane_coords = []
        for lane in lanes[:20]:  # Limit number of lanes
            coords = lane['coords']
            if len(coords) > 0:
                # Normalize lane coordinates
                norm_coords = coords.copy()
                norm_coords[:, 0] = (norm_coords[:, 0] / video_width) * 2 - 1
                norm_coords[:, 1] = (norm_coords[:, 1] / video_height) * 2 - 1
                lane_coords.extend(norm_coords)
        
        if not lane_coords:
            # Create dummy lane data
            lane_coords = [[-1, -1], [1, 1]]
        
        lane_coords = np.array(lane_coords)
        
        # Pad or truncate lane data
        max_lane_points = 100
        if len(lane_coords) > max_lane_points:
            lane_coords = lane_coords[:max_lane_points]
        else:
            padding_length = max_lane_points - len(lane_coords)
            last_point = lane_coords[-1:] if len(lane_coords) > 0 else np.array([[0, 0]])
            padding = np.repeat(last_point, padding_length, axis=0)
            lane_coords = np.vstack([lane_coords, padding])
        
        # Create graph features
        graph = {
            'ctrs': torch.from_numpy(lane_coords).float().unsqueeze(0),  # Centers
            'feats': torch.zeros((1, len(lane_coords), 2)),  # Features
            'turn': torch.zeros((1, len(lane_coords), 2)),   # Turn features
            'control': torch.zeros((1, len(lane_coords), 2)), # Control features
            'intersect': torch.zeros((1, len(lane_coords), 2)) # Intersection features
        }
        
        return {
            'actors': actors.to(self.device),
            'actor_idcs': torch.tensor([[0]]).to(self.device),
            'actor_ctrs': torch.zeros((1, 2)).to(self.device),
            'graph': {k: v.to(self.device) for k, v in graph.items()}
        }
    
    def predict(self, input_data):
        """Make trajectory prediction"""
        if input_data is None:
            return None
        
        try:
            with torch.no_grad():
                output = self.model(input_data)
                
                # Handle different output formats
                if isinstance(output, dict):
                    if 'reg' in output:
                        predictions = output['reg']
                    else:
                        # Take the first available output
                        predictions = list(output.values())[0]
                elif isinstance(output, (list, tuple)):
                    predictions = output[0]
                else:
                    predictions = output
                
                # Convert to numpy
                if isinstance(predictions, torch.Tensor):
                    predictions = predictions.cpu().numpy()
                
                return predictions
                
        except Exception as e:
            print(f"Prediction error: {e}")
            return None

class VideoInference:
    """Main video inference class"""
    def __init__(self, args):
        self.args = args
        self.lane_manager = LaneManager(args.geojson_path)
        self.vehicle_tracker = VehicleTracker(args.yolo_model, args.history_length)
        self.lanegcn = LaneGCNInference(args.checkpoint)
        self.predictions_data = []
        
        # Video properties
        cap = cv2.VideoCapture(args.video_path)
        self.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        print(f"Video: {self.video_width}x{self.video_height}, {self.video_fps:.2f} FPS, {self.total_frames} frames")
    
    def denormalize_prediction(self, prediction):
        """Convert normalized prediction back to pixel coordinates"""
        if prediction is None:
            return None
        
        pred = prediction.copy()
        pred[:, 0] = (pred[:, 0] + 1) / 2 * self.video_width
        pred[:, 1] = (pred[:, 1] + 1) / 2 * self.video_height
        return pred
    
    def draw_prediction(self, frame, track_id, prediction, color):
        """Draw prediction trajectory on frame"""
        if prediction is None or len(prediction) == 0:
            return
        
        # Draw prediction path
        for i in range(len(prediction) - 1):
            pt1 = (int(prediction[i][0]), int(prediction[i][1]))
            pt2 = (int(prediction[i+1][0]), int(prediction[i+1][1]))
            
            # Check bounds
            if (0 <= pt1[0] < self.video_width and 0 <= pt1[1] < self.video_height and
                0 <= pt2[0] < self.video_width and 0 <= pt2[1] < self.video_height):
                cv2.line(frame, pt1, pt2, color, 2)
        
        # Draw arrow at end
        if len(prediction) >= 2:
            end_point = prediction[-1]
            prev_point = prediction[-2]
            
            # Calculate arrow direction
            dx = end_point[0] - prev_point[0]
            dy = end_point[1] - prev_point[1]
            length = np.sqrt(dx**2 + dy**2)
            
            if length > 0:
                dx /= length
                dy /= length
                
                arrow_length = 10
                end_pt = (int(end_point[0]), int(end_point[1]))
                arrow_pt1 = (int(end_point[0] - arrow_length * (dx + dy * 0.5)),
                           int(end_point[1] - arrow_length * (dy - dx * 0.5)))
                arrow_pt2 = (int(end_point[0] - arrow_length * (dx - dy * 0.5)),
                           int(end_point[1] - arrow_length * (dy + dx * 0.5)))
                
                cv2.line(frame, end_pt, arrow_pt1, color, 2)
                cv2.line(frame, end_pt, arrow_pt2, color, 2)
    
    def draw_lanes(self, frame):
        """Draw lane lines on frame"""
        for lane in self.lane_manager.lanes:
            coords = lane['coords'].astype(int)
            for i in range(len(coords) - 1):
                pt1 = tuple(coords[i])
                pt2 = tuple(coords[i+1])
                cv2.line(frame, pt1, pt2, (100, 100, 100), 1)
    
    def process_video(self):
        """Process entire video and generate predictions"""
        cap = cv2.VideoCapture(self.args.video_path)
        
        # Setup output video if requested
        if self.args.save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_path = os.path.join(self.args.output_dir, 'predictions.mp4')
            out = cv2.VideoWriter(output_path, fourcc, self.args.fps, 
                                (self.video_width, self.video_height))
        
        frame_idx = 0
        
        with tqdm(total=self.total_frames, desc="Processing video") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Track vehicles
                tracks = self.vehicle_tracker.detect_and_track(frame)
                
                # Draw lanes
                self.draw_lanes(frame)
                
                frame_predictions = {}
                
                # Process each track
                for track_id, track_data in tracks.items():
                    history = track_data['history']
                    color = self.vehicle_tracker.track_colors[track_id]
                    
                    # Draw vehicle box
                    box = track_data['box']
                    if hasattr(box, 'xyxy'):
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f'ID:{track_id}', (x1, y1-5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # Draw history trail
                    if len(history) > 1:
                        for i in range(len(history) - 1):
                            pt1 = (int(history[i][0]), int(history[i][1]))
                            pt2 = (int(history[i+1][0]), int(history[i+1][1]))
                            cv2.line(frame, pt1, pt2, color, 1)
                    
                    # Make prediction if enough history
                    if len(history) >= 5:
                        input_data = self.lanegcn.prepare_input(
                            history, self.lane_manager.lanes, 
                            self.video_width, self.video_height
                        )
                        
                        prediction = self.lanegcn.predict(input_data)
                        
                        if prediction is not None and len(prediction.shape) >= 2:
                            # Handle different prediction shapes
                            if len(prediction.shape) == 3:
                                prediction = prediction[0]  # Take first batch
                            if len(prediction.shape) == 3:
                                prediction = prediction[0]  # Take first mode
                            
                            # Ensure we have the right number of points
                            if prediction.shape[0] != self.args.prediction_horizon:
                                prediction = prediction[:self.args.prediction_horizon]
                            
                            # Denormalize prediction
                            denorm_pred = self.denormalize_prediction(prediction)
                            
                            if denorm_pred is not None:
                                # Draw prediction
                                self.draw_prediction(frame, track_id, denorm_pred, color)
                                
                                # Store prediction data
                                frame_predictions[track_id] = {
                                    'history': history,
                                    'prediction': denorm_pred.tolist(),
                                    'current_position': track_data['center']
                                }
                
                # Store frame data
                self.predictions_data.append({
                    'frame': frame_idx,
                    'timestamp': frame_idx / self.video_fps,
                    'predictions': frame_predictions
                })
                
                # Save frame if requested
                if self.args.save_video:
                    out.write(frame)
                
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        if self.args.save_video:
            out.release()
            print(f"Output video saved to: {output_path}")
    
    def save_predictions(self):
        """Save predictions to JSON file"""
        output_path = os.path.join(self.args.output_dir, 'predictions.json')
        
        # Prepare summary statistics
        total_tracks = set()
        total_predictions = 0
        
        for frame_data in self.predictions_data:
            total_tracks.update(frame_data['predictions'].keys())
            total_predictions += len(frame_data['predictions'])
        
        output_data = {
            'video_info': {
                'path': self.args.video_path,
                'width': self.video_width,
                'height': self.video_height,
                'fps': self.video_fps,
                'total_frames': self.total_frames
            },
            'model_info': {
                'checkpoint': self.args.checkpoint,
                'history_length': self.args.history_length,
                'prediction_horizon': self.args.prediction_horizon
            },
            'statistics': {
                'total_tracks': len(total_tracks),
                'total_predictions': total_predictions,
                'frames_processed': len(self.predictions_data)
            },
            'predictions': self.predictions_data
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Predictions saved to: {output_path}")
        print(f"Statistics:")
        print(f"  - Total tracks: {len(total_tracks)}")
        print(f"  - Total predictions: {total_predictions}")
        print(f"  - Frames processed: {len(self.predictions_data)}")

def main():
    args = parse_args()
    
    # Validate inputs
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)
    
    if not os.path.exists(args.geojson_path):
        print(f"Error: GeoJSON file not found: {args.geojson_path}")
        sys.exit(1)
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run inference
    print("Starting video inference...")
    inference = VideoInference(args)
    inference.process_video()
    inference.save_predictions()
    
    print("Video inference complete!")

if __name__ == "__main__":
    main()
