"""
Track Data Loader
Loads pre-computed tracking data from PrayagProjectv1.5
"""
import os
import json
import csv
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


class TrackDataLoader:
    """
    Loads tracking data from CSV/JSON files produced by the detection pipeline.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the track data loader.
        
        Args:
            data_path: Path to directory containing tracking data files
        """
        self.data_path = data_path
        self.tracks = {}  # track_id -> list of frame data
        self.frame_data = defaultdict(list)  # frame_id -> list of detections
        self.metadata = {}
        
        self._load_data()
    
    def _load_data(self):
        """Load tracking data from available files."""
        csv_path = None
        json_path = None
        
        if os.path.isfile(self.data_path):
            if self.data_path.endswith('.csv'):
                csv_path = self.data_path
            elif self.data_path.endswith('.json'):
                json_path = self.data_path
        else:
            # Find tracking files
            for filename in os.listdir(self.data_path):
                if filename.endswith('_tracks.csv'):
                    csv_path = os.path.join(self.data_path, filename)
                elif filename.endswith('_tracks.json'):
                    json_path = os.path.join(self.data_path, filename)
        
        # Prefer CSV for efficiency
        if csv_path and os.path.exists(csv_path):
            self._load_csv(csv_path)
        elif json_path and os.path.exists(json_path):
            self._load_json(json_path)
        else:
            raise FileNotFoundError(f"No tracking data found in {self.data_path}")
    
    def _load_csv(self, path: str):
        """Load tracking data from CSV file."""
        print(f"Loading tracking data from CSV: {path}")
        
        with open(path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                frame_id = int(row['frame_id'])
                track_id = int(row['track_id'])
                center_x = float(row['center_x'])
                center_y = float(row['center_y'])
                
                # Parse OBB corners if available
                obb_corners = []
                for i in range(1, 5):
                    try:
                        cx = float(row[f'obb_corner{i}_x'])
                        cy = float(row[f'obb_corner{i}_y'])
                        obb_corners.append((cx, cy))
                    except (KeyError, ValueError):
                        pass
                
                detection = {
                    'frame_id': frame_id,
                    'track_id': track_id,
                    'center_x': center_x,
                    'center_y': center_y,
                    'obb_corners': obb_corners if len(obb_corners) == 4 else None
                }
                
                # Store by track_id
                if track_id not in self.tracks:
                    self.tracks[track_id] = []
                self.tracks[track_id].append(detection)
                
                # Store by frame_id
                self.frame_data[frame_id].append(detection)
        
        # Sort track data by frame
        for track_id in self.tracks:
            self.tracks[track_id].sort(key=lambda x: x['frame_id'])
        
        self._compute_metadata()
        print(f"Loaded {len(self.tracks)} tracks across {len(self.frame_data)} frames")
    
    def _load_json(self, path: str):
        """Load tracking data from JSON file."""
        print(f"Loading tracking data from JSON: {path}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            # List of detections
            for item in data:
                frame_id = int(item.get('frame_id', item.get('frame', 0)))
                track_id = int(item.get('track_id', item.get('id', 0)))
                
                # Get center position
                center_x = float(item.get('center_x', item.get('x', 0)))
                center_y = float(item.get('center_y', item.get('y', 0)))
                
                detection = {
                    'frame_id': frame_id,
                    'track_id': track_id,
                    'center_x': center_x,
                    'center_y': center_y,
                    'obb_corners': item.get('obb_corners')
                }
                
                if track_id not in self.tracks:
                    self.tracks[track_id] = []
                self.tracks[track_id].append(detection)
                self.frame_data[frame_id].append(detection)
        
        elif isinstance(data, dict):
            # Dictionary with tracks or frames
            if 'tracks' in data:
                for track_id_str, track_data in data['tracks'].items():
                    track_id = int(track_id_str)
                    for frame_data in track_data:
                        detection = {
                            'frame_id': int(frame_data.get('frame_id', 0)),
                            'track_id': track_id,
                            'center_x': float(frame_data.get('center_x', 0)),
                            'center_y': float(frame_data.get('center_y', 0)),
                            'obb_corners': frame_data.get('obb_corners')
                        }
                        
                        if track_id not in self.tracks:
                            self.tracks[track_id] = []
                        self.tracks[track_id].append(detection)
                        self.frame_data[detection['frame_id']].append(detection)
        
        # Sort track data by frame
        for track_id in self.tracks:
            self.tracks[track_id].sort(key=lambda x: x['frame_id'])
        
        self._compute_metadata()
        print(f"Loaded {len(self.tracks)} tracks across {len(self.frame_data)} frames")
    
    def _compute_metadata(self):
        """Compute metadata about the loaded tracks."""
        if not self.tracks:
            return
        
        all_frames = list(self.frame_data.keys())
        
        self.metadata = {
            'num_tracks': len(self.tracks),
            'num_frames': len(self.frame_data),
            'min_frame': min(all_frames) if all_frames else 0,
            'max_frame': max(all_frames) if all_frames else 0,
            'track_lengths': {
                tid: len(data) for tid, data in self.tracks.items()
            }
        }
    
    def get_track(self, track_id: int) -> List[Dict]:
        """Get all detections for a specific track."""
        return self.tracks.get(track_id, [])
    
    def get_track_positions(self, track_id: int) -> List[Tuple[float, float]]:
        """Get position history for a specific track."""
        track = self.tracks.get(track_id, [])
        return [(d['center_x'], d['center_y']) for d in track]
    
    def get_frame_detections(self, frame_id: int) -> List[Dict]:
        """Get all detections for a specific frame."""
        return self.frame_data.get(frame_id, [])
    
    def get_track_at_frame(self, track_id: int, frame_id: int) -> Optional[Dict]:
        """Get detection for a specific track at a specific frame."""
        track = self.tracks.get(track_id, [])
        for detection in track:
            if detection['frame_id'] == frame_id:
                return detection
        return None
    
    def get_track_history(self, track_id: int, current_frame: int, 
                          num_frames: int) -> List[Tuple[float, float]]:
        """
        Get position history for a track up to current frame.
        
        Args:
            track_id: Track identifier
            current_frame: Current frame number
            num_frames: Number of past frames to retrieve
            
        Returns:
            List of (x, y) positions
        """
        track = self.tracks.get(track_id, [])
        history = []
        
        for detection in track:
            if detection['frame_id'] <= current_frame:
                history.append((detection['center_x'], detection['center_y']))
        
        # Return last num_frames positions
        return history[-num_frames:] if len(history) > num_frames else history
    
    def get_valid_tracks(self, min_length: int = 10, 
                         min_duration_frames: int = None) -> List[int]:
        """
        Get list of track IDs that meet minimum requirements.
        
        Args:
            min_length: Minimum number of detections
            min_duration_frames: Minimum frame span
            
        Returns:
            List of valid track IDs
        """
        valid_tracks = []
        
        for track_id, track_data in self.tracks.items():
            if len(track_data) < min_length:
                continue
            
            if min_duration_frames:
                frames = [d['frame_id'] for d in track_data]
                duration = max(frames) - min(frames)
                if duration < min_duration_frames:
                    continue
            
            valid_tracks.append(track_id)
        
        return valid_tracks
    
    def get_active_tracks_at_frame(self, frame_id: int) -> List[int]:
        """Get list of track IDs that have detections at a specific frame."""
        return [d['track_id'] for d in self.frame_data.get(frame_id, [])]
    
    def extract_lane_trajectories(self, min_duration: int = 60,
                                   min_displacement: float = 100.0) -> List[List[Tuple[float, float]]]:
        """
        Extract complete vehicle trajectories suitable for lane extraction.
        
        Args:
            min_duration: Minimum track duration in frames
            min_displacement: Minimum total displacement in pixels
            
        Returns:
            List of trajectory point lists
        """
        trajectories = []
        
        for track_id, track_data in self.tracks.items():
            if len(track_data) < min_duration:
                continue
            
            positions = [(d['center_x'], d['center_y']) for d in track_data]
            
            # Check displacement
            if len(positions) < 2:
                continue
            
            start = np.array(positions[0])
            end = np.array(positions[-1])
            displacement = np.linalg.norm(end - start)
            
            if displacement >= min_displacement:
                trajectories.append(positions)
        
        return trajectories
