"""
Emerging Lane Extractor
Extracts lane patterns from tracking data (uses pre-computed detections)
"""
import os
import json
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import directed_hausdorff
from scipy.signal import savgol_filter
from scipy.interpolate import splprep, splev
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points
import geojson as geojson_lib

import config
from track_data_loader import TrackDataLoader


class EmergingLaneExtractor:
    """
    Extracts emerging lane patterns from vehicle trajectories.
    Uses pre-computed tracking data instead of running detection models.
    """
    
    def __init__(self, track_loader: TrackDataLoader, fps: float):
        """
        Initialize the lane extractor.
        
        Args:
            track_loader: TrackDataLoader instance with loaded tracking data
            fps: Video frame rate
        """
        self.track_loader = track_loader
        self.fps = fps
        self.lanes = []
        self.lane_connections = []
    
    def extract_lanes(self) -> dict:
        """
        Extract emerging lanes from tracking data.
        
        Returns:
            Dictionary containing lanes and connections
        """
        print("Extracting emerging lanes from tracking data...")
        
        # Get valid trajectories
        min_frames = int(config.MIN_TRACK_DURATION_SECONDS * self.fps)
        trajectories = self.track_loader.extract_lane_trajectories(
            min_duration=min_frames,
            min_displacement=80.0
        )
        
        print(f"Found {len(trajectories)} valid trajectories for lane extraction")
        
        if not trajectories:
            print("No valid trajectories found!")
            return {'lanes': [], 'connections': []}
        
        # Smooth trajectories
        # Use Savitzky-Golay for initial denoising
        # Wrap in dict to track confidence counts
        smoothed_tracks = [{'points': self._smooth_track_savitzky_golay(t), 'count': 1} for t in trajectories]
        
        # Merge similar tracks
        merged_tracks = self._merge_similar_tracks(smoothed_tracks)
        print(f"After merging similar tracks: {len(merged_tracks)} unique lanes")
        
        # Fit B-Splines to merged tracks for final smooth geometry
        final_lanes = []
        for t in merged_tracks:
            fitted = self._fit_bspline_lane(t['points'], smoothing=100.0)
            final_lanes.append({'points': fitted, 'count': t['count']})
        
        # Additional tight merging pass (3 pixels) to combine very close lanes
        # This helps remove duplicate lanes that are virtually identical
        print("Performing tight merge (3.0px)...")
        refined_lanes = self._merge_similar_tracks(final_lanes, threshold=3.0)
        print(f"After tight merge: {len(refined_lanes)} unique lanes")
        
        # Calculate confidence scores
        if refined_lanes:
            max_count = max(t['count'] for t in refined_lanes)
            print(f"Max track support for a lane: {max_count}")
            for t in refined_lanes:
                # Logarithmic scoring to boost lower counts but reward high counts
                # score = 0.5 + 0.5 * (count / max_count)
                t['score'] = min(1.0, 0.3 + 0.7 * (t['count'] / max_count))
        
        # Snap endpoints
        snapped_tracks = self._snap_endpoints(refined_lanes)
        
        # Extract just the points for connection finding (legacy support)
        lane_points = [t['points'] for t in snapped_tracks]
        
        # Find lane connections
        connections = self._find_lane_connections(lane_points)
        print(f"Found {len(connections)} lane connections")
        
        # Store results
        self.lanes = snapped_tracks
        self.lane_connections = connections
        
        return {
            'lanes': snapped_tracks,
            'connections': connections
        }
    
    def _smooth_track_savitzky_golay(self, points: list, window_length: int = 11, polyorder: int = 2) -> list:
        """
        Apply Savitzky-Golay filter to smooth the track.
        Preserves geometric features better than moving average.
        """
        if len(points) < window_length:
            return points
        
        coords = np.array(points)
        try:
            # Apply filter to x and y coordinates independently
            x_smooth = savgol_filter(coords[:, 0], window_length, polyorder)
            y_smooth = savgol_filter(coords[:, 1], window_length, polyorder)
            return list(zip(x_smooth, y_smooth))
        except Exception as e:
            print(f"Error in Savitzky-Golay smoothing: {e}")
            return points

    def _fit_bspline_lane(self, points: list, smoothing: float = 5.0) -> list:
        """
        Fit a B-Spline to the lane points for C2 continuity.
        """
        if len(points) < 4:
            return points
            
        try:
            # Remove duplicates to avoid errors
            coords = np.array(points)
            _, idx = np.unique(coords, axis=0, return_index=True)
            coords = coords[np.sort(idx)]
            
            if len(coords) < 4:
                return points
                
            # Fit B-Spline
            tck, u = splprep([coords[:, 0], coords[:, 1]], s=smoothing, k=3)
            
            # Evaluate spline at regular intervals
            u_new = np.linspace(0, 1, num=max(20, len(points)))
            x_new, y_new = splev(u_new, tck)
            
            return list(zip(x_new, y_new))
        except Exception as e:
            print(f"Error in B-Spline fitting: {e}")
            return points

    def _smooth_polyline(self, points: list, window_size: int = None) -> list:
        """Apply moving average smoothing to a polyline."""
        if window_size is None:
            window_size = config.SMOOTHING_WINDOW_SIZE
        
        if len(points) < window_size:
            return points
        
        coords = np.array(points)
        smoothed = []
        
        for i in range(len(coords)):
            start = max(0, i - window_size // 2)
            end = min(len(coords), i + window_size // 2 + 1)
            avg_point = np.mean(coords[start:end], axis=0)
            smoothed.append(avg_point.tolist())
        
        return smoothed
    
    def _calculate_hausdorff_distance(self, line1: list, line2: list) -> float:
        """Calculate Hausdorff distance between two lines."""
        points1 = np.array(line1)
        points2 = np.array(line2)
        
        if len(points1) < 2 or len(points2) < 2:
            return float('inf')
        
        d1 = directed_hausdorff(points1, points2)[0]
        d2 = directed_hausdorff(points2, points1)[0]
        
        return max(d1, d2)
    
    def _merge_similar_tracks(self, tracks: list, 
                              threshold: float = None) -> list:
        """Merge tracks that are similar (based on Hausdorff distance)."""
        if threshold is None:
            threshold = config.HAUSDORFF_THRESHOLD
        
        if len(tracks) <= 1:
            return tracks
        
        merged = []
        used = set()
        
        for i, track1_data in enumerate(tracks):
            if i in used:
                continue
            
            # Handle both dict (with count) and list (raw points) inputs
            if isinstance(track1_data, dict):
                track1 = track1_data['points']
                total_count = track1_data.get('count', 1)
            else:
                track1 = track1_data
                total_count = 1
                
            similar_tracks = [track1]
            used.add(i)
            
            for j, track2_data in enumerate(tracks[i+1:], i+1):
                if j in used:
                    continue
                
                if isinstance(track2_data, dict):
                    track2 = track2_data['points']
                    count2 = track2_data.get('count', 1)
                else:
                    track2 = track2_data
                    count2 = 1
                
                if self._calculate_hausdorff_distance(track1, track2) < threshold:
                    similar_tracks.append(track2)
                    total_count += count2
                    used.add(j)
            
            if len(similar_tracks) == 1:
                merged.append({'points': track1, 'count': total_count})
            else:
                # Average similar tracks
                all_points = []
                for track in similar_tracks:
                    all_points.extend(track)
                merged.append({'points': self._smooth_polyline(all_points), 'count': total_count})
        
        return merged
    
    def _snap_endpoints(self, tracks: list, 
                        tolerance: float = None) -> list:
        """Snap nearby endpoints together."""
        if tolerance is None:
            tolerance = config.ENDPOINT_SNAP_TOLERANCE
        
        if len(tracks) <= 1:
            return tracks
        
        snapped_tracks = []
        
        for track_data in tracks:
            # Handle dict/list input
            if isinstance(track_data, dict):
                track = track_data['points']
                metadata = track_data
            else:
                track = track_data
                metadata = {'points': track}
                
            if len(track) < 2:
                continue
            
            start_point = Point(track[0])
            end_point = Point(track[-1])
            
            snapped_track = [p for p in track]  # Copy
            
            for other_data in tracks:
                if other_data is track_data:
                    continue
                    
                if isinstance(other_data, dict):
                    other_track = other_data['points']
                else:
                    other_track = other_data
                    
                if len(other_track) < 2:
                    continue
                
                other_start = Point(other_track[0])
                other_end = Point(other_track[-1])
                
                if start_point.distance(other_start) < tolerance:
                    snapped_track[0] = other_track[0]
                elif start_point.distance(other_end) < tolerance:
                    snapped_track[0] = other_track[-1]
                
                if end_point.distance(other_start) < tolerance:
                    snapped_track[-1] = other_track[0]
                elif end_point.distance(other_end) < tolerance:
                    snapped_track[-1] = other_track[-1]
            
            # Preserve metadata
            new_data = metadata.copy()
            new_data['points'] = snapped_track
            snapped_tracks.append(new_data)
        
        return snapped_tracks
    
    def _find_lane_connections(self, tracks: list) -> list:
        """Find connections between lane endpoints."""
        connections = []
        connection_threshold = 80.0
        
        for i, track1 in enumerate(tracks):
            if len(track1) < 5:
                continue
            
            for j, track2 in enumerate(tracks):
                if i == j or len(track2) < 5:
                    continue
                
                # Check end of track1 to start of track2
                end1 = np.array(track1[-1])
                start2 = np.array(track2[0])
                distance = np.linalg.norm(end1 - start2)
                
                if distance < connection_threshold and distance > 5:
                    # Get directions
                    dir1 = self._get_direction_at_point(track1, -1)
                    dir2 = self._get_direction_at_point(track2, 0)
                    
                    if dir1 is not None and dir2 is not None:
                        # Check angle compatibility
                        dot = np.dot(dir1, dir2)
                        if dot > 0.3:  # Roughly same direction
                            connections.append({
                                'start': track1[-1],
                                'end': track2[0],
                                'track1_idx': i,
                                'track2_idx': j,
                                'distance': distance,
                                'type': 'endpoint'
                            })
        
        return connections
    
    def _get_direction_at_point(self, track: list, idx: int, 
                                 window: int = 3) -> np.ndarray:
        """Get direction vector at a point in the track."""
        if len(track) < 2:
            return None
        
        if idx < 0:
            idx = len(track) + idx
        
        start_idx = max(0, idx - window)
        end_idx = min(len(track), idx + window + 1)
        
        if end_idx - start_idx < 2:
            return None
        
        start_point = np.array(track[start_idx])
        end_point = np.array(track[end_idx - 1])
        
        direction = end_point - start_point
        length = np.linalg.norm(direction)
        
        if length < 1e-6:
            return None
        
        return direction / length
    
    def export_geojson(self, output_path: str, video_info: dict) -> str:
        """
        Export lanes and connections to GeoJSON format.
        
        Args:
            output_path: Directory to save output
            video_info: Dictionary with video metadata
            
        Returns:
            Path to saved GeoJSON file
        """
        os.makedirs(output_path, exist_ok=True)
        
        features = []
        lane_counter = 0
        
        # Add lane features
        for i, lane_data in enumerate(self.lanes):
            # Handle dict/list input
            if isinstance(lane_data, dict):
                lane = lane_data['points']
                score = lane_data.get('score', 0.0)
                count = lane_data.get('count', 1)
            else:
                lane = lane_data
                score = 0.0
                count = 1
                
            if len(lane) < 2:
                continue
            
            try:
                line = LineString(lane)
                simplified = line.simplify(config.SIMPLIFY_TOLERANCE, preserve_topology=True)
                
                feature = geojson_lib.Feature(
                    geometry=geojson_lib.LineString(list(simplified.coords)),
                    properties={
                        "lane_id": lane_counter,
                        "feature_type": "lane",
                        "is_connection": False,
                        "length": simplified.length,
                        "point_count": len(lane),
                        "confidence_score": score,
                        "support_count": count
                    }
                )
                features.append(feature)
                lane_counter += 1
            except Exception as e:
                print(f"Error creating lane feature {i}: {e}")
        
        # Add connection features (bidirectional)
        for conn in self.lane_connections:
            try:
                points = [conn['start'], conn['end']]
                line = LineString(points)
                
                # Forward direction
                features.append(geojson_lib.Feature(
                    geometry=geojson_lib.LineString(list(line.coords)),
                    properties={
                        "lane_id": lane_counter,
                        "feature_type": "lane",
                        "is_connection": True,
                        "connection_type": conn['type'],
                        "length": line.length
                    }
                ))
                lane_counter += 1
                
                # Reverse direction
                reversed_line = LineString(list(line.coords)[::-1])
                features.append(geojson_lib.Feature(
                    geometry=geojson_lib.LineString(list(reversed_line.coords)),
                    properties={
                        "lane_id": lane_counter,
                        "feature_type": "lane",
                        "is_connection": True,
                        "connection_type": conn['type'],
                        "length": reversed_line.length
                    }
                ))
                lane_counter += 1
            except Exception as e:
                print(f"Error creating connection feature: {e}")
        
        # Create feature collection
        feature_collection = geojson_lib.FeatureCollection(
            features,
            properties={
                "video_width": video_info.get("width", 0),
                "video_height": video_info.get("height", 0),
                "fps": video_info.get("fps", 0),
                "total_frames": video_info.get("total_frames", 0),
                "total_lanes": lane_counter,
                "total_connections": len(self.lane_connections)
            }
        )
        
        # Save to file
        geojson_path = os.path.join(output_path, "emerging_lanes.geojson")
        with open(geojson_path, 'w') as f:
            json.dump(feature_collection, f, indent=2)
        
        print(f"Exported {lane_counter} lane features to {geojson_path}")
        return geojson_path
