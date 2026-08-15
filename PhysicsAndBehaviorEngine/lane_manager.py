"""
Lane Manager
Manages emerging lane data and provides lane-based path planning
"""
import os
import json
import math
from typing import List, Tuple, Dict, Optional
import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points


class LaneManager:
    """
    Manages lane data and provides utilities for lane-based navigation.
    """
    
    def __init__(self, geojson_path: str = None):
        """
        Initialize the lane manager.
        
        Args:
            geojson_path: Path to GeoJSON file with lane data
        """
        self.lanes = []
        self.lane_connections = {}
        
        if geojson_path and os.path.exists(geojson_path):
            self.load_lanes(geojson_path)
    
    def load_lanes(self, geojson_path: str):
        """Load lanes from GeoJSON file."""
        try:
            with open(geojson_path, 'r') as f:
                data = json.load(f)
            
            for feature in data.get('features', []):
                props = feature.get('properties', {})
                
                # Only load lane features
                if props.get('feature_type') != 'lane':
                    continue
                
                coords = feature['geometry']['coordinates']
                if not coords or len(coords) < 2:
                    continue
                
                lane_id = props.get('lane_id', len(self.lanes))
                
                self.lanes.append({
                    'id': lane_id,
                    'coords': coords,
                    'linestring': LineString(coords),
                    'is_connection': props.get('is_connection', False),
                    'connection_type': props.get('connection_type', 'none'),
                    'length': props.get('length', LineString(coords).length)
                })
            
            print(f"Loaded {len(self.lanes)} lane segments")
            
        except Exception as e:
            print(f"Error loading lanes from {geojson_path}: {e}")
    
    def add_lanes_from_extractor(self, lanes: List[List], connections: List[Dict]):
        """
        Add lanes directly from lane extractor output.
        
        Args:
            lanes: List of lane point lists
            connections: List of connection dictionaries
        """
        lane_counter = len(self.lanes)
        
        # Add main lanes
        for lane_points in lanes:
            if len(lane_points) < 2:
                continue
            
            self.lanes.append({
                'id': lane_counter,
                'coords': lane_points,
                'linestring': LineString(lane_points),
                'is_connection': False,
                'connection_type': 'none',
                'length': LineString(lane_points).length
            })
            lane_counter += 1
        
        # Add connections (bidirectional)
        for conn in connections:
            points = [conn['start'], conn['end']]
            
            # Forward
            self.lanes.append({
                'id': lane_counter,
                'coords': points,
                'linestring': LineString(points),
                'is_connection': True,
                'connection_type': conn.get('type', 'endpoint'),
                'length': LineString(points).length
            })
            lane_counter += 1
            
            # Reverse
            self.lanes.append({
                'id': lane_counter,
                'coords': points[::-1],
                'linestring': LineString(points[::-1]),
                'is_connection': True,
                'connection_type': conn.get('type', 'endpoint'),
                'length': LineString(points[::-1]).length
            })
            lane_counter += 1
        
        print(f"Added lanes from extractor. Total: {len(self.lanes)} lanes")
    
    def get_lane_by_id(self, lane_id: int) -> Optional[Dict]:
        """Get lane by ID."""
        for lane in self.lanes:
            if lane['id'] == lane_id:
                return lane
        return None
    
    def get_lanes_within_distance(self, point: Tuple[float, float],
                                   max_distance: float = 60) -> List[Dict]:
        """
        Get lanes within a certain distance of a point.
        
        Args:
            point: (x, y) coordinates
            max_distance: Maximum distance threshold
            
        Returns:
            List of lane info dictionaries, sorted by distance
        """
        point_geom = Point(point)
        nearby_lanes = []
        
        for lane in self.lanes:
            try:
                distance = point_geom.distance(lane['linestring'])
                
                if distance <= max_distance:
                    closest_point_geom = nearest_points(point_geom, lane['linestring'])[1]
                    progress = lane['linestring'].project(point_geom)
                    
                    nearby_lanes.append({
                        'lane': lane,
                        'distance': distance,
                        'closest_point': [closest_point_geom.x, closest_point_geom.y],
                        'progress': progress
                    })
            except Exception:
                continue
        
        return sorted(nearby_lanes, key=lambda x: x['distance'])
    
    def get_lane_direction_at_progress(self, lane: Dict, 
                                        progress: float) -> Optional[List[float]]:
        """
        Get lane direction at a specific progress along the lane.
        
        Args:
            lane: Lane dictionary
            progress: Distance along the lane
            
        Returns:
            Normalized direction vector [dx, dy] or None
        """
        if not lane or progress is None:
            return None
        
        linestring = lane['linestring']
        if linestring.length == 0:
            return None
        
        # Get point at progress and slightly ahead
        point_at = linestring.interpolate(progress)
        look_ahead = min(progress + 5.0, linestring.length)
        
        if look_ahead <= progress:
            look_ahead = linestring.length
        
        point_ahead = linestring.interpolate(look_ahead)
        
        dx = point_ahead.x - point_at.x
        dy = point_ahead.y - point_at.y
        
        length = math.sqrt(dx*dx + dy*dy)
        if length > 0:
            return [dx/length, dy/length]
        
        # Fallback to last segment
        if len(lane['coords']) >= 2:
            dx = lane['coords'][-1][0] - lane['coords'][-2][0]
            dy = lane['coords'][-1][1] - lane['coords'][-2][1]
            length = math.sqrt(dx*dx + dy*dy)
            if length > 0:
                return [dx/length, dy/length]
        
        return None
    
    def get_path_along_lane(self, lane: Dict, start_progress: float,
                            distance: float, num_points: int = 10) -> List[Tuple[float, float]]:
        """
        Get a path along a lane starting from a progress point.
        
        Args:
            lane: Lane dictionary
            start_progress: Starting progress along lane
            distance: Distance to travel
            num_points: Number of points in path
            
        Returns:
            List of (x, y) points along the lane
        """
        if not lane or lane['linestring'].length <= start_progress:
            return []
        
        end_progress = min(start_progress + distance, lane['linestring'].length)
        
        if end_progress <= start_progress:
            return []
        
        # Calculate number of points based on distance
        effective_points = max(2, int(num_points * (distance / 50.0)))
        progress_step = (end_progress - start_progress) / max(1, effective_points - 1)
        
        path = []
        for i in range(effective_points):
            current_progress = min(start_progress + i * progress_step, 
                                   lane['linestring'].length)
            point = lane['linestring'].interpolate(current_progress)
            path.append((point.x, point.y))
            
            if current_progress >= lane['linestring'].length:
                break
        
        return path
    
    def find_continuation_lanes(self, lane: Dict, 
                                 at_end: bool = True) -> List[Dict]:
        """
        Find lanes that continue from the end (or start) of a lane.
        
        Args:
            lane: Current lane
            at_end: If True, find continuations at end; else at start
            
        Returns:
            List of continuation lane info dictionaries
        """
        if not lane:
            return []
        
        # Get reference point
        if at_end:
            ref_point = lane['coords'][-1]
            ref_dir = self.get_lane_direction_at_progress(lane, lane['linestring'].length)
        else:
            ref_point = lane['coords'][0]
            ref_dir = self.get_lane_direction_at_progress(lane, 0)
        
        if ref_dir is None:
            return []
        
        ref_dir = np.array(ref_dir)
        continuations = []
        
        for other_lane in self.lanes:
            if other_lane['id'] == lane['id']:
                continue
            
            # Check if other lane starts near our reference point
            start_dist = np.linalg.norm(np.array(other_lane['coords'][0]) - np.array(ref_point))
            
            if start_dist < 50:  # Within 50 pixels
                other_dir = self.get_lane_direction_at_progress(other_lane, 0)
                
                if other_dir:
                    other_dir = np.array(other_dir)
                    alignment = np.dot(ref_dir, other_dir)
                    
                    if alignment > 0.3:  # Reasonably aligned
                        continuations.append({
                            'lane': other_lane,
                            'distance': start_dist,
                            'alignment': alignment
                        })
        
        return sorted(continuations, key=lambda x: (-x['alignment'], x['distance']))
    
    def get_extended_path(self, start_lane: Dict, start_progress: float,
                          total_distance: float, 
                          num_points: int = 15) -> List[Tuple[float, float]]:
        """
        Get an extended path that may span multiple lanes.
        
        Args:
            start_lane: Starting lane
            start_progress: Starting progress on lane
            total_distance: Total distance to travel
            num_points: Target number of points
            
        Returns:
            List of (x, y) points
        """
        if not start_lane:
            return []
        
        path = []
        remaining_distance = total_distance
        current_lane = start_lane
        current_progress = start_progress
        
        while remaining_distance > 5 and current_lane:
            # Get path on current lane
            lane_remaining = current_lane['linestring'].length - current_progress
            path_distance = min(lane_remaining, remaining_distance)
            
            lane_path = self.get_path_along_lane(
                current_lane, current_progress, path_distance,
                num_points=int(num_points * path_distance / total_distance) + 2
            )
            
            if lane_path:
                # Add points (skip first if we already have points to avoid duplicates)
                if path:
                    path.extend(lane_path[1:])
                else:
                    path.extend(lane_path)
            
            remaining_distance -= path_distance
            
            if remaining_distance <= 5:
                break
            
            # Find continuation lane
            continuations = self.find_continuation_lanes(current_lane, at_end=True)
            
            if continuations:
                current_lane = continuations[0]['lane']
                current_progress = 0
            else:
                break
        
        return path
    
    def get_all_lane_paths(self, point: Tuple[float, float],
                            direction: Tuple[float, float],
                            distance: float,
                            max_paths: int = 6) -> List[Dict]:
        """
        Get all possible lane-based paths from a point.
        
        Args:
            point: Starting (x, y) position
            direction: Current direction vector
            distance: Distance to predict
            max_paths: Maximum number of paths to return
            
        Returns:
            List of path dictionaries with points and metadata
        """
        paths = []
        direction = np.array(direction)
        dir_norm = np.linalg.norm(direction)
        if dir_norm > 0:
            direction = direction / dir_norm
        
        # Get nearby lanes
        nearby = self.get_lanes_within_distance(point, max_distance=80)
        
        for lane_info in nearby[:max_paths * 2]:
            lane = lane_info['lane']
            progress = lane_info['progress']
            
            # Check direction alignment
            lane_dir = self.get_lane_direction_at_progress(lane, progress)
            if lane_dir is None:
                continue
            
            alignment = np.dot(direction, lane_dir)
            
            # Only consider lanes going roughly same direction
            if alignment < 0.3:
                continue
            
            # Get extended path
            path_points = self.get_extended_path(lane, progress, distance)
            
            if len(path_points) >= 2:
                paths.append({
                    'lane_id': lane['id'],
                    'path_points': path_points,
                    'alignment': alignment,
                    'distance_to_lane': lane_info['distance'],
                    'is_connection': lane.get('is_connection', False),
                    'length': len(path_points)
                })
        
        # Sort by alignment and distance
        paths.sort(key=lambda x: (-x['alignment'], x['distance_to_lane']))
        
        return paths[:max_paths]
