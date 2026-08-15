"""
Game Theory Cooperative Predictor
Implements cooperative trajectory prediction based on game-theoretic principles
"""
import numpy as np
import math
from typing import List, Tuple, Dict, Optional

import config


class GameTheoryPredictor:
    """
    Implements game-theoretic cooperative path prediction.
    Handles yielding, right-of-way, and cooperative maneuvers.
    """
    
    def __init__(self):
        """Initialize the game theory predictor."""
        pass
    
    def calculate_priority_score(self, vehicle: Dict, 
                                  relative_to: Dict) -> float:
        """
        Calculate priority score for a vehicle relative to another.
        Higher score = higher priority (right of way).
        
        Args:
            vehicle: Vehicle state dictionary
            relative_to: Other vehicle to compare against
            
        Returns:
            Priority score (0-1)
        """
        score = 0.0
        
        # Speed factor: faster vehicles have lower priority (should yield)
        # This encourages defensive driving
        ego_speed = np.linalg.norm(vehicle.get('velocity', [0, 0]))
        other_speed = np.linalg.norm(relative_to.get('velocity', [0, 0]))
        
        if ego_speed + other_speed > 0:
            speed_ratio = other_speed / (ego_speed + other_speed + 1e-6)
            score += config.SPEED_PRIORITY_WEIGHT * speed_ratio
        
        # Direction factor: vehicle going straight has priority over turning
        ego_dir = self._get_direction_vector(vehicle)
        other_dir = self._get_direction_vector(relative_to)
        
        if ego_dir is not None and other_dir is not None:
            # Calculate relative angle
            dot = np.dot(ego_dir, other_dir)
            cross = ego_dir[0] * other_dir[1] - ego_dir[1] * other_dir[0]
            angle = abs(math.degrees(math.atan2(cross, dot)))
            
            # Give priority to vehicle going more straight
            direction_score = 1.0 - (angle / 180.0)
            score += config.DIRECTION_PRIORITY_WEIGHT * direction_score
        
        # Distance factor: closer vehicle has priority
        ego_pos = np.array(vehicle['position'])
        other_pos = np.array(relative_to['position'])
        distance = np.linalg.norm(other_pos - ego_pos)
        
        max_dist = config.RIGHT_OF_WAY_DISTANCE * 2
        distance_score = max(0, 1.0 - distance / max_dist)
        score += config.DISTANCE_PRIORITY_WEIGHT * distance_score
        
        return min(1.0, max(0.0, score))
    
    def _get_direction_vector(self, vehicle: Dict) -> Optional[np.ndarray]:
        """Get normalized direction vector for a vehicle."""
        if 'velocity' in vehicle:
            vel = np.array(vehicle['velocity'])
            norm = np.linalg.norm(vel)
            if norm > 0.1:
                return vel / norm
        
        if 'direction' in vehicle:
            return np.array(vehicle['direction'])
        
        return None
    
    def should_yield(self, ego: Dict, other: Dict) -> Tuple[bool, str]:
        """
        Determine if ego vehicle should yield to another vehicle.
        
        Args:
            ego: Ego vehicle state
            other: Other vehicle state
            
        Returns:
            Tuple of (should_yield, reason)
        """
        ego_priority = self.calculate_priority_score(ego, other)
        other_priority = self.calculate_priority_score(other, ego)
        
        # Check if vehicles are on crossing paths
        if self._are_paths_crossing(ego, other):
            # Vehicle with lower priority should yield
            if ego_priority < other_priority - 0.1:
                return True, "crossing_paths_lower_priority"
            
            # If similar priority, slower vehicle yields
            ego_speed = np.linalg.norm(ego.get('velocity', [0, 0]))
            other_speed = np.linalg.norm(other.get('velocity', [0, 0]))
            
            if ego_speed < other_speed * 0.8:
                return True, "crossing_paths_slower"
        
        # Check if other is directly ahead and going same direction
        if self._is_ahead_same_direction(ego, other):
            return True, "following_vehicle_ahead"
        
        # Check if ego is cutting across other's path
        if self._is_cutting_across(ego, other):
            return True, "cutting_across"
        
        return False, "no_yield_needed"
    
    def _are_paths_crossing(self, ego: Dict, other: Dict) -> bool:
        """Check if two vehicles have crossing paths."""
        ego_dir = self._get_direction_vector(ego)
        other_dir = self._get_direction_vector(other)
        
        if ego_dir is None or other_dir is None:
            return False
        
        # Calculate angle between directions
        dot = np.dot(ego_dir, other_dir)
        angle = math.degrees(math.acos(max(-1, min(1, dot))))
        
        # Paths cross if angle is between threshold values
        return config.YIELD_ANGLE_THRESHOLD < angle < (180 - config.YIELD_ANGLE_THRESHOLD)
    
    def _is_ahead_same_direction(self, ego: Dict, other: Dict) -> bool:
        """Check if other vehicle is ahead and going same direction."""
        ego_pos = np.array(ego['position'])
        other_pos = np.array(other['position'])
        ego_dir = self._get_direction_vector(ego)
        other_dir = self._get_direction_vector(other)
        
        if ego_dir is None or other_dir is None:
            return False
        
        # Check direction alignment
        dot = np.dot(ego_dir, other_dir)
        if dot < 0.7:  # Not same direction
            return False
        
        # Check if other is ahead
        to_other = other_pos - ego_pos
        to_other_norm = to_other / (np.linalg.norm(to_other) + 1e-6)
        
        forward_dist = np.dot(to_other, ego_dir)
        
        return forward_dist > 0 and np.linalg.norm(to_other) < config.RIGHT_OF_WAY_DISTANCE
    
    def _is_cutting_across(self, ego: Dict, other: Dict) -> bool:
        """Check if ego is cutting across other's path."""
        ego_pos = np.array(ego['position'])
        other_pos = np.array(other['position'])
        ego_dir = self._get_direction_vector(ego)
        other_dir = self._get_direction_vector(other)
        
        if ego_dir is None or other_dir is None:
            return False
        
        # Project ego's future position
        ego_vel = np.array(ego.get('velocity', [0, 0]))
        ego_future = ego_pos + ego_vel * 30  # 30 frames ahead
        
        # Check if ego crosses other's path
        other_vel = np.array(other.get('velocity', [0, 0]))
        
        # Check if ego trajectory crosses in front of other
        to_ego_future = ego_future - other_pos
        forward_component = np.dot(to_ego_future, other_dir)
        
        if 0 < forward_component < config.RIGHT_OF_WAY_DISTANCE:
            lateral_component = abs(np.cross(other_dir, to_ego_future))
            if lateral_component < config.SOCIAL_POTENTIAL_RADIUS_HIGH:
                return True
        
        return False
    
    def get_cooperative_maneuver(self, ego: Dict, 
                                  nearby_vehicles: List[Dict]) -> Dict:
        """
        Determine the best cooperative maneuver for ego vehicle.
        
        Args:
            ego: Ego vehicle state
            nearby_vehicles: List of nearby vehicle states
            
        Returns:
            Maneuver recommendation dictionary
        """
        maneuver = {
            'type': 'continue',
            'speed_adjustment': 1.0,  # Multiplier
            'lateral_offset': 0.0,
            'yield_to': [],
            'reason': 'no_conflict'
        }
        
        for other in nearby_vehicles:
            should_yield, reason = self.should_yield(ego, other)
            
            if should_yield:
                maneuver['yield_to'].append({
                    'vehicle_id': other.get('id'),
                    'reason': reason
                })
        
        # Determine maneuver based on yield decisions
        if maneuver['yield_to']:
            maneuver['type'] = 'yield'
            
            # Calculate speed reduction
            num_yields = len(maneuver['yield_to'])
            maneuver['speed_adjustment'] = max(0.3, 1.0 - 0.2 * num_yields)
            
            # Calculate lateral offset if needed
            for yield_info in maneuver['yield_to']:
                reason = yield_info['reason']
                if reason == 'cutting_across':
                    maneuver['lateral_offset'] = -10.0  # Move right
                    break
            
            maneuver['reason'] = maneuver['yield_to'][0]['reason']
        
        return maneuver
    
    def predict_cooperative_trajectories(self, ego: Dict,
                                          nearby_vehicles: List[Dict],
                                          time_horizon: float,
                                          fps: float) -> List[Dict]:
        """
        Predict trajectories considering cooperative behavior.
        
        Args:
            ego: Ego vehicle state with position, velocity, history
            nearby_vehicles: Nearby vehicle states
            time_horizon: Prediction time in seconds
            fps: Frame rate
            
        Returns:
            List of predicted trajectory options with probabilities
        """
        predictions = []
        maneuver = self.get_cooperative_maneuver(ego, nearby_vehicles)
        
        ego_pos = np.array(ego['position'])
        ego_vel = np.array(ego.get('velocity', [0, 0]))
        
        num_frames = int(time_horizon * fps)
        
        # Option 1: Continue current trajectory (if safe)
        if maneuver['type'] == 'continue':
            path = self._predict_constant_velocity(ego_pos, ego_vel, num_frames)
            predictions.append({
                'type': 'continue',
                'path': path,
                'probability': 0.8,
                'maneuver': maneuver
            })
        
        # Option 2: Yield (slow down)
        if maneuver['type'] == 'yield':
            adjusted_vel = ego_vel * maneuver['speed_adjustment']
            path = self._predict_constant_velocity(ego_pos, adjusted_vel, num_frames)
            predictions.append({
                'type': 'yield',
                'path': path,
                'probability': 0.7,
                'maneuver': maneuver
            })
        
        # Option 3: Lane change or lateral offset
        if maneuver['lateral_offset'] != 0:
            lateral_dir = np.array([-ego_vel[1], ego_vel[0]])
            lateral_dir = lateral_dir / (np.linalg.norm(lateral_dir) + 1e-6)
            offset_pos = ego_pos + lateral_dir * maneuver['lateral_offset']
            
            path = self._predict_constant_velocity(offset_pos, ego_vel, num_frames)
            predictions.append({
                'type': 'lane_change',
                'path': path,
                'probability': 0.5,
                'maneuver': maneuver
            })
        
        # Option 4: Follow behind
        for other in nearby_vehicles:
            if self._is_ahead_same_direction(ego, other):
                other_pos = np.array(other['position'])
                other_vel = np.array(other.get('velocity', [0, 0]))
                
                # Maintain safe following distance
                follow_offset = -config.SOCIAL_POTENTIAL_RADIUS_MEDIUM
                follow_dir = other_vel / (np.linalg.norm(other_vel) + 1e-6)
                follow_pos = other_pos + follow_dir * follow_offset
                
                path = self._predict_following(follow_pos, other_vel, num_frames)
                predictions.append({
                    'type': 'follow',
                    'path': path,
                    'probability': 0.6,
                    'following': other.get('id')
                })
                break
        
        return predictions
    
    def _predict_constant_velocity(self, pos: np.ndarray, vel: np.ndarray,
                                    num_frames: int) -> List[Tuple[float, float]]:
        """Predict path with constant velocity."""
        path = []
        current_pos = pos.copy()
        
        for _ in range(num_frames):
            current_pos = current_pos + vel
            path.append((float(current_pos[0]), float(current_pos[1])))
        
        return path
    
    def _predict_following(self, start_pos: np.ndarray, lead_vel: np.ndarray,
                           num_frames: int) -> List[Tuple[float, float]]:
        """Predict path following another vehicle."""
        path = []
        current_pos = start_pos.copy()
        
        for _ in range(num_frames):
            current_pos = current_pos + lead_vel
            path.append((float(current_pos[0]), float(current_pos[1])))
        
        return path
    
    def evaluate_path_cooperativeness(self, path: List[Tuple[float, float]],
                                       ego: Dict,
                                       nearby_vehicles: List[Dict],
                                       time_horizon: float,
                                       fps: float) -> float:
        """
        Evaluate how cooperative a predicted path is.
        
        Args:
            path: Predicted path points
            ego: Ego vehicle state
            nearby_vehicles: Nearby vehicle states
            time_horizon: Prediction time
            fps: Frame rate
            
        Returns:
            Cooperativeness score (0-1, higher is more cooperative)
        """
        if not path:
            return 0.0
        
        score = 1.0
        num_frames = int(time_horizon * fps)
        
        for i, point in enumerate(path):
            # Project other vehicles forward
            t = i / fps  # Time at this point
            
            for other in nearby_vehicles:
                other_pos = np.array(other['position'])
                other_vel = np.array(other.get('velocity', [0, 0]))
                
                # Future position of other vehicle
                other_future = other_pos + other_vel * i
                
                # Distance at this time step
                distance = np.linalg.norm(np.array(point) - other_future)
                
                # Penalize getting too close
                if distance < config.SOCIAL_POTENTIAL_RADIUS_CRITICAL:
                    score -= 0.3
                elif distance < config.SOCIAL_POTENTIAL_RADIUS_HIGH:
                    score -= 0.1
                elif distance < config.SOCIAL_POTENTIAL_RADIUS_MEDIUM:
                    score -= 0.05
        
        return max(0.0, min(1.0, score))
