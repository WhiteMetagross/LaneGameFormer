"""
Trajectory Prediction Engine
Deterministic mathematical trajectory prediction using kinematics,
social potential fields, game theory, and road constraints.
"""
import numpy as np
import math
from typing import List, Tuple, Dict, Optional
from collections import deque

import config
from road_mask_manager import RoadMaskManager
from lane_manager import LaneManager
from social_potential_field import SocialPotentialField
from game_theory_predictor import GameTheoryPredictor


class TrajectoryPredictor:
    """
    Mathematical deterministic trajectory prediction engine.
    Combines kinematics, lane following, social potential, and game theory.
    """
    
    def __init__(self, road_mask: RoadMaskManager, lane_manager: LaneManager,
                 social_field: SocialPotentialField, game_theory: GameTheoryPredictor,
                 fps: float):
        """
        Initialize the trajectory predictor.
        
        Args:
            road_mask: Road mask manager for boundary constraints
            lane_manager: Lane manager for lane-based paths
            social_field: Social potential field for collision avoidance
            game_theory: Game theory predictor for cooperative behavior
            fps: Video frame rate
        """
        self.road_mask = road_mask
        self.lane_manager = lane_manager
        self.social_field = social_field
        self.game_theory = game_theory
        self.fps = fps
        
        # History requirements - lowered for more responsive predictions
        self.min_history_frames = int(0.1 * fps)  # At least 0.1 seconds (~6 frames at 60fps)
        self.history_frames = int(config.PAST_HISTORY_SECONDS * fps)
    
    def compute_kinematics(self, history: List[Tuple[float, float]]) -> Dict:
        """
        Compute kinematic state from position history.
        
        Args:
            history: List of (x, y) positions (oldest to newest)
            
        Returns:
            Dictionary with position, velocity, acceleration, direction
        """
        if len(history) < 3:
            return None
        
        positions = np.array(history)
        
        # Current position
        position = positions[-1]
        
        # Compute velocities
        dt = 1.0 / self.fps
        velocities = np.diff(positions, axis=0) / dt
        
        # Smoothed velocity (average of last few)
        window = min(config.VELOCITY_SMOOTHING_WINDOW, len(velocities))
        velocity = np.mean(velocities[-window:], axis=0)
        
        # Speed
        speed = np.linalg.norm(velocity)
        
        # Direction
        if speed > 0.5:
            direction = velocity / speed
        else:
            # Use last displacement for direction
            displacement = positions[-1] - positions[-max(3, len(positions)//2)]
            disp_norm = np.linalg.norm(displacement)
            direction = displacement / disp_norm if disp_norm > 0 else np.array([1, 0])
        
        # Compute accelerations
        if len(velocities) >= 3:
            accelerations = np.diff(velocities, axis=0) / dt
            window = min(config.ACCELERATION_SMOOTHING_WINDOW, len(accelerations))
            acceleration = np.mean(accelerations[-window:], axis=0)
        else:
            acceleration = np.array([0.0, 0.0])
        
        # Angular velocity (turning rate)
        angular_velocity = 0.0
        if len(velocities) >= 2 and speed > 1.0:
            v1 = velocities[-2]
            v2 = velocities[-1]
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            dot = np.dot(v1, v2)
            angular_velocity = math.atan2(cross, dot) * self.fps
        
        return {
            'position': position,
            'velocity': velocity,
            'speed': speed,
            'direction': direction,
            'acceleration': acceleration,
            'angular_velocity': angular_velocity
        }
    
    def predict_constant_velocity(self, kinematics: Dict,
                                   time_horizon: float) -> List[Tuple[float, float]]:
        """
        Predict path using constant velocity model with realistic physics.
        Path stops or slows when reaching road boundary.
        
        Args:
            kinematics: Kinematic state dictionary
            time_horizon: Prediction time in seconds
            
        Returns:
            List of predicted (x, y) positions
        """
        num_points = max(20, int(time_horizon * self.fps))  # More points for smoother curves
        
        position = kinematics['position'].copy()
        velocity = kinematics['velocity'].copy()
        speed = kinematics['speed']
        
        dt = time_horizon / num_points
        path = []
        current_pos = position.copy()
        current_vel = velocity.copy()
        
        for i in range(num_points):
            # Predict next position
            next_pos = current_pos + current_vel * dt
            
            # Check if next position is on road
            if self.road_mask.is_point_on_road((float(next_pos[0]), float(next_pos[1])), 
                                                margin=config.ROAD_BOUNDARY_MARGIN):
                path.append((float(next_pos[0]), float(next_pos[1])))
                current_pos = next_pos
            else:
                # At road boundary - stop here
                # Add final stopping point on road
                path.append((float(current_pos[0]), float(current_pos[1])))
                break
        
        return path
    
    def predict_constant_acceleration(self, kinematics: Dict,
                                       time_horizon: float) -> List[Tuple[float, float]]:
        """
        Predict path using constant acceleration model with realistic physics limits.
        Limits acceleration to realistic max values and stops at road boundaries.
        
        Args:
            kinematics: Kinematic state dictionary
            time_horizon: Prediction time in seconds
            
        Returns:
            List of predicted (x, y) positions
        """
        num_points = max(20, int(time_horizon * self.fps))  # More points for smoother curves
        
        position = kinematics['position'].copy()
        velocity = kinematics['velocity'].copy()
        acceleration = kinematics['acceleration'].copy()
        
        # Limit acceleration to realistic max values
        accel_magnitude = np.linalg.norm(acceleration)
        if accel_magnitude > config.MAX_ACCELERATION:
            acceleration = acceleration * (config.MAX_ACCELERATION / accel_magnitude)
        
        dt = time_horizon / num_points
        path = []
        current_pos = position.copy()
        current_vel = velocity.copy()
        
        for i in range(num_points):
            # Update velocity with acceleration
            current_vel = current_vel + acceleration * dt
            
            # Clamp velocity (prevent unrealistic speeds)
            speed = np.linalg.norm(current_vel)
            max_speed = kinematics['speed'] * 2.0  # Max 2x current speed
            if speed > max_speed and speed > 0:
                current_vel = current_vel * (max_speed / speed)
            
            # Predict next position
            next_pos = current_pos + current_vel * dt
            
            # Check if next position is on road
            if self.road_mask.is_point_on_road((float(next_pos[0]), float(next_pos[1])), 
                                                margin=config.ROAD_BOUNDARY_MARGIN):
                path.append((float(next_pos[0]), float(next_pos[1])))
                current_pos = next_pos
            else:
                # At road boundary - stop here
                path.append((float(current_pos[0]), float(current_pos[1])))
                break
        
        return path
    
    def predict_curvilinear(self, kinematics: Dict,
                            time_horizon: float) -> List[Tuple[float, float]]:
        """
        Predict path using curvilinear motion (constant turn rate) with realistic physics.
        Uses max lateral acceleration limit and stops at road boundaries.
        
        Args:
            kinematics: Kinematic state dictionary
            time_horizon: Prediction time in seconds
            
        Returns:
            List of predicted (x, y) positions
        """
        num_points = max(20, int(time_horizon * self.fps))  # More points for smooth curves
        
        position = kinematics['position'].copy()
        speed = kinematics['speed']
        direction = kinematics['direction'].copy()
        omega = kinematics['angular_velocity']
        
        # Limit angular velocity based on max lateral acceleration
        # lateral_accel = speed * omega, so omega_max = max_lateral_accel / speed
        if speed > 1.0:
            omega_max = config.MAX_LATERAL_ACCELERATION / speed
            omega = np.clip(omega, -omega_max, omega_max)
        else:
            omega = np.clip(omega, -0.5, 0.5)
        
        dt = time_horizon / num_points
        path = []
        
        current_pos = position.copy()
        current_dir = direction.copy()
        current_speed = speed
        
        for i in range(num_points):
            # Update direction with limited turn rate
            angle_change = omega * dt
            cos_a = math.cos(angle_change)
            sin_a = math.sin(angle_change)
            new_dir = np.array([
                current_dir[0] * cos_a - current_dir[1] * sin_a,
                current_dir[0] * sin_a + current_dir[1] * cos_a
            ])
            
            # Update position
            step = current_speed * dt
            next_pos = current_pos + new_dir * step
            
            # Check if next position is on road
            if self.road_mask.is_point_on_road((float(next_pos[0]), float(next_pos[1])), 
                                                margin=config.ROAD_BOUNDARY_MARGIN):
                path.append((float(next_pos[0]), float(next_pos[1])))
                current_pos = next_pos
                current_dir = new_dir
            else:
                # At road boundary - stop here
                path.append((float(current_pos[0]), float(current_pos[1])))
                break
        
        return path

    def predict_parametric_curve(self, kinematics: Dict,
                                 time_horizon: float,
                                 speed_factor: float,
                                 turn_rate: float) -> List[Tuple[float, float]]:
        """
        Predict path using parametric control inputs.
        
        Args:
            kinematics: Kinematic state dictionary
            time_horizon: Prediction time in seconds
            speed_factor: Target speed multiplier (1.0 = constant speed)
            turn_rate: Angular velocity in radians/sec
            
        Returns:
            List of predicted (x, y) positions
        """
        num_points = max(20, int(time_horizon * self.fps))
        
        position = kinematics['position'].copy()
        current_speed = kinematics['speed']
        direction = kinematics['direction'].copy()
        
        # Calculate target speed
        target_speed = current_speed * speed_factor
        
        # Calculate acceleration to reach target speed
        accel = (target_speed - current_speed) / time_horizon
        
        dt = time_horizon / num_points
        path = []
        
        current_pos = position.copy()
        current_dir = direction.copy()
        sim_speed = current_speed
        
        for i in range(num_points):
            # Update speed
            sim_speed += accel * dt
            if sim_speed < 0: sim_speed = 0
            
            # Update direction
            angle_change = turn_rate * dt
            cos_a = math.cos(angle_change)
            sin_a = math.sin(angle_change)
            new_dir = np.array([
                current_dir[0] * cos_a - current_dir[1] * sin_a,
                current_dir[0] * sin_a + current_dir[1] * cos_a
            ])
            
            # Update position
            step = sim_speed * dt
            next_pos = current_pos + new_dir * step
            
            # Check road bounds
            if self.road_mask.is_point_on_road((float(next_pos[0]), float(next_pos[1])), 
                                                margin=config.ROAD_BOUNDARY_MARGIN):
                path.append((float(next_pos[0]), float(next_pos[1])))
                current_pos = next_pos
                current_dir = new_dir
            else:
                path.append((float(current_pos[0]), float(current_pos[1])))
                break
                
        return path
    
    def generate_candidate_paths(self, kinematics: Dict,
                                  time_horizon: float) -> List[Dict]:
        """
        Generate multiple candidate prediction paths.
        Creates a diverse set of paths including lane following and 
        various kinematic maneuvers (turns, speed changes).
        """
        candidates = []
        
        # 1. Lane-based predictions (High Priority)
        travel_distance = kinematics['speed'] * time_horizon
        lane_paths = self.lane_manager.get_all_lane_paths(
            tuple(kinematics['position']),
            tuple(kinematics['direction']),
            travel_distance,
            max_paths=4
        )
        
        for lp in lane_paths:
            if lp['alignment'] > 0.5:
                candidates.append({
                    'type': 'lane_following',
                    'path': lp['path_points'],
                    'lane_id': lp['lane_id'],
                    'alignment': lp['alignment'],
                    'base_probability': 0.7 + 0.2 * lp['alignment']
                })

        # 2. Kinematic Diversity Generation
        # Generate paths with different curvatures and speeds
        # (turn_rate, speed_factor, type_name, base_prob)
        
        current_omega = kinematics['angular_velocity']
        
        maneuvers = [
            # Maintain current motion
            (current_omega, 1.0, 'maintain_motion', 0.6),
            
            # Gentle turns (left/right)
            (current_omega + 0.3, 1.0, 'turn_left_gentle', 0.5),
            (current_omega - 0.3, 1.0, 'turn_right_gentle', 0.5),
            
            # Sharp turns (left/right)
            (current_omega + 0.6, 0.8, 'turn_left_sharp', 0.4),
            (current_omega - 0.6, 0.8, 'turn_right_sharp', 0.4),
            
            # Speed variations (Acceleration/Deceleration)
            (current_omega, 1.3, 'accelerate', 0.45),
            (current_omega, 0.7, 'decelerate', 0.45),
            
            # Complex curves (S-curves or varying curvature could be added here, 
            # but parametric curves with constant turn rate are usually sufficient for short horizon)
            # Let's add some "different curved directions" as requested
            (0.15, 1.1, 'curve_left_fast', 0.4),
            (-0.15, 1.1, 'curve_right_fast', 0.4),
            (0.4, 0.6, 'curve_left_slow', 0.35),
            (-0.4, 0.6, 'curve_right_slow', 0.35)
        ]
        
        for turn_rate, speed_factor, ptype, prob in maneuvers:
            # Limit turn rate based on physics
            if kinematics['speed'] > 2.0:
                max_turn = config.MAX_LATERAL_ACCELERATION / kinematics['speed']
                turn_rate = np.clip(turn_rate, -max_turn, max_turn)
            
            path = self.predict_parametric_curve(kinematics, time_horizon, speed_factor, turn_rate)
            constrained_path = self._constrain_path_to_road(path, kinematics)
            
            if constrained_path:
                candidates.append({
                    'type': ptype,
                    'path': constrained_path,
                    'base_probability': prob
                })
        
        return candidates
    
    def _constrain_path_to_road(self, path: List[Tuple[float, float]], 
                                 kinematics: Dict) -> List[Tuple[float, float]]:
        """
        Additional constraint pass for paths that may have been generated
        by external sources (e.g., lane-following). Stops at road boundary.
        
        Args:
            path: Original predicted path
            kinematics: Kinematic state (includes velocity)
            
        Returns:
            Constrained path that stays on road
        """
        if not path:
            return path
        
        constrained_path = []
        
        for point in path:
            # Check if point is on road
            if self.road_mask.is_point_on_road(point, margin=config.ROAD_BOUNDARY_MARGIN):
                constrained_path.append(point)
            else:
                # At road boundary - stop here
                break
        
        # If we got no valid points, return current position as stopped
        if not constrained_path and path:
            pos = kinematics['position']
            constrained_path = [(float(pos[0]), float(pos[1]))]
        
        return constrained_path
    
    def score_path(self, path: List[Tuple[float, float]],
                   kinematics: Dict,
                   nearby_vehicles: List[Dict],
                   base_probability: float = 0.5,
                   ego_id: int = None,
                   all_obbs: List[Dict] = None,
                   strip_width: float = 20.0) -> Tuple[float, Dict]:
        """
        Score a candidate path based on multiple factors.
        Heavily penalizes U-turns, off-road paths, and OBB/strip collisions.
        OBB interiors and other vehicle strips are no-go zones.
        
        Args:
            path: List of (x, y) points
            kinematics: Ego vehicle kinematics
            nearby_vehicles: List of nearby vehicle states
            base_probability: Initial probability for this path type
            ego_id: ID of ego vehicle (to exclude own OBB)
            all_obbs: List of OBB data for collision checking
            strip_width: Width of the prediction strip for collision checking
            
        Returns:
            Tuple of (score, details_dict)
        """
        if not path:
            return 0.0, {'reason': 'empty_path'}
        
        if all_obbs is None:
            all_obbs = []
        
        score = base_probability
        details = {
            'base': base_probability,
            'road_factor': 1.0,
            'social_factor': 1.0,
            'cooperative_factor': 1.0,
            'smoothness_factor': 1.0,
            'direction_factor': 1.0,
            'obb_collision_factor': 1.0,
            'strip_collision_factor': 1.0
        }
        
        # Strip Collision Factor - check if strip collides with any OBB
        if all_obbs:
            strip_collision_penalty = self.social_field.calculate_strip_collision_penalty(
                path, strip_width, all_obbs, exclude_id=ego_id
            )
            if strip_collision_penalty > 0:
                # Complete rejection if strip collides with any OBB
                score = 0.0
                details['strip_collision_factor'] = 0.0
                details['reason'] = 'strip_obb_collision'
                return score, details
        
        # Also check point-based OBB collision as fallback
        if all_obbs:
            obb_collision_penalty = self.social_field.calculate_obb_collision_penalty(
                path, all_obbs, exclude_id=ego_id
            )
            if obb_collision_penalty > 0:
                score = 0.0
                details['obb_collision_factor'] = 0.0
                details['reason'] = 'obb_collision'
                return score, details
        
        # Direction consistency factor - penalize U-turns heavily
        if len(path) >= 2:
            start_pos = np.array(kinematics['position'])
            end_pos = np.array(path[-1])
            path_direction = end_pos - start_pos
            path_dir_norm = np.linalg.norm(path_direction)
            
            if path_dir_norm > 5:  # Only check if there's meaningful movement
                path_direction = path_direction / path_dir_norm
                vehicle_direction = np.array(kinematics['direction'])
                
                # Dot product: 1 = same direction, -1 = opposite (U-turn)
                direction_alignment = np.dot(path_direction, vehicle_direction)
                
                if direction_alignment < 0:  # Going backwards / U-turn
                    direction_factor = 0.05  # Severely penalize
                elif direction_alignment < 0.3:  # Sharp turn
                    direction_factor = 0.2
                else:
                    direction_factor = 0.5 + 0.5 * direction_alignment
                
                score *= direction_factor
                details['direction_factor'] = direction_factor
        
        # Road constraint factor
        is_on_road, road_fraction = self.road_mask.is_path_on_road(
            path, margin=config.ROAD_BOUNDARY_MARGIN
        )
        
        # Calculate road preference score (integral of potential surface)
        road_preference = 0.0
        if len(path) > 0:
            total_score = 0.0
            for p in path:
                total_score += self.road_mask.get_road_score(p[0], p[1])
            road_preference = total_score / len(path)
        
        if not is_on_road:
            # Heavily penalize off-road paths
            road_factor = 0.05 * road_fraction
        else:
            # Combine binary validity with continuous preference
            # Base 0.5 + up to 0.5 from preference (0.4 to 1.0 range)
            road_factor = 0.5 + 0.5 * road_preference
        
        score *= road_factor
        details['road_factor'] = road_factor
        details['road_preference'] = road_preference
        
        # Social potential factor
        total_potential, potentials = self.social_field.calculate_path_potential(
            path, nearby_vehicles, exclude_id=None
        )
        
        # Convert potential to factor (higher potential = lower score)
        if total_potential > 0:
            social_factor = math.exp(-total_potential * 2)
        else:
            social_factor = 1.0
        
        score *= social_factor
        details['social_factor'] = social_factor
        
        # Cooperative factor
        ego_state = {
            'position': kinematics['position'],
            'velocity': kinematics['velocity']
        }
        
        coop_score = self.game_theory.evaluate_path_cooperativeness(
            path, ego_state, nearby_vehicles,
            config.FUTURE_PREDICTION_SECONDS, self.fps
        )
        
        cooperative_factor = 0.5 + 0.5 * coop_score
        score *= cooperative_factor
        details['cooperative_factor'] = cooperative_factor
        
        # Path smoothness factor - penalize sharp turns
        if len(path) >= 3:
            curvature_sum = 0
            max_curvature = 0
            for i in range(1, len(path) - 1):
                p0 = np.array(path[i-1])
                p1 = np.array(path[i])
                p2 = np.array(path[i+1])
                
                v1 = p1 - p0
                v2 = p2 - p1
                
                cross = abs(v1[0]*v2[1] - v1[1]*v2[0])
                dot = np.dot(v1, v2)
                
                curvature = abs(math.atan2(cross, dot + 1e-6))
                curvature_sum += curvature
                max_curvature = max(max_curvature, curvature)
            
            avg_curvature = curvature_sum / (len(path) - 2)
            # Penalize both average and max curvature (avoid sharp turns)
            smoothness_factor = math.exp(-avg_curvature * 0.5) * math.exp(-max_curvature * 0.3)
        else:
            smoothness_factor = 1.0
        
        score *= smoothness_factor
        details['smoothness_factor'] = smoothness_factor
        
        return score, details
    
    def predict_trajectories(self, ego_id: int,
                              ego_history: List[Tuple[float, float]],
                              all_vehicles: Dict[int, List[Tuple[float, float]]],
                              time_horizon: float = None,
                              all_obbs: List[Dict] = None,
                              ego_strip_width: float = None) -> List[Dict]:
        """
        Predict top trajectories for a vehicle.
        OBB interiors and other vehicle strips are no-go zones.
        
        Args:
            ego_id: ID of ego vehicle
            ego_history: Position history of ego vehicle
            all_vehicles: Dictionary of all vehicle histories
            time_horizon: Prediction time in seconds
            all_obbs: List of OBB data dicts with 'track_id' and 'obb_corners'
            ego_strip_width: Width of ego vehicle's strip (from OBB)
            
        Returns:
            List of top predicted path dictionaries, sorted by probability
        """
        if time_horizon is None:
            time_horizon = config.FUTURE_PREDICTION_SECONDS
        
        if all_obbs is None:
            all_obbs = []
        
        # Compute strip width from ego's OBB if not provided
        if ego_strip_width is None:
            ego_strip_width = 20.0  # Default
            for obb_data in all_obbs:
                if obb_data.get('track_id') == ego_id:
                    obb_corners = obb_data.get('obb_corners', [])
                    if len(obb_corners) == 4:
                        ego_strip_width = self._compute_obb_width(obb_corners)
                    break
        
        # Check minimum history (very low threshold)
        if len(ego_history) < 3:
            return []
        
        # Compute kinematics
        kinematics = self.compute_kinematics(ego_history)
        if kinematics is None:
            return []
        
        # Even slow/stationary vehicles get predictions
        if kinematics['speed'] < 0.5:
            # For stationary vehicles, just return current position
            # Return 2 points to allow arrow drawing
            pos = kinematics['position']
            direction = kinematics['direction']
            # Ensure direction is valid
            if np.linalg.norm(direction) < 0.1:
                direction = np.array([1.0, 0.0])
            
            p1 = (float(pos[0]), float(pos[1]))
            p2 = (float(pos[0] + direction[0] * 10), float(pos[1] + direction[1] * 10))
            
            return [{
                'path_points': [p1, p2],
                'probability': 1.0,
                'type': 'stationary',
                'details': {},
                'strip_width': ego_strip_width
            }]
        
        # Build nearby vehicle states
        nearby_vehicles = []
        for vid, vhistory in all_vehicles.items():
            if vid == ego_id or len(vhistory) < 5:
                continue
            
            v_kinematics = self.compute_kinematics(vhistory)
            if v_kinematics is not None:
                # Calculate distance
                ego_pos = kinematics['position']
                v_pos = v_kinematics['position']
                distance = np.linalg.norm(ego_pos - v_pos)
                
                if distance < config.SOCIAL_POTENTIAL_RADIUS_LOW * 2:
                    nearby_vehicles.append({
                        'id': vid,
                        'position': v_kinematics['position'],
                        'velocity': v_kinematics['velocity'],
                        'direction': v_kinematics['direction'],
                        'speed': v_kinematics['speed']
                    })
        
        # Generate candidate paths
        candidates = self.generate_candidate_paths(kinematics, time_horizon)
        
        # Add cooperative predictions
        ego_state = {
            'id': ego_id,
            'position': kinematics['position'],
            'velocity': kinematics['velocity'],
            'direction': kinematics['direction']
        }
        
        coop_predictions = self.game_theory.predict_cooperative_trajectories(
            ego_state, nearby_vehicles, time_horizon, self.fps
        )
        
        for cp in coop_predictions:
            candidates.append({
                'type': cp['type'],
                'path': cp['path'],
                'base_probability': cp['probability']
            })
        
        # Score all candidates (with strip collision checking)
        scored_paths = []
        for candidate in candidates:
            score, details = self.score_path(
                candidate['path'],
                kinematics,
                nearby_vehicles,
                candidate.get('base_probability', 0.5),
                ego_id=ego_id,
                all_obbs=all_obbs,
                strip_width=ego_strip_width
            )
            
            if score > config.MIN_PATH_PROBABILITY:
                scored_paths.append({
                    'path_points': candidate['path'],
                    'probability': score,
                    'type': candidate['type'],
                    'lane_id': candidate.get('lane_id'),
                    'details': details,
                    'strip_width': ego_strip_width
                })
        
        # Sort by probability and return top paths
        scored_paths.sort(key=lambda x: x['probability'], reverse=True)
        
        # Remove duplicates (paths that are too similar)
        unique_paths = []
        for path in scored_paths:
            is_duplicate = False
            for existing in unique_paths:
                if self._paths_similar(path['path_points'], existing['path_points']):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_paths.append(path)
            
            if len(unique_paths) >= config.NUM_PREDICTION_PATHS:
                break
        
        return unique_paths
    
    def _compute_obb_width(self, obb_corners: List[Tuple[float, float]]) -> float:
        """
        Compute the width (shorter dimension) of an OBB from its corners.
        
        Args:
            obb_corners: List of 4 corner points
            
        Returns:
            Width of the OBB
        """
        if len(obb_corners) != 4:
            return 20.0  # Default
        
        # Compute lengths of first two edges
        p0, p1, p2, p3 = obb_corners
        edge1_len = math.sqrt((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2)
        edge2_len = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
        
        # Width is the shorter edge
        return min(edge1_len, edge2_len)
    
    def _paths_similar(self, path1: List[Tuple[float, float]],
                       path2: List[Tuple[float, float]],
                       threshold: float = 15.0) -> bool:
        """Check if two paths are similar (endpoint distance)."""
        if not path1 or not path2:
            return False
        
        end1 = np.array(path1[-1])
        end2 = np.array(path2[-1])
        
        return np.linalg.norm(end1 - end2) < threshold
    
    def get_safe_paths(self, predictions: List[Dict]) -> List[Dict]:
        """
        Filter predictions to only include safe paths.
        
        Args:
            predictions: List of predicted path dictionaries
            
        Returns:
            List of safe paths
        """
        safe_paths = []
        
        for pred in predictions:
            details = pred.get('details', {})
            
            # Check all safety factors
            if (details.get('road_factor', 0) > 0.5 and
                details.get('social_factor', 0) > 0.3 and
                details.get('cooperative_factor', 0) > 0.4):
                safe_paths.append(pred)
        
        return safe_paths
