"""
Social Potential Field System
Implements social force model for collision avoidance and cooperative behavior
"""
import numpy as np
import cv2
import math
from typing import List, Tuple, Dict, Optional

import config


class SocialPotentialField:
    """
    Implements a social potential field around vehicles for collision avoidance.
    Based on social force models with exponential decay.
    """
    
    def __init__(self, frame_width: int, frame_height: int):
        """
        Initialize the social potential field system.
        
        Args:
            frame_width: Video frame width
            frame_height: Video frame height
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Radius bands (from config)
        self.radius_bands = [
            (config.SOCIAL_POTENTIAL_RADIUS_CRITICAL, config.POTENTIAL_DECAY_CRITICAL),
            (config.SOCIAL_POTENTIAL_RADIUS_HIGH, config.POTENTIAL_DECAY_HIGH),
            (config.SOCIAL_POTENTIAL_RADIUS_MEDIUM, config.POTENTIAL_DECAY_MEDIUM),
            (config.SOCIAL_POTENTIAL_RADIUS_LOW, config.POTENTIAL_DECAY_LOW),
        ]
    
    def calculate_potential_at_point(self, point: Tuple[float, float],
                                     vehicles: List[Dict],
                                     exclude_id: int = None) -> float:
        """
        Calculate the social potential at a specific point.
        
        Args:
            point: (x, y) coordinates
            vehicles: List of vehicle states with position, velocity
            exclude_id: Vehicle ID to exclude (self)
            
        Returns:
            Total potential value at the point (higher = more dangerous)
        """
        total_potential = 0.0
        px, py = point
        
        for vehicle in vehicles:
            if exclude_id is not None and vehicle.get('id') == exclude_id:
                continue
            
            vx, vy = vehicle['position']
            distance = math.sqrt((px - vx)**2 + (py - vy)**2)
            
            if distance < 1e-6:
                distance = 1e-6
            
            # Calculate potential based on distance bands
            potential = 0.0
            for radius, decay in self.radius_bands:
                if distance < radius:
                    # Exponential decay with distance
                    potential = max(potential, 
                                   decay * math.exp(-distance / (radius * 0.3)))
            
            # Factor in velocity (vehicles moving towards point are more dangerous)
            if 'velocity' in vehicle:
                vel = vehicle['velocity']
                to_point = np.array([px - vx, py - vy])
                to_point_norm = to_point / (np.linalg.norm(to_point) + 1e-6)
                vel_norm = np.array(vel) / (np.linalg.norm(vel) + 1e-6)
                
                # Dot product: positive if moving towards point
                approach_factor = np.dot(vel_norm, to_point_norm)
                if approach_factor > 0:
                    potential *= (1.0 + 0.5 * approach_factor)
            
            total_potential += potential
        
        return total_potential
    
    def calculate_path_potential(self, path: List[Tuple[float, float]],
                                  vehicles: List[Dict],
                                  exclude_id: int = None) -> Tuple[float, List[float]]:
        """
        Calculate total potential along a path.
        
        Args:
            path: List of (x, y) points
            vehicles: List of vehicle states
            exclude_id: Vehicle ID to exclude
            
        Returns:
            Tuple of (total_potential, list of potentials at each point)
        """
        if not path:
            return 0.0, []
        
        potentials = []
        for point in path:
            p = self.calculate_potential_at_point(point, vehicles, exclude_id)
            potentials.append(p)
        
        # Total is max + average (penalize any high-risk point)
        if potentials:
            total = max(potentials) + 0.5 * np.mean(potentials)
        else:
            total = 0.0
        
        return total, potentials
    
    def get_gradient_at_point(self, point: Tuple[float, float],
                               vehicles: List[Dict],
                               exclude_id: int = None,
                               delta: float = 2.0) -> Tuple[float, float]:
        """
        Calculate the gradient of the potential field at a point.
        Gradient points towards lower potential (safer direction).
        
        Args:
            point: (x, y) coordinates
            vehicles: List of vehicle states
            exclude_id: Vehicle ID to exclude
            delta: Step size for numerical gradient
            
        Returns:
            (dx, dy) gradient vector pointing towards safer areas
        """
        px, py = point
        
        # Sample potential in cardinal directions
        pot_center = self.calculate_potential_at_point(point, vehicles, exclude_id)
        pot_right = self.calculate_potential_at_point((px + delta, py), vehicles, exclude_id)
        pot_left = self.calculate_potential_at_point((px - delta, py), vehicles, exclude_id)
        pot_up = self.calculate_potential_at_point((px, py - delta), vehicles, exclude_id)
        pot_down = self.calculate_potential_at_point((px, py + delta), vehicles, exclude_id)
        
        # Gradient (negative because we want to move to lower potential)
        grad_x = -(pot_right - pot_left) / (2 * delta)
        grad_y = -(pot_down - pot_up) / (2 * delta)
        
        return (grad_x, grad_y)
    
    def is_path_safe(self, path: List[Tuple[float, float]],
                     vehicles: List[Dict],
                     exclude_id: int = None,
                     threshold: float = 0.5) -> bool:
        """
        Check if a path is safe (potential below threshold).
        
        Args:
            path: List of (x, y) points
            vehicles: List of vehicle states
            exclude_id: Vehicle ID to exclude
            threshold: Maximum acceptable potential
            
        Returns:
            True if path is safe
        """
        total, potentials = self.calculate_path_potential(path, vehicles, exclude_id)
        
        # Check if any point exceeds critical threshold
        for p in potentials:
            if p > config.POTENTIAL_DECAY_CRITICAL * 0.8:
                return False
        
        return total < threshold
    
    def create_potential_overlay(self, vehicles: List[Dict],
                                  alpha: float = None) -> np.ndarray:
        """
        Create a visualization overlay of the social potential field.
        
        Args:
            vehicles: List of vehicle states with positions
            alpha: Overlay transparency
            
        Returns:
            BGRA overlay image with potential field visualization
        """
        if alpha is None:
            alpha = config.SOCIAL_POTENTIAL_ALPHA
        
        overlay = np.zeros((self.frame_height, self.frame_width, 4), dtype=np.uint8)
        
        for vehicle in vehicles:
            vx, vy = vehicle['position']
            vx, vy = int(round(vx)), int(round(vy))
            
            # Draw concentric circles for each band
            for i, (radius, decay) in enumerate(self.radius_bands):
                if i < len(config.HEATMAP_COLORS):
                    color = config.HEATMAP_COLORS[i]
                else:
                    color = (128, 128, 128)
                
                # Draw filled circle with decreasing alpha
                band_alpha = int(255 * alpha * decay)
                thickness = max(2, int(radius * 0.1))
                
                # Draw as rings (filled circles would overlap badly)
                cv2.circle(overlay, (vx, vy), int(radius), 
                          (*color, band_alpha), thickness)
        
        return overlay
    
    def create_gradient_potential_overlay(self, frame: np.ndarray,
                                           vehicles: List[Dict],
                                           alpha: float = None) -> np.ndarray:
        """
        Create a smooth gradient heatmap overlay for social potential field.
        Closer to center = more critical (red), farther = safer (green/blue).
        
        Args:
            frame: Input BGR frame
            vehicles: List of vehicle states with positions
            alpha: Base overlay transparency
            
        Returns:
            Frame with gradient potential field overlay
        """
        if alpha is None:
            alpha = config.SOCIAL_POTENTIAL_ALPHA
        
        result = frame.copy()
        max_radius = config.SOCIAL_POTENTIAL_RADIUS_LOW
        
        # Create a combined potential field map
        potential_map = np.zeros((self.frame_height, self.frame_width), dtype=np.float32)
        
        for vehicle in vehicles:
            vx, vy = vehicle['position']
            vx, vy = int(round(vx)), int(round(vy))
            
            if not (0 <= vx < self.frame_width and 0 <= vy < self.frame_height):
                continue
            
            # Create distance map from this vehicle
            y_coords, x_coords = np.ogrid[:self.frame_height, :self.frame_width]
            dist_map = np.sqrt((x_coords - vx)**2 + (y_coords - vy)**2)
            
            # Calculate potential (higher = more dangerous, exponential decay)
            # Mask out areas beyond max radius
            mask = dist_map < max_radius
            vehicle_potential = np.zeros_like(dist_map)
            vehicle_potential[mask] = np.exp(-dist_map[mask] / (max_radius * 0.25))
            
            # Add to combined map (take max for overlapping fields)
            potential_map = np.maximum(potential_map, vehicle_potential)
        
        # Normalize potential map to 0-1
        if potential_map.max() > 0:
            potential_map = potential_map / potential_map.max()
        
        # Create heatmap colors using custom gradient
        # Red (danger) -> Orange -> Yellow -> Green -> Blue (safe)
        heatmap = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        
        # Only color areas with potential > threshold
        mask = potential_map > 0.05
        
        if mask.any():
            # Map potential to heatmap colors (BGR format)
            # High potential (1.0) = Red (danger), Low potential (0.0) = Green/Blue (safe)
            for y in range(self.frame_height):
                for x in range(self.frame_width):
                    if mask[y, x]:
                        p = potential_map[y, x]
                        # Color gradient: Red (1.0) -> Orange (0.75) -> Yellow (0.5) -> Green (0.25) -> Blue (0.0)
                        if p > 0.75:
                            # Red to Orange
                            t = (p - 0.75) / 0.25
                            heatmap[y, x] = (0, int(128 * (1 - t)), 255)  # BGR: Red -> Orange
                        elif p > 0.5:
                            # Orange to Yellow
                            t = (p - 0.5) / 0.25
                            heatmap[y, x] = (0, int(128 + 127 * (1 - t)), 255)  # BGR: Orange -> Yellow
                        elif p > 0.25:
                            # Yellow to Green
                            t = (p - 0.25) / 0.25
                            heatmap[y, x] = (0, 255, int(255 * t))  # BGR: Yellow -> Green
                        else:
                            # Green to Cyan/Blue
                            t = p / 0.25
                            heatmap[y, x] = (int(255 * (1 - t)), 255, 0)  # BGR: Green -> Cyan
            
            # Create alpha mask based on potential (stronger potential = more opaque)
            alpha_mask = (potential_map * alpha * 255).astype(np.uint8)
            alpha_mask = cv2.GaussianBlur(alpha_mask, (5, 5), 0)
            
            # Blend with frame
            for c in range(3):
                result[:, :, c] = np.where(
                    mask,
                    (result[:, :, c] * (255 - alpha_mask) + heatmap[:, :, c] * alpha_mask) // 255,
                    result[:, :, c]
                )
        
        return result
    
    def create_gradient_potential_overlay_fast(self, frame: np.ndarray,
                                                vehicles: List[Dict],
                                                alpha: float = None) -> np.ndarray:
        """
        Fast version of gradient potential overlay using OpenCV color mapping.
        
        Args:
            frame: Input BGR frame
            vehicles: List of vehicle states with positions
            alpha: Base overlay transparency
            
        Returns:
            Frame with gradient potential field overlay
        """
        if alpha is None:
            alpha = config.SOCIAL_POTENTIAL_ALPHA
        
        if not vehicles:
            return frame
        
        result = frame.copy()
        max_radius = int(config.SOCIAL_POTENTIAL_RADIUS_LOW)
        
        # Create a combined potential field map
        potential_map = np.zeros((self.frame_height, self.frame_width), dtype=np.float32)
        
        for vehicle in vehicles:
            vx, vy = vehicle['position']
            vx, vy = int(round(vx)), int(round(vy))
            
            if not (0 <= vx < self.frame_width and 0 <= vy < self.frame_height):
                continue
            
            # Create local region around vehicle for efficiency
            x1 = max(0, vx - max_radius)
            y1 = max(0, vy - max_radius)
            x2 = min(self.frame_width, vx + max_radius)
            y2 = min(self.frame_height, vy + max_radius)
            
            # Create distance map for local region
            local_y = np.arange(y1, y2)
            local_x = np.arange(x1, x2)
            xx, yy = np.meshgrid(local_x, local_y)
            dist_map = np.sqrt((xx - vx)**2 + (yy - vy)**2)
            
            # Calculate potential (exponential decay)
            vehicle_potential = np.exp(-dist_map / (max_radius * 0.3))
            vehicle_potential[dist_map > max_radius] = 0
            
            # Add to combined map using maximum
            potential_map[y1:y2, x1:x2] = np.maximum(
                potential_map[y1:y2, x1:x2], 
                vehicle_potential
            )
        
        # Normalize and create mask
        mask = potential_map > 0.05
        if not mask.any():
            return result
        
        # Normalize to 0-255 for colormap
        potential_norm = np.zeros_like(potential_map, dtype=np.uint8)
        potential_norm[mask] = (potential_map[mask] * 255).astype(np.uint8)
        
        # Apply colormap (JET: Blue->Cyan->Green->Yellow->Red)
        # We reverse it so high potential = red (danger)
        heatmap = cv2.applyColorMap(potential_norm, cv2.COLORMAP_JET)
        
        # Create smooth alpha mask based on potential
        alpha_mask = (potential_map * alpha).astype(np.float32)
        alpha_mask = np.clip(alpha_mask, 0, 1)
        alpha_3ch = np.dstack([alpha_mask, alpha_mask, alpha_mask])
        
        # Blend where there's potential
        result = (result * (1 - alpha_3ch) + heatmap * alpha_3ch).astype(np.uint8)
        
        return result
    
    def apply_potential_overlay(self, frame: np.ndarray,
                                 vehicles: List[Dict],
                                 alpha: float = None) -> np.ndarray:
        """
        Apply social potential visualization to a frame.
        
        Args:
            frame: Input BGR frame
            vehicles: List of vehicle states
            alpha: Overlay transparency
            
        Returns:
            Frame with potential field overlay
        """
        if alpha is None:
            alpha = config.SOCIAL_POTENTIAL_ALPHA
        
        result = frame.copy()
        
        for vehicle in vehicles:
            vx, vy = vehicle['position']
            vx, vy = int(round(vx)), int(round(vy))
            
            # Skip if out of frame
            if vx < 0 or vx >= self.frame_width or vy < 0 or vy >= self.frame_height:
                continue
            
            # Draw bands from outer to inner (so inner overlaps outer)
            for i in range(len(self.radius_bands) - 1, -1, -1):
                radius, decay = self.radius_bands[i]
                color = config.HEATMAP_COLORS[i] if i < len(config.HEATMAP_COLORS) else (128, 128, 128)
                
                # Create temporary overlay for this band
                temp_overlay = np.zeros_like(result)
                cv2.circle(temp_overlay, (vx, vy), int(radius), color, -1)
                
                # Blend with result
                band_alpha = alpha * decay
                cv2.addWeighted(temp_overlay, band_alpha, result, 1 - band_alpha, 0, result)
        
        return result
    
    def get_collision_risk(self, ego_pos: Tuple[float, float],
                           ego_vel: Tuple[float, float],
                           other_pos: Tuple[float, float],
                           other_vel: Tuple[float, float],
                           time_horizon: float = 2.0,
                           fps: float = 30.0) -> Tuple[float, float]:
        """
        Calculate collision risk between two vehicles.
        
        Args:
            ego_pos: Ego vehicle position
            ego_vel: Ego vehicle velocity (pixels/frame)
            other_pos: Other vehicle position
            other_vel: Other vehicle velocity (pixels/frame)
            time_horizon: Prediction time in seconds
            fps: Frame rate
            
        Returns:
            Tuple of (collision_risk 0-1, time_to_collision or inf)
        """
        # Convert to numpy
        p1 = np.array(ego_pos)
        v1 = np.array(ego_vel)
        p2 = np.array(other_pos)
        v2 = np.array(other_vel)
        
        # Relative position and velocity
        rel_pos = p2 - p1
        rel_vel = v2 - v1
        
        # Current distance
        current_dist = np.linalg.norm(rel_pos)
        
        # Check if already in collision range
        if current_dist < config.SOCIAL_POTENTIAL_RADIUS_CRITICAL:
            return 1.0, 0.0
        
        # Time to closest approach (quadratic formula)
        rel_vel_sq = np.dot(rel_vel, rel_vel)
        
        if rel_vel_sq < 1e-6:
            # Vehicles moving parallel or stationary
            return self._distance_to_risk(current_dist), float('inf')
        
        t_closest = -np.dot(rel_pos, rel_vel) / rel_vel_sq
        
        # Check if closest approach is in the future within horizon
        num_frames = int(time_horizon * fps)
        
        if t_closest < 0:
            # Already diverging
            return self._distance_to_risk(current_dist), float('inf')
        
        if t_closest > num_frames:
            # Too far in future
            return 0.0, float('inf')
        
        # Calculate distance at closest approach
        closest_pos = rel_pos + rel_vel * t_closest
        closest_dist = np.linalg.norm(closest_pos)
        
        # Calculate risk
        risk = self._distance_to_risk(closest_dist)
        time_to_collision = t_closest / fps if risk > 0.5 else float('inf')
        
        return risk, time_to_collision
    
    def _distance_to_risk(self, distance: float) -> float:
        """Convert distance to collision risk (0-1)."""
        if distance < config.SOCIAL_POTENTIAL_RADIUS_CRITICAL:
            return 1.0
        elif distance < config.SOCIAL_POTENTIAL_RADIUS_HIGH:
            return 0.8
        elif distance < config.SOCIAL_POTENTIAL_RADIUS_MEDIUM:
            return 0.5
        elif distance < config.SOCIAL_POTENTIAL_RADIUS_LOW:
            return 0.2
        else:
            return 0.0
    
    def is_point_inside_obb(self, point: Tuple[float, float], 
                            obb_corners: List[Tuple[float, float]]) -> bool:
        """
        Check if a point is inside an oriented bounding box.
        Uses cross product method for convex polygon point-in-polygon test.
        
        Args:
            point: (x, y) coordinates to check
            obb_corners: List of 4 corner points of the OBB
            
        Returns:
            True if point is inside the OBB
        """
        if not obb_corners or len(obb_corners) != 4:
            return False
        
        px, py = point
        n = len(obb_corners)
        
        # Check if point is on the same side of all edges
        sign = None
        for i in range(n):
            x1, y1 = obb_corners[i]
            x2, y2 = obb_corners[(i + 1) % n]
            
            # Cross product of edge vector and vector to point
            cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
            
            if abs(cross) < 1e-6:
                continue  # Point is on the edge
            
            current_sign = cross > 0
            
            if sign is None:
                sign = current_sign
            elif sign != current_sign:
                return False  # Point is outside
        
        return True
    
    def is_path_intersecting_obb(self, path: List[Tuple[float, float]],
                                  obb_corners: List[Tuple[float, float]]) -> bool:
        """
        Check if any point on a path intersects with an OBB.
        
        Args:
            path: List of (x, y) points
            obb_corners: List of 4 corner points of the OBB
            
        Returns:
            True if path intersects with the OBB
        """
        if not path or not obb_corners:
            return False
        
        for point in path:
            if self.is_point_inside_obb(point, obb_corners):
                return True
        
        return False
    
    def calculate_obb_collision_penalty(self, path: List[Tuple[float, float]],
                                         all_obbs: List[Dict],
                                         exclude_id: int = None) -> float:
        """
        Calculate collision penalty for a path based on OBB intersections.
        OBB interiors are treated as super critical zones - no path should pass through.
        
        Args:
            path: List of (x, y) points
            all_obbs: List of dictionaries with 'track_id' and 'obb_corners'
            exclude_id: Vehicle ID to exclude (self)
            
        Returns:
            Penalty value (0.0 = no collision, 1.0 = full collision)
        """
        if not path or not all_obbs:
            return 0.0
        
        collision_count = 0
        total_obbs = 0
        
        for obb_data in all_obbs:
            track_id = obb_data.get('track_id')
            obb_corners = obb_data.get('obb_corners')
            
            if exclude_id is not None and track_id == exclude_id:
                continue
            
            if not obb_corners or len(obb_corners) != 4:
                continue
            
            total_obbs += 1
            
            if self.is_path_intersecting_obb(path, obb_corners):
                collision_count += 1
        
        if total_obbs == 0:
            return 0.0
        
        # Return 1.0 if ANY collision (super critical)
        return 1.0 if collision_count > 0 else 0.0
    
    def get_strip_polygon(self, path: List[Tuple[float, float]], 
                          strip_width: float) -> List[Tuple[float, float]]:
        """
        Create a strip polygon from a path for collision detection.
        
        Args:
            path: List of (x, y) points
            strip_width: Width of the strip
            
        Returns:
            List of polygon corner points
        """
        import math
        
        if not path or len(path) < 2:
            return []
        
        half_width = strip_width / 2.0
        left_edge = []
        right_edge = []
        
        for i, (px, py) in enumerate(path):
            if i == 0:
                dx = path[1][0] - px
                dy = path[1][1] - py
            elif i == len(path) - 1:
                dx = px - path[i-1][0]
                dy = py - path[i-1][1]
            else:
                dx = path[i+1][0] - path[i-1][0]
                dy = path[i+1][1] - path[i-1][1]
            
            length = math.sqrt(dx*dx + dy*dy)
            if length < 1e-6:
                continue
            
            perp_x = -dy / length
            perp_y = dx / length
            
            left_edge.append((px + perp_x * half_width, py + perp_y * half_width))
            right_edge.append((px - perp_x * half_width, py - perp_y * half_width))
        
        return left_edge + right_edge[::-1]
    
    def do_polygons_intersect(self, poly1: List[Tuple[float, float]], 
                               poly2: List[Tuple[float, float]]) -> bool:
        """
        Check if two convex polygons intersect using Separating Axis Theorem.
        
        Args:
            poly1: First polygon vertices
            poly2: Second polygon vertices
            
        Returns:
            True if polygons intersect
        """
        if not poly1 or not poly2 or len(poly1) < 3 or len(poly2) < 3:
            return False
        
        def get_edges(polygon):
            edges = []
            for i in range(len(polygon)):
                p1 = polygon[i]
                p2 = polygon[(i + 1) % len(polygon)]
                edges.append((p2[0] - p1[0], p2[1] - p1[1]))
            return edges
        
        def project_polygon(polygon, axis):
            min_proj = float('inf')
            max_proj = float('-inf')
            for p in polygon:
                proj = p[0] * axis[0] + p[1] * axis[1]
                min_proj = min(min_proj, proj)
                max_proj = max(max_proj, proj)
            return min_proj, max_proj
        
        # Check all edges of both polygons as potential separating axes
        for polygon in [poly1, poly2]:
            edges = get_edges(polygon)
            for edge in edges:
                # Normal to edge (perpendicular)
                axis = (-edge[1], edge[0])
                axis_len = (axis[0]**2 + axis[1]**2)**0.5
                if axis_len < 1e-6:
                    continue
                axis = (axis[0]/axis_len, axis[1]/axis_len)
                
                # Project both polygons onto axis
                min1, max1 = project_polygon(poly1, axis)
                min2, max2 = project_polygon(poly2, axis)
                
                # Check for gap (separating axis found)
                if max1 < min2 or max2 < min1:
                    return False  # Separated, no intersection
        
        return True  # No separating axis found, polygons intersect
    
    def is_strip_colliding_with_obb(self, path: List[Tuple[float, float]],
                                     strip_width: float,
                                     obb_corners: List[Tuple[float, float]]) -> bool:
        """
        Check if a strip (path with width) collides with an OBB.
        
        Args:
            path: List of (x, y) points
            strip_width: Width of the strip
            obb_corners: 4 corners of the OBB
            
        Returns:
            True if strip collides with OBB
        """
        strip_poly = self.get_strip_polygon(path, strip_width)
        if not strip_poly:
            return False
        
        return self.do_polygons_intersect(strip_poly, obb_corners)
    
    def calculate_strip_collision_penalty(self, path: List[Tuple[float, float]],
                                           strip_width: float,
                                           all_obbs: List[Dict],
                                           exclude_id: int = None) -> float:
        """
        Calculate collision penalty for a strip against all OBBs.
        
        Args:
            path: List of (x, y) points
            strip_width: Width of the strip
            all_obbs: List of OBB dictionaries
            exclude_id: Vehicle ID to exclude
            
        Returns:
            1.0 if collision, 0.0 otherwise
        """
        if not path or not all_obbs:
            return 0.0
        
        for obb_data in all_obbs:
            track_id = obb_data.get('track_id')
            obb_corners = obb_data.get('obb_corners')
            
            if exclude_id is not None and track_id == exclude_id:
                continue
            
            if not obb_corners or len(obb_corners) != 4:
                continue
            
            if self.is_strip_colliding_with_obb(path, strip_width, obb_corners):
                return 1.0
        
        return 0.0
