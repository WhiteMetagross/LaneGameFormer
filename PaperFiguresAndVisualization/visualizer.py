"""
Visualization System
Renders trajectory predictions, social potential fields, and road masks
"""
import cv2
import numpy as np
import math
from typing import List, Tuple, Dict, Optional

import config


class Visualizer:
    """
    Visualization system for trajectory predictions and related overlays.
    """
    
    def __init__(self, frame_width: int, frame_height: int):
        """
        Initialize the visualizer.
        
        Args:
            frame_width: Video frame width
            frame_height: Video frame height
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
    
    def draw_road_overlay(self, frame: np.ndarray, 
                          road_mask: np.ndarray,
                          alpha: float = None,
                          color: Tuple[int, int, int] = None) -> np.ndarray:
        """
        Draw road mask overlay as a heatmap representing Traffic Flow Potential.
        
        Args:
            frame: Input BGR frame (original video)
            road_mask: Continuous road mask (0.0 to 1.0)
            alpha: Max transparency (0-1)
            color: Ignored (uses colormap)
            
        Returns:
            Frame with heatmap overlay
        """
        if alpha is None:
            alpha = 0.3  # Visible but transparent
        
        if road_mask is None:
            return frame
        
        result = frame.copy()
        
        # Normalize mask to 0-255
        mask_norm = (np.clip(road_mask, 0, 1.0) * 255).astype(np.uint8)
        
        # Apply colormap (JET for heatmap: Blue=Low, Red=High)
        # We want low values (base road) to be subtle, high values (lanes) to be distinct
        heatmap = cv2.applyColorMap(mask_norm, cv2.COLORMAP_JET)
        
        # Create alpha channel based on mask intensity
        # Base road (0.4) gets lower alpha, Lanes (1.0) get higher alpha
        alpha_mask = (road_mask * alpha).astype(np.float32)
        alpha_mask = np.clip(alpha_mask, 0, 0.6) # Cap opacity
        
        # Expand alpha to 3 channels
        alpha_3ch = np.dstack([alpha_mask, alpha_mask, alpha_mask])
        
        # Blend heatmap with frame
        # Only apply where mask > 0
        mask_bool = road_mask > 0.01
        
        # Vectorized blending
        result = (result * (1 - alpha_3ch) + heatmap * alpha_3ch).astype(np.uint8)
        
        # Draw subtle road boundary outline
        contours, _ = cv2.findContours(
            (road_mask > 0.1).astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result, contours, -1, (100, 100, 100), 1)
        
        return result
    
    def draw_social_potential_bands(self, frame: np.ndarray,
                                     vehicles: List[Dict],
                                     alpha: float = None,
                                     use_gradient: bool = True) -> np.ndarray:
        """
        Draw social potential bands around vehicles as heatmap with smooth gradient.
        Closer to center = more critical/dangerous (red), farther = safer (green/blue).
        
        Args:
            frame: Input BGR frame
            vehicles: List of vehicle states with positions
            alpha: Base transparency
            use_gradient: If True, use smooth gradient; otherwise use discrete bands
            
        Returns:
            Frame with social potential overlay
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
            if len(local_x) == 0 or len(local_y) == 0:
                continue
            xx, yy = np.meshgrid(local_x, local_y)
            dist_map = np.sqrt((xx - vx)**2 + (yy - vy)**2)
            
            # Calculate potential (exponential decay - closer = higher potential)
            vehicle_potential = np.exp(-dist_map / (max_radius * 0.3))
            vehicle_potential[dist_map > max_radius] = 0
            
            # Add to combined map using maximum (overlapping fields)
            potential_map[y1:y2, x1:x2] = np.maximum(
                potential_map[y1:y2, x1:x2],
                vehicle_potential
            )
        
        # Create mask for areas with potential
        mask = potential_map > 0.03  # Lower threshold to show more of the gradient
        if not mask.any():
            return result
        
        # Normalize to 0-255 for colormap with boosted visibility
        potential_norm = np.zeros_like(potential_map, dtype=np.uint8)
        # Boost the potential values for more visible gradient
        boosted_potential = np.clip(potential_map[mask] * 1.5, 0, 1.0)
        potential_norm[mask] = np.clip(boosted_potential * 255, 0, 255).astype(np.uint8)
        
        # Apply colormap (JET: Blue->Cyan->Green->Yellow->Red)
        # High potential = Red (danger), Low potential = Blue (safe)
        heatmap = cv2.applyColorMap(potential_norm, cv2.COLORMAP_JET)
        
        # Create smooth alpha mask based on potential strength - more opaque
        alpha_mask = (potential_map * alpha * 2.0).astype(np.float32)  # Increased boost for visibility
        alpha_mask = np.clip(alpha_mask, 0, 0.85)  # Higher cap for more opacity
        # Add minimum alpha for visible areas to ensure gradient is always visible
        alpha_mask[mask] = np.maximum(alpha_mask[mask], 0.15)
        alpha_3ch = np.dstack([alpha_mask, alpha_mask, alpha_mask])
        
        # Blend heatmap with frame where there's potential
        result = (result * (1 - alpha_3ch) + heatmap * alpha_3ch).astype(np.uint8)
        
        return result
    
    def draw_trajectory_path(self, frame: np.ndarray,
                              path: List[Tuple[float, float]],
                              color: Tuple[int, int, int],
                              probability: float = 1.0,
                              thickness: int = None,
                              strip_width: float = 20.0,
                              alpha: float = 0.4,
                              draw_arrow: bool = True) -> np.ndarray:
        """
        Draw a predicted trajectory as a semi-transparent strip (thick polygon)
        AND a centerline with an arrowhead.
        
        Args:
            frame: Input BGR frame
            path: List of (x, y) points
            color: BGR color
            probability: Path probability (affects opacity)
            thickness: Deprecated - use strip_width
            strip_width: Width of the strip in pixels
            alpha: Transparency of the strip (0-1)
            draw_arrow: Whether to draw the centerline arrow
            
        Returns:
            Frame with strip and arrow drawn
        """
        if not path or len(path) < 2:
            return frame
        
        # Filter valid points
        valid_points = []
        for x, y in path:
            ix, iy = int(round(x)), int(round(y))
            if 0 <= ix < self.frame_width and 0 <= iy < self.frame_height:
                valid_points.append((float(x), float(y)))
        
        if len(valid_points) < 2:
            return frame
        
        result = frame.copy()
        half_width = strip_width / 2.0
        
        # --- 1. Draw Strip Polygon ---
        # Build the strip polygon by creating left and right edges
        left_edge = []
        right_edge = []
        
        for i, (px, py) in enumerate(valid_points):
            # Calculate perpendicular direction at this point
            if i == 0:
                # Use direction to next point
                dx = valid_points[1][0] - px
                dy = valid_points[1][1] - py
            elif i == len(valid_points) - 1:
                # Use direction from previous point
                dx = px - valid_points[i-1][0]
                dy = py - valid_points[i-1][1]
            else:
                # Average of directions
                dx = valid_points[i+1][0] - valid_points[i-1][0]
                dy = valid_points[i+1][1] - valid_points[i-1][1]
            
            # Normalize and get perpendicular
            length = math.sqrt(dx*dx + dy*dy)
            if length < 1e-6:
                continue
            
            # Perpendicular direction (90 degrees rotation)
            perp_x = -dy / length
            perp_y = dx / length
            
            # Create left and right points
            left_edge.append((px + perp_x * half_width, py + perp_y * half_width))
            right_edge.append((px - perp_x * half_width, py - perp_y * half_width))
        
        if len(left_edge) >= 2:
            # Create closed polygon: left edge forward, right edge backward
            polygon_points = left_edge + right_edge[::-1]
            polygon_pts = np.array([(int(round(x)), int(round(y))) for x, y in polygon_points], dtype=np.int32)
            
            # Create overlay for semi-transparent drawing
            overlay = result.copy()
            
            # Draw filled polygon on overlay
            cv2.fillPoly(overlay, [polygon_pts], color)
            
            # Blend overlay with result using alpha
            effective_alpha = alpha * (0.7 + 0.3 * probability)
            cv2.addWeighted(overlay, effective_alpha, result, 1 - effective_alpha, 0, result)
            
            # Draw thin outline for definition
            cv2.polylines(result, [polygon_pts], isClosed=True, color=(0, 0, 0), thickness=1)
        
        # --- 2. Draw Centerline Arrow ---
        if draw_arrow:
            # Draw centerline
            pts = np.array([(int(round(x)), int(round(y))) for x, y in valid_points], dtype=np.int32)
            cv2.polylines(result, [pts], False, color, 2)
            
            # Draw arrowhead at the end
            end_pt = valid_points[-1]
            prev_pt = valid_points[-2]
            
            dx = end_pt[0] - prev_pt[0]
            dy = end_pt[1] - prev_pt[1]
            length = math.sqrt(dx*dx + dy*dy)
            
            if length > 0:
                # Normalize direction
                dx /= length
                dy /= length
                
                # Arrow head parameters
                arrow_len = 15
                arrow_angle = 0.5  # radians
                
                # Calculate arrow wing points
                angle = math.atan2(dy, dx)
                
                x1 = end_pt[0] - arrow_len * math.cos(angle - arrow_angle)
                y1 = end_pt[1] - arrow_len * math.sin(angle - arrow_angle)
                
                x2 = end_pt[0] - arrow_len * math.cos(angle + arrow_angle)
                y2 = end_pt[1] - arrow_len * math.sin(angle + arrow_angle)
                
                # Draw arrow head
                arrow_pts = np.array([
                    (int(end_pt[0]), int(end_pt[1])),
                    (int(x1), int(y1)),
                    (int(x2), int(y2))
                ], dtype=np.int32)
                
                cv2.fillPoly(result, [arrow_pts], color)
        
        return result
    
    def draw_top_predictions(self, frame: np.ndarray,
                              predictions: List[Dict],
                              max_paths: int = None,
                              strip_width: float = 20.0) -> np.ndarray:
        """
        Draw top predicted paths as semi-transparent strips.
        
        Args:
            frame: Input BGR frame
            predictions: List of prediction dictionaries
            max_paths: Maximum paths to draw
            strip_width: Width of prediction strips
            
        Returns:
            Frame with predictions drawn
        """
        if max_paths is None:
            max_paths = config.NUM_PREDICTION_PATHS
        
        result = frame.copy()
        
        for i, pred in enumerate(predictions[:max_paths]):
            if i < len(config.PATH_COLORS):
                color = config.PATH_COLORS[i]
            else:
                color = (128, 128, 128)
            
            probability = pred.get('probability', 1.0)
            path = pred.get('path_points', [])
            
            if path and probability > config.MIN_PATH_PROBABILITY:
                result = self.draw_trajectory_path(
                    result, path, color, probability,
                    strip_width=strip_width,
                    alpha=0.4  # Semi-transparent
                )
        
        return result
    
    def draw_vehicle_trail(self, frame: np.ndarray,
                           history: List[Tuple[float, float]],
                           color: Tuple[int, int, int],
                           max_length: int = None) -> np.ndarray:
        """
        Draw vehicle position history trail.
        
        Args:
            frame: Input BGR frame
            history: List of (x, y) positions
            color: BGR color
            max_length: Maximum trail length
            
        Returns:
            Frame with trail drawn
        """
        if not history:
            return frame
        
        if max_length is None:
            max_length = config.TRAIL_LENGTH
        
        trail = history[-max_length:]
        
        if len(trail) < 2:
            return frame
        
        result = frame.copy()
        
        # Draw with fading effect
        for i in range(1, len(trail)):
            alpha = i / len(trail)  # Fade from old to new
            pt1 = (int(trail[i-1][0]), int(trail[i-1][1]))
            pt2 = (int(trail[i][0]), int(trail[i][1]))
            
            # Check bounds
            if not (0 <= pt1[0] < self.frame_width and 0 <= pt1[1] < self.frame_height):
                continue
            if not (0 <= pt2[0] < self.frame_width and 0 <= pt2[1] < self.frame_height):
                continue
            
            # Adjust color based on alpha
            faded_color = tuple(int(c * alpha) for c in color)
            cv2.line(result, pt1, pt2, faded_color, config.TRAIL_THICKNESS)
        
        return result
    
    def draw_vehicle_marker(self, frame: np.ndarray,
                            position: Tuple[float, float],
                            direction: Tuple[float, float] = None,
                            color: Tuple[int, int, int] = (0, 0, 255),
                            size: int = 8) -> np.ndarray:
        """
        Draw a vehicle position marker with optional direction.
        
        Args:
            frame: Input BGR frame
            position: (x, y) position
            direction: Optional direction vector
            color: BGR color
            size: Marker size
            
        Returns:
            Frame with marker drawn
        """
        result = frame.copy()
        
        px, py = int(round(position[0])), int(round(position[1]))
        
        if not (0 <= px < self.frame_width and 0 <= py < self.frame_height):
            return result
        
        # Draw center circle with outline for visibility
        cv2.circle(result, (px, py), size + 2, (0, 0, 0), -1)  # Black outline
        cv2.circle(result, (px, py), size, color, -1)  # Colored fill
        cv2.circle(result, (px, py), size, (255, 255, 255), 2)  # White border
        
        # Draw direction indicator
        if direction is not None:
            dx, dy = direction
            if isinstance(direction, np.ndarray):
                dx, dy = float(dx), float(dy)
            
            arrow_len = size * 3
            
            end_x = int(px + dx * arrow_len)
            end_y = int(py + dy * arrow_len)
            
            cv2.arrowedLine(result, (px, py), (end_x, end_y), (0, 0, 0), 4, tipLength=0.3)
            cv2.arrowedLine(result, (px, py), (end_x, end_y), color, 2, tipLength=0.3)
        
        return result
    
    def draw_lanes(self, frame: np.ndarray,
                   lanes: List[Dict],
                   color: Tuple[int, int, int] = (80, 80, 80),
                   connection_color: Tuple[int, int, int] = (120, 120, 80)) -> np.ndarray:
        """
        Draw lane network on frame.
        
        Args:
            frame: Input BGR frame
            lanes: List of lane dictionaries
            color: Color for main lanes
            connection_color: Color for lane connections
            
        Returns:
            Frame with lanes drawn
        """
        result = frame.copy()
        
        for lane in lanes:
            coords = lane.get('coords', [])
            if len(coords) < 2:
                continue
            
            # Filter valid points
            valid_pts = []
            for x, y in coords:
                ix, iy = int(round(x)), int(round(y))
                if 0 <= ix < self.frame_width and 0 <= iy < self.frame_height:
                    valid_pts.append((ix, iy))
            
            if len(valid_pts) < 2:
                continue
            
            pts_array = np.array(valid_pts, dtype=np.int32)
            
            if lane.get('is_connection', False):
                cv2.polylines(result, [pts_array], False, connection_color, 1)
            else:
                cv2.polylines(result, [pts_array], False, color, 1)
        
        return result
    
    def draw_info_overlay(self, frame: np.ndarray,
                          frame_num: int,
                          num_tracked: int,
                          num_predictions: int,
                          additional_info: Dict = None) -> np.ndarray:
        """
        Draw information overlay text.
        
        Args:
            frame: Input BGR frame
            frame_num: Current frame number
            num_tracked: Number of tracked vehicles
            num_predictions: Number of active predictions
            additional_info: Additional info to display
            
        Returns:
            Frame with info overlay
        """
        result = frame.copy()
        
        y_offset = 30
        line_height = 25
        
        info_lines = [
            f"Frame: {frame_num}",
            f"Tracked: {num_tracked}",
            f"Predictions: {num_predictions}"
        ]
        
        if additional_info:
            for key, value in additional_info.items():
                info_lines.append(f"{key}: {value}")
        
        for i, line in enumerate(info_lines):
            y = y_offset + i * line_height
            cv2.putText(result, line, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(result, line, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return result
    
    def draw_legend(self, frame: np.ndarray,
                    position: str = 'bottom_right') -> np.ndarray:
        """
        Draw color legend for predictions.
        
        Args:
            frame: Input BGR frame
            position: Legend position ('bottom_right', 'top_right')
            
        Returns:
            Frame with legend
        """
        result = frame.copy()
        
        legend_items = [
            ("Best path", config.PATH_COLORS[0]),
            ("2nd best", config.PATH_COLORS[1]),
            ("3rd best", config.PATH_COLORS[2]),
            ("4th best", config.PATH_COLORS[3]),
        ]
        
        line_height = 20
        box_width = 140
        box_height = len(legend_items) * line_height + 15
        
        if position == 'bottom_right':
            x_start = self.frame_width - box_width - 10
            y_start = self.frame_height - box_height - 10
        else:  # top_right
            x_start = self.frame_width - box_width - 10
            y_start = 10
        
        # Draw background
        cv2.rectangle(result, (x_start, y_start), 
                     (x_start + box_width, y_start + box_height),
                     (40, 40, 40), -1)
        cv2.rectangle(result, (x_start, y_start),
                     (x_start + box_width, y_start + box_height),
                     (100, 100, 100), 1)
        
        # Draw items
        for i, (label, color) in enumerate(legend_items):
            y = y_start + 15 + i * line_height
            
            # Color box
            cv2.rectangle(result, (x_start + 5, y - 10),
                         (x_start + 20, y + 2), color, -1)
            
            # Label
            cv2.putText(result, label, (x_start + 25, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return result
    
    def create_composite_visualization(self, frame: np.ndarray,
                                        road_mask: np.ndarray,
                                        vehicles: List[Dict],
                                        predictions: Dict[int, List[Dict]],
                                        lanes: List[Dict],
                                        frame_num: int,
                                        detections: List[Dict] = None) -> np.ndarray:
        """
        Create a complete composite visualization with OBBs, tracking IDs,
        gradient social potential field, and subtle yellow road mask.
        
        Args:
            frame: Input BGR frame
            road_mask: Binary road mask
            vehicles: List of vehicle states
            predictions: Dictionary of predictions per vehicle
            lanes: List of lane dictionaries
            frame_num: Current frame number
            detections: Optional list of detections with OBB data
            
        Returns:
            Complete visualized frame
        """
        result = frame.copy()
        
        # 1. Road mask overlay (subtle light yellow)
        result = self.draw_road_overlay(result, road_mask)
        
        # 2. Lane network - DISABLED as per request to show flow regions instead
        # result = self.draw_lanes(result, lanes)
        
        # 3. Social potential bands (gradient heatmap)
        result = self.draw_social_potential_bands(result, vehicles)
        
        # 4. Vehicle predictions (no trails - just prediction strips)
        for vehicle in vehicles:
            vid = vehicle.get('id')
            
            # Draw predictions only (no trail)
            if vid in predictions:
                preds = predictions[vid]
                # Get strip width from first prediction or use default
                strip_width = 20.0
                if preds and len(preds) > 0:
                    strip_width = preds[0].get('strip_width', 20.0)
                result = self.draw_top_predictions(result, preds, strip_width=strip_width)
        
        # 5. Draw OBBs for each vehicle with unique colors (directly from detections)
        # OBBs are drawn from raw detection data, not from vehicle state, so they don't "move"
        if detections:
            for det in detections:
                track_id = det.get('track_id')
                obb_corners = det.get('obb_corners')
                
                if obb_corners and len(obb_corners) == 4:
                    color = self._get_vehicle_color(track_id)
                    result = self.draw_vehicle_obb(result, obb_corners, color, thickness=2)
        
        # 6. Draw tracking IDs and vehicle markers
        for vehicle in vehicles:
            vid = vehicle.get('id')
            color = self._get_vehicle_color(vid)
            
            # Draw marker
            result = self.draw_vehicle_marker(
                result, vehicle['position'],
                vehicle.get('direction'), color
            )
            
            # Draw tracking ID
            result = self.draw_tracking_id(
                result, vehicle['position'], vid, color
            )
        
        # 7. Info overlay
        result = self.draw_info_overlay(
            result, frame_num, len(vehicles),
            sum(len(p) for p in predictions.values())
        )
        
        # 8. Legend
        result = self.draw_legend(result)
        
        return result
    
    def _get_vehicle_color(self, vehicle_id: int) -> Tuple[int, int, int]:
        """Generate consistent color for vehicle ID."""
        import colorsys
        
        hue = (vehicle_id * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        return (int(b * 255), int(g * 255), int(r * 255))
    
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
    
    def draw_vehicle_obb(self, frame: np.ndarray,
                         obb_corners: List[Tuple[float, float]],
                         color: Tuple[int, int, int],
                         thickness: int = 2) -> np.ndarray:
        """
        Draw oriented bounding box for a vehicle.
        
        Args:
            frame: Input BGR frame
            obb_corners: List of 4 corner points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
            color: BGR color for the OBB
            thickness: Line thickness
            
        Returns:
            Frame with OBB drawn
        """
        if not obb_corners or len(obb_corners) != 4:
            return frame
        
        result = frame.copy()
        
        # Convert corners to integer points
        pts = np.array([(int(round(x)), int(round(y))) for x, y in obb_corners], dtype=np.int32)
        
        # Draw the OBB polygon
        cv2.polylines(result, [pts], isClosed=True, color=color, thickness=thickness)
        
        # Draw slightly thicker outline for visibility
        cv2.polylines(result, [pts], isClosed=True, color=(0, 0, 0), thickness=thickness + 1)
        cv2.polylines(result, [pts], isClosed=True, color=color, thickness=thickness)
        
        return result
    
    def draw_tracking_id(self, frame: np.ndarray,
                         position: Tuple[float, float],
                         track_id: int,
                         color: Tuple[int, int, int],
                         font_scale: float = 0.6,
                         offset_y: int = -15) -> np.ndarray:
        """
        Draw tracking ID label near a vehicle.
        
        Args:
            frame: Input BGR frame
            position: (x, y) position to place label
            track_id: Tracking ID to display
            color: BGR color for the text
            font_scale: Font size scale
            offset_y: Vertical offset from position (negative = above)
            
        Returns:
            Frame with tracking ID drawn
        """
        result = frame.copy()
        
        px, py = int(round(position[0])), int(round(position[1]))
        label = f"ID:{track_id}"
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
        )
        
        # Position for text (above the vehicle)
        text_x = px - text_width // 2
        text_y = py + offset_y
        
        # Ensure text stays within frame bounds
        text_x = max(5, min(text_x, self.frame_width - text_width - 5))
        text_y = max(text_height + 5, min(text_y, self.frame_height - 5))
        
        # Draw background rectangle for readability
        padding = 3
        cv2.rectangle(
            result,
            (text_x - padding, text_y - text_height - padding),
            (text_x + text_width + padding, text_y + padding),
            (0, 0, 0), -1
        )
        
        # Draw text with outline for visibility
        cv2.putText(result, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3)
        cv2.putText(result, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
        
        return result
    
    def draw_road_overlay_yellow(self, frame: np.ndarray,
                                  road_mask: np.ndarray,
                                  alpha: float = 0.15) -> np.ndarray:
        """
        Draw semi-transparent road mask overlay in subtle yellow.
        
        Args:
            frame: Input BGR frame
            road_mask: Binary road mask
            alpha: Transparency (0-1), default very subtle
            
        Returns:
            Frame with subtle yellow road overlay
        """
        if road_mask is None:
            return frame
        
        result = frame.copy()
        
        # Subtle light yellow color (BGR format)
        yellow_color = np.array([200, 255, 255], dtype=np.uint8)  # Light yellow in BGR
        
        # Create overlay with yellow tint on road areas
        overlay = result.copy()
        overlay[road_mask > 0] = (
            overlay[road_mask > 0] * (1 - alpha) + 
            yellow_color * alpha
        ).astype(np.uint8)
        
        result = overlay
        
        # Optional: Draw subtle road boundary outline
        contours, _ = cv2.findContours(
            road_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result, contours, -1, (180, 220, 220), 1)  # Subtle yellow outline
        
        return result
