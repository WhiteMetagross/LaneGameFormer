"""
Road Mask Manager
Handles loading and rasterizing road annotations for trajectory constraint
"""
import json
import numpy as np
import cv2
from typing import Tuple, List, Optional


class RoadMaskManager:
    """
    Manages road mask data from JSON annotations.
    Provides road boundary checking and visualization overlay.
    """
    
    def __init__(self, road_mask_path: str, frame_width: int, frame_height: int):
        """
        Initialize the road mask manager.
        
        Args:
            road_mask_path: Path to road annotation JSON file
            frame_width: Video frame width
            frame_height: Video frame height
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.polygons = []
        self.road_mask = None
        self.distance_transform = None
        
        self._load_road_mask(road_mask_path)
        self._rasterize_mask()
        self._compute_distance_transform()
    
    def _load_road_mask(self, path: str):
        """Load road polygons from JSON annotation file."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            for polygon_data in data.get('polygons', []):
                vertices = polygon_data.get('vertices', [])
                poly_type = polygon_data.get('type', 'additive')
                
                if len(vertices) >= 6:  # At least 3 points (x,y pairs)
                    # Convert flat list to list of (x, y) tuples
                    points = []
                    for i in range(0, len(vertices), 2):
                        if i + 1 < len(vertices):
                            x = int(round(vertices[i]))
                            y = int(round(vertices[i + 1]))
                            # Clamp to frame boundaries
                            x = max(0, min(x, self.frame_width - 1))
                            y = max(0, min(y, self.frame_height - 1))
                            points.append((x, y))
                    
                    if len(points) >= 3:
                        self.polygons.append({
                            'points': np.array(points, dtype=np.int32),
                            'type': poly_type
                        })
            
            print(f"Loaded {len(self.polygons)} road polygons from annotation")
            
        except Exception as e:
            print(f"Error loading road mask: {e}")
            self.polygons = []
    
    def _rasterize_mask(self):
        """Rasterize road polygons into a continuous potential surface."""
        # Initialize as float32 for continuous values
        self.road_mask = np.zeros((self.frame_height, self.frame_width), dtype=np.float32)
        
        # Create temporary binary mask for polygons
        poly_mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        
        # First apply all additive polygons
        for polygon in self.polygons:
            if polygon['type'] == 'additive':
                cv2.fillPoly(poly_mask, [polygon['points']], 255)
        
        # Then apply subtractive polygons
        for polygon in self.polygons:
            if polygon['type'] == 'subtractive':
                cv2.fillPoly(poly_mask, [polygon['points']], 0)
        
        # Apply slight dilation to smooth edges
        kernel = np.ones((3, 3), np.uint8)
        poly_mask = cv2.dilate(poly_mask, kernel, iterations=1)
        
        # Convert to float and set base road value (e.g., 0.4)
        self.road_mask[poly_mask > 0] = 0.4
        
        road_pixels = np.sum(self.road_mask > 0)
        total_pixels = self.frame_width * self.frame_height
        print(f"Road mask coverage: {100 * road_pixels / total_pixels:.1f}%")

    def embed_lane_spline(self, lane_points: List[Tuple[float, float]], 
                          sigma: float = 15.0, 
                          weight: float = 0.6,
                          confidence_score: float = 1.0):
        """
        Embed a lane spline into the road mask as a Gaussian profile.
        
        Args:
            lane_points: List of (x, y) points defining the lane
            sigma: Standard deviation (width) of the lane influence
            weight: Maximum added value for the lane center
            confidence_score: Confidence score (0.0-1.0) to scale the weight
        """
        if not lane_points or len(lane_points) < 2:
            return
            
        # Adjust weight based on confidence
        # Base weight is 0.3, plus up to 0.7 based on confidence
        effective_weight = weight * (0.3 + 0.7 * confidence_score)
            
        # Create a temporary mask for this lane
        lane_mask = np.zeros((self.frame_height, self.frame_width), dtype=np.float32)
        
        # Draw the lane centerline
        pts = np.array([(int(x), int(y)) for x, y in lane_points], dtype=np.int32)
        cv2.polylines(lane_mask, [pts], False, 1.0, 1)
        
        # Apply Gaussian blur to create the profile
        # Kernel size should be large enough to cover 3*sigma
        ksize = int(sigma * 6) | 1  # Ensure odd
        lane_heatmap = cv2.GaussianBlur(lane_mask, (ksize, ksize), sigma)
        
        # Normalize so the peak is roughly 1.0 (GaussianBlur reduces peak)
        # But simpler: just scale it up. The integral of Gaussian is 1, but peak depends on sigma.
        # Instead of exact math, let's just normalize the peak to 'weight'
        if lane_heatmap.max() > 0:
            lane_heatmap = lane_heatmap / lane_heatmap.max() * effective_weight
            
        # Add to main mask
        # We add to the base road value (0.4)
        # Using maximum ensures we don't double-count overlapping lanes, 
        # but rather take the "strongest" flow definition at any point.
        current_base = (self.road_mask > 0) * 0.4
        self.road_mask = np.maximum(self.road_mask, lane_heatmap + current_base)
        
        # Clip to 1.0
        self.road_mask = np.clip(self.road_mask, 0.0, 1.0)

    def get_road_score(self, x: float, y: float) -> float:
        """Get the navigation potential score at a point."""
        ix, iy = int(round(x)), int(round(y))
        if ix < 0 or ix >= self.frame_width or iy < 0 or iy >= self.frame_height:
            return 0.0
        return float(self.road_mask[iy, ix])
    
    def _compute_distance_transform(self):
        """Compute distance transform for road mask (distance to nearest boundary)."""
        if self.road_mask is not None:
            # Create binary mask for distance transform
            binary_mask = (self.road_mask > 0).astype(np.uint8)
            self.distance_transform = cv2.distanceTransform(
                binary_mask, cv2.DIST_L2, 5
            )
    
    def is_on_road(self, x: float, y: float, margin: float = 0) -> bool:
        """
        Check if a point is on the road.
        """
        ix, iy = int(round(x)), int(round(y))
        
        if ix < 0 or ix >= self.frame_width or iy < 0 or iy >= self.frame_height:
            return False
        
        if self.road_mask is None:
            return True
        
        # For float mask, check if value > 0
        if margin <= 0:
            return self.road_mask[iy, ix] > 0.0
        
        if self.distance_transform is not None:
            return self.distance_transform[iy, ix] >= margin
        
        return self.road_mask[iy, ix] > 0.0
    
    def is_point_on_road(self, point: Tuple[float, float], margin: float = 0) -> bool:
        """
        Check if a point (x, y) tuple is on the road.
        Convenience wrapper for is_on_road.
        
        Args:
            point: (x, y) coordinate tuple
            margin: Minimum distance from road edge required
            
        Returns:
            True if point is on road with sufficient margin
        """
        return self.is_on_road(point[0], point[1], margin)
    
    def is_path_on_road(self, path: List[Tuple[float, float]], margin: float = 0) -> Tuple[bool, float]:
        """
        Check if an entire path stays on the road.
        
        Args:
            path: List of (x, y) points
            margin: Minimum distance from road edge required
            
        Returns:
            Tuple of (is_valid, on_road_fraction)
        """
        if not path:
            return False, 0.0
        
        on_road_count = 0
        for x, y in path:
            if self.is_on_road(x, y, margin):
                on_road_count += 1
        
        fraction = on_road_count / len(path)
        return fraction >= 0.95, fraction  # 95% threshold for valid path
    
    def get_distance_to_boundary(self, x: float, y: float) -> float:
        """
        Get distance to nearest road boundary.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Distance to road boundary (negative if off-road)
        """
        ix, iy = int(round(x)), int(round(y))
        
        if ix < 0 or ix >= self.frame_width or iy < 0 or iy >= self.frame_height:
            return -100.0  # Far off-screen
        
        if self.distance_transform is None:
            return 100.0  # No mask
        
        if self.road_mask[iy, ix] > 0:
            return self.distance_transform[iy, ix]
        else:
            # Off road - compute negative distance
            inv_mask = 255 - self.road_mask
            inv_dist = cv2.distanceTransform(inv_mask, cv2.DIST_L2, 5)
            return -inv_dist[iy, ix]
    
    def get_road_mask_overlay(self, alpha: float = 0.3, 
                               color: Tuple[int, int, int] = (100, 150, 100)) -> np.ndarray:
        """
        Get a colored overlay of the road mask (heatmap style).
        """
        overlay = np.zeros((self.frame_height, self.frame_width, 4), dtype=np.uint8)
        
        if self.road_mask is not None:
            # Normalize mask to 0-255
            mask_norm = (self.road_mask * 255).astype(np.uint8)
            
            # Apply colormap (JET for heatmap)
            heatmap = cv2.applyColorMap(mask_norm, cv2.COLORMAP_JET)
            
            # Set alpha based on mask value
            alpha_channel = (self.road_mask * 255 * alpha).astype(np.uint8)
            
            overlay[:, :, :3] = heatmap
            overlay[:, :, 3] = alpha_channel
        
        return overlay
    
    def apply_overlay(self, frame: np.ndarray, alpha: float = 0.3,
                      color: Tuple[int, int, int] = (100, 150, 100)) -> np.ndarray:
        """
        Apply road mask overlay to a frame.
        
        Args:
            frame: Input BGR frame
            alpha: Transparency (0-1)
            color: BGR color for road area
            
        Returns:
            Frame with road overlay
        """
        if self.road_mask is None:
            return frame
        
        result = frame.copy()
        
        # Create colored overlay
        overlay = np.zeros_like(frame)
        overlay[self.road_mask > 0] = color
        
        # Blend with original frame
        mask_float = (self.road_mask > 0).astype(np.float32) * alpha
        mask_3d = np.repeat(mask_float[:, :, np.newaxis], 3, axis=2)
        
        result = (result * (1 - mask_3d) + overlay * mask_3d).astype(np.uint8)
        
        return result
    
    def get_road_boundary_points(self) -> List[np.ndarray]:
        """
        Get the boundary contours of the road mask.
        
        Returns:
            List of contour point arrays
        """
        if self.road_mask is None:
            return []
        
        contours, _ = cv2.findContours(
            self.road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        return contours
