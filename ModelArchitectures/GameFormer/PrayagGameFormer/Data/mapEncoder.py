"""
Map encoding for PrayagGameFormer.

This module handles the conversion of skeletonized road masks to
GameFormer-compatible lane and crosswalk encodings.

Following the "Graph Trap" strategy: Simple skeletonization without
sophisticated lane connection heuristics.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    import cv2
except ImportError:
    cv2 = None


class MapEncoder:
    """
    Encodes skeletonized road masks into GameFormer lane/crosswalk format.
    
    GameFormer lane format per point (16 dims):
        - self_line (3): x, y, heading
        - left_line (3): x, y, heading  
        - right_line (3): x, y, heading
        - speed_limit (1): normalized speed
        - self_type (1): lane type
        - left_type (1): left boundary type
        - right_type (1): right boundary type  
        - traffic_light (1): traffic light state
        - interpolating (1): is interpolated
        - stop_sign (1): has stop sign
    
    For Prayag dataset, we use simplified encoding since we don't have
    full HD map data - only skeletonized road masks.
    
    Memory Optimization: Uses LRU cache with bounded size to prevent OOM.
    """
    
    # Maximum cache entries (prevents unbounded memory growth)
    MAX_CACHE_SIZE = 100
    
    def __init__(
        self,
        config: dict,
        cacheDir: Optional[str] = None
    ):
        """
        Initialize map encoder.
        
        Args:
            config: Configuration dictionary
            cacheDir: Directory to cache encoded maps
        """
        self.config = config
        self.numLanes = config.get("numLanes", 50)
        self.numCrosswalks = config.get("numCrosswalks", 10)
        self.lanePoints = config.get("lanePoints", 100)
        self.crosswalkPoints = config.get("crosswalkPoints", 10)
        
        # Cache
        self.cacheDir = Path(cacheDir) if cacheDir else None
        if self.cacheDir:
            self.cacheDir.mkdir(parents=True, exist_ok=True)
        
        # LRU cache with bounded size
        self.cache = {}
        self.cacheOrder = []  # Track insertion order for LRU eviction
    
    def _evictIfNeeded(self) -> None:
        """Evict oldest cache entries if cache exceeds max size."""
        while len(self.cache) > self.MAX_CACHE_SIZE and self.cacheOrder:
            oldestKey = self.cacheOrder.pop(0)
            if oldestKey in self.cache:
                del self.cache[oldestKey]
    
    def loadAnnotation(self, annotationPath: str) -> dict:
        """
        Load annotation JSON file.
        
        Args:
            annotationPath: Path to annotation JSON
            
        Returns:
            Annotation dictionary
        """
        with open(annotationPath, "r") as f:
            return json.load(f)
    
    def skeletonize(self, polygons: List[dict], imageShape: Tuple[int, int]) -> np.ndarray:
        """
        Skeletonize road polygons using Zhang-Suen algorithm.
        
        Args:
            polygons: List of polygon dictionaries with "points" key
            imageShape: (height, width) of the image
            
        Returns:
            Binary skeleton image
        """
        if cv2 is None:
            raise ImportError("OpenCV not available for skeletonization")
        
        # Create binary mask
        mask = np.zeros(imageShape, dtype=np.uint8)
        
        for poly in polygons:
            points = np.array(poly.get("points", []), dtype=np.int32)
            if len(points) >= 3:
                cv2.fillPoly(mask, [points], 255)
        
        # Skeletonize using Zhang-Suen algorithm
        if hasattr(cv2, "ximgproc"):
            skeleton = cv2.ximgproc.thinning(
                mask, 
                thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
            )
        else:
            # Fallback: morphological skeletonization
            skeleton = np.zeros_like(mask)
            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            while True:
                opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, element)
                temp = cv2.subtract(mask, opened)
                eroded = cv2.erode(mask, element)
                skeleton = cv2.bitwise_or(skeleton, temp)
                mask = eroded.copy()
                if cv2.countNonZero(mask) == 0:
                    break
        
        return skeleton
    
    def extractCenterlines(
        self,
        skeleton: np.ndarray,
        maxLanes: int
    ) -> List[np.ndarray]:
        """
        Extract centerlines from skeleton image.
        
        Uses connected components to find separate lane segments.
        
        Args:
            skeleton: Binary skeleton image
            maxLanes: Maximum number of lanes to extract
            
        Returns:
            List of centerline arrays, each (N, 2)
        """
        if cv2 is None:
            return []
        
        # Find contours of skeleton
        contours, _ = cv2.findContours(
            skeleton,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_NONE
        )
        
        centerlines = []
        for contour in contours:
            if len(contour) < 5:
                continue
            
            # Reshape contour to (N, 2)
            points = contour.reshape(-1, 2).astype(np.float32)
            
            # Sample points if too long
            if len(points) > self.lanePoints:
                indices = np.linspace(0, len(points) - 1, self.lanePoints, dtype=int)
                points = points[indices]
            
            centerlines.append(points)
            
            if len(centerlines) >= maxLanes:
                break
        
        return centerlines
    
    def encodeLanes(
        self,
        centerlines: List[np.ndarray],
        agentPosition: np.ndarray
    ) -> np.ndarray:
        """
        Encode centerlines to GameFormer lane format.
        
        Args:
            centerlines: List of centerline arrays
            agentPosition: (x, y) position of the agent for relative encoding
            
        Returns:
            Encoded lanes (numLanes, lanePoints, 16)
        """
        lanes = np.zeros(
            (self.numLanes, self.lanePoints, 16),
            dtype=np.float32
        )
        
        for i, cl in enumerate(centerlines[:self.numLanes]):
            numPoints = min(len(cl), self.lanePoints)
            
            # Compute relative positions
            relPos = cl[:numPoints] - agentPosition
            
            # Compute headings from tangent vectors
            if numPoints > 1:
                tangent = np.diff(cl[:numPoints], axis=0)
                headings = np.arctan2(tangent[:, 1], tangent[:, 0])
                headings = np.pad(headings, (0, 1), mode="edge")
            else:
                headings = np.zeros(numPoints)
            
            # Fill lane encoding
            # self_line (x, y, heading)
            lanes[i, :numPoints, 0] = relPos[:, 0]  # x
            lanes[i, :numPoints, 1] = relPos[:, 1]  # y
            lanes[i, :numPoints, 2] = headings[:numPoints]  # heading
            
            # left_line and right_line (simplified - offset from centerline)
            offset = 1.8  # Approximate lane half-width
            perpendicular = np.stack([
                -np.sin(headings[:numPoints]),
                np.cos(headings[:numPoints])
            ], axis=-1)
            
            leftLine = relPos + offset * perpendicular
            rightLine = relPos - offset * perpendicular
            
            lanes[i, :numPoints, 3:5] = leftLine
            lanes[i, :numPoints, 5] = headings[:numPoints]
            lanes[i, :numPoints, 6:8] = rightLine
            lanes[i, :numPoints, 8] = headings[:numPoints]
            
            # speed_limit (normalized, default 1.0)
            lanes[i, :numPoints, 9] = 1.0
            
            # Types (all set to 1 = regular lane)
            lanes[i, :numPoints, 10] = 1  # self_type
            lanes[i, :numPoints, 11] = 1  # left_type
            lanes[i, :numPoints, 12] = 1  # right_type
            
            # traffic_light, interpolating, stop_sign (all 0)
            # Already initialized to 0
        
        return lanes
    
    def encodeCrosswalks(self, agentPosition: np.ndarray) -> np.ndarray:
        """
        Encode crosswalks (placeholder - no crosswalk data in Prayag dataset).
        
        Args:
            agentPosition: (x, y) position of the agent
            
        Returns:
            Encoded crosswalks (numCrosswalks, crosswalkPoints, 3)
        """
        return np.zeros(
            (self.numCrosswalks, self.crosswalkPoints, 3),
            dtype=np.float32
        )
    
    def encode(
        self,
        annotationPath: str,
        agentPosition: np.ndarray,
        frameIdx: int,
        imageShape: Tuple[int, int] = (1080, 1920)
    ) -> Dict[str, np.ndarray]:
        """
        Encode map features for GameFormer.
        
        Args:
            annotationPath: Path to annotation JSON
            agentPosition: (x, y) position of the agent
            frameIdx: Frame index (for caching)
            imageShape: Image dimensions
            
        Returns:
            Dictionary with "lanes" and "crosswalks" keys
        """
        # Check cache
        cacheKey = annotationPath
        if cacheKey in self.cache:
            centerlines = self.cache[cacheKey]
            # Move to end for LRU (most recently used)
            if cacheKey in self.cacheOrder:
                self.cacheOrder.remove(cacheKey)
                self.cacheOrder.append(cacheKey)
        else:
            # Load and process
            annotation = self.loadAnnotation(annotationPath)
            
            # Get road polygons
            polygons = annotation.get("roadPolygons", [])
            if not polygons:
                polygons = annotation.get("road_polygons", [])
            
            if polygons:
                skeleton = self.skeletonize(polygons, imageShape)
                centerlines = self.extractCenterlines(skeleton, self.numLanes)
            else:
                centerlines = []
            
            # Add to cache with LRU eviction
            self.cache[cacheKey] = centerlines
            self.cacheOrder.append(cacheKey)
            self._evictIfNeeded()
        
        # Encode for this agent
        lanes = self.encodeLanes(centerlines, agentPosition)
        crosswalks = self.encodeCrosswalks(agentPosition)
        
        return {
            "lanes": lanes,
            "crosswalks": crosswalks
        }
    
    def clearCache(self):
        """Clear the centerline cache (call between HPO trials to free memory)."""
        self.cache.clear()
        self.cacheOrder.clear()
