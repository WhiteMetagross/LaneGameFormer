"""
Dataset for PrayagGameFormer.

Loads ChunkedProjectPrayagBEVDataset and formats data for GameFormer.
Uses CSV track files and road annotation JSON files.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from torch.utils.data import Dataset

from .mapEncoder import MapEncoder


class PrayagDataset(Dataset):
    """
    Dataset for loading Prayag BEV data in GameFormer format.
    
    GameFormer expects inputs with specific structure:
    - ego_state: (T, 9) - ego vehicle trajectory and attributes
    - neighbors_state: (N, T, 9) - neighbor trajectories
    - map_lanes: (N+1, numLanes, lanePoints, 16) - lane features per agent
    - map_crosswalks: (N+1, numCrosswalks, crosswalkPoints, 3) - crosswalk features
    """
    
    def __init__(
        self,
        config: dict,
        split: str = "train",
        testChunks: Optional[int] = None
    ):
        """
        Initialize dataset.
        
        Args:
            config: Configuration dictionary
            split: "train", "val", or "test"
            testChunks: Limit to N chunks for testing
        """
        self.config = config
        self.split = split
        
        # Paths
        if split == "train":
            self.annotationDir = Path(config["trainAnnotations"])
            self.videoDir = Path(config["trainVideos"])
        elif split == "val":
            self.annotationDir = Path(config["valAnnotations"])
            self.videoDir = Path(config["valVideos"])
        else:
            self.annotationDir = Path(config["testAnnotations"])
            self.videoDir = Path(config["testVideos"])
        
        # Settings
        self.obsHorizon = config["obsHorizon"]
        self.predHorizon = config["predHorizon"]
        self.totalHorizon = self.obsHorizon + self.predHorizon
        self.maxAgents = config["maxAgents"]
        self.neighborsToPredict = config.get("neighborsToPredict", 31)
        self.resolution = config.get("resolution", (1080, 1920))
        self.fps = config.get("fps", 10)
        
        # Map encoder
        cacheDir = Path(config.get("saveDir", ".")) / "map_cache"
        self.mapEncoder = MapEncoder(config, cacheDir=str(cacheDir))
        
        # Load chunk list
        chunkListPath = config["chunkList"][split]
        if os.path.exists(chunkListPath):
            with open(chunkListPath, "r") as f:
                self.chunks = [line.strip() for line in f if line.strip()]
        else:
            # List all track files and extract chunk names
            self.chunks = [
                p.stem.replace("_tracks", "") 
                for p in self.annotationDir.glob("*_tracks.csv")
            ]
        
        # Limit chunks for testing
        if testChunks is not None:
            self.chunks = self.chunks[:testChunks]
        
        # Build index
        self.sequences = self._buildSequenceIndex()
        
        print(f"PrayagDataset ({split}): {len(self.chunks)} chunks, {len(self.sequences)} sequences")
    
    def _buildSequenceIndex(self) -> List[dict]:
        """
        Build index of valid sequences from CSV track files.
        
        Returns:
            List of sequence dictionaries with chunk info and frames
        """
        sequences = []
        stride = max(1, self.fps // 2)  # ~0.5 second stride
        
        for chunkName in self.chunks:
            tracksPath = self.annotationDir / f"{chunkName}_tracks.csv"
            
            if not tracksPath.exists():
                continue
            
            try:
                df = pd.read_csv(tracksPath)
                frameIds = sorted(df['frame_id'].unique())
                
                if len(frameIds) < self.totalHorizon:
                    continue
                
                # Find consecutive frame ranges
                consecutiveRanges = []
                if len(frameIds) > 0:
                    start = frameIds[0]
                    prev = start
                    for fid in frameIds[1:]:
                        if fid != prev + 1:
                            consecutiveRanges.append((start, prev))
                            start = fid
                        prev = fid
                    consecutiveRanges.append((start, prev))
                
                # Create sequences
                for rangeStart, rangeEnd in consecutiveRanges:
                    rangeLen = rangeEnd - rangeStart + 1
                    if rangeLen < self.totalHorizon:
                        continue
                    
                    for startFrame in range(rangeStart, rangeEnd - self.totalHorizon + 2, stride):
                        frames = list(range(startFrame, startFrame + self.totalHorizon))
                        sequences.append({
                            'chunkName': chunkName,
                            'frames': frames,
                            'tracksPath': str(tracksPath)
                        })
                        
            except Exception as e:
                print(f"Warning: Error loading chunk {chunkName}: {e}")
                continue
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def _getAgentStates(
        self,
        df: pd.DataFrame,
        frames: List[int],
        maxAgents: int
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Get agent states for given frame range from CSV data.
        
        Args:
            df: Track DataFrame
            frames: List of frame indices
            maxAgents: Maximum number of agents
            
        Returns:
            states: (maxAgents, T, 9) array of agent states
            agentIds: List of agent IDs in order
        """
        states = np.zeros((maxAgents, len(frames), 9), dtype=np.float32)
        
        # Get agents present in observation period
        obsFrames = frames[:self.obsHorizon]
        obsDf = df[df['frame_id'].isin(obsFrames)]
        
        if len(obsDf) == 0:
            return states, []
        
        # Count observations per agent
        agentCounts = obsDf['track_id'].value_counts()
        
        # Also get position at last observation frame to sort by distance from center
        H, W = self.resolution
        centerX, centerY = W / 2, H / 2
        
        currentFrame = obsFrames[-1]
        currentDf = df[df['frame_id'] == currentFrame]
        
        # Score agents by observation count and centrality
        agentScores = {}
        for trackId, count in agentCounts.items():
            if trackId in currentDf['track_id'].values:
                row = currentDf[currentDf['track_id'] == trackId].iloc[0]
                dist = np.sqrt((row['center_x'] - centerX)**2 + (row['center_y'] - centerY)**2)
                agentScores[trackId] = (count, -dist)  # Higher count, closer to center = better
            else:
                agentScores[trackId] = (count, -1000)  # Lower priority if not visible at current frame
        
        # Sort and select top agents
        sortedAgents = sorted(agentScores.keys(), key=lambda x: agentScores[x], reverse=True)[:maxAgents]
        agentIds = list(sortedAgents)
        
        # Extract states for each agent
        for i, trackId in enumerate(agentIds):
            trackDf = df[df['track_id'] == trackId]
            
            for t, frameId in enumerate(frames):
                frameDf = trackDf[trackDf['frame_id'] == frameId]
                
                if len(frameDf) > 0:
                    row = frameDf.iloc[0]
                    
                    x = row.get('center_x', 0)
                    y = row.get('center_y', 0)
                    vx = row.get('vx', 0) if 'vx' in row else 0
                    vy = row.get('vy', 0) if 'vy' in row else 0
                    heading = row.get('yaw', row.get('heading', 0))
                    width = row.get('width', 2.0)
                    length = row.get('length', 4.5)
                    height = row.get('height', 1.5)
                    agentType = row.get('class_id', 1)
                    
                    states[i, t] = [x, y, vx, vy, heading, width, length, height, agentType]
        
        return states, agentIds
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a data sample.
        
        Returns:
            Dictionary with GameFormer-compatible inputs
        """
        seqInfo = self.sequences[idx]
        chunkName = seqInfo['chunkName']
        frames = seqInfo['frames']
        tracksPath = seqInfo['tracksPath']
        
        # Load track data
        df = pd.read_csv(tracksPath)
        
        # Get agent states
        states, agentIds = self._getAgentStates(df, frames, self.maxAgents)
        
        # --- Ego-centric normalization ---
        # Origin: ego agent's position at last observation frame
        origin = states[0, self.obsHorizon - 1, :2].copy()  # (2,)
        
        # Save absolute last-obs positions for map encoding (before transform)
        absLastObs = states[:, self.obsHorizon - 1, :2].copy()  # (maxAgents, 2)
        
        # Subtract origin from xy of valid (non-zero) entries only
        # so that zero-padded agents/frames stay zero
        validMask = np.any(states != 0, axis=-1)  # (maxAgents, T)
        xyShift = np.zeros_like(states[..., :2])  # (maxAgents, T, 2)
        xyShift[validMask] = origin
        states[..., :2] -= xyShift
        # No rotation applied (identity, same as LaneGameFormer for BEV data)
        # --- End ego-centric normalization ---
        
        # Split into observation and prediction
        obsStates = states[:, :self.obsHorizon]
        predStates = states[:, self.obsHorizon:]
        
        # Ego is first agent (most observed and closest to center)
        egoState = obsStates[0]  # (obsHorizon, 9)
        neighborsState = obsStates[1:self.neighborsToPredict + 1]  # (N, obsHorizon, 9)
        
        # Ground truth future (already in ego-centric space)
        egoFuture = predStates[0]  # (predHorizon, 9)
        neighborsFuture = predStates[1:self.neighborsToPredict + 1]  # (N, predHorizon, 9)
        
        # Encode maps using absolute positions (MapEncoder computes agent-relative features)
        numAgentSlots = self.neighborsToPredict + 1  # Fixed size for all samples
        numAgentsActual = min(len(agentIds), numAgentSlots)
        roadAnnotationPath = str(self.annotationDir / f"{chunkName}_road_annotation.json")
        
        mapLanes = np.zeros(
            (numAgentSlots, self.config["numLanes"], self.config["lanePoints"], 16),
            dtype=np.float32
        )
        mapCrosswalks = np.zeros(
            (numAgentSlots, self.config["numCrosswalks"], self.config["crosswalkPoints"], 3),
            dtype=np.float32
        )
        
        for i in range(numAgentsActual):
            agentPos = absLastObs[i]  # Absolute position for map query
            mapFeatures = self.mapEncoder.encode(
                roadAnnotationPath, agentPos, frames[0]
            )
            mapLanes[i] = mapFeatures["lanes"]
            mapCrosswalks[i] = mapFeatures["crosswalks"]
        
        # Build output
        sample = {
            "ego_state": torch.from_numpy(egoState),
            "neighbors_state": torch.from_numpy(neighborsState),
            "map_lanes": torch.from_numpy(mapLanes),
            "map_crosswalks": torch.from_numpy(mapCrosswalks),
            "ego_future": torch.from_numpy(egoFuture[:, :2]),  # Only xy
            "neighbors_future": torch.from_numpy(neighborsFuture[:, :, :2]),
            "origin": torch.from_numpy(origin),  # (2,) absolute pixel position of ego at current time
            "chunk": chunkName,
            "start_frame": frames[0],
        }
        
        return sample


def collateFn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for batching samples.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Batched dictionary
    """
    batchOut = {}
    
    for key in batch[0].keys():
        if key in ["chunk", "start_frame"]:
            batchOut[key] = [sample[key] for sample in batch]
        else:
            tensors = [sample[key] for sample in batch]
            batchOut[key] = torch.stack(tensors, dim=0)
    
    return batchOut
