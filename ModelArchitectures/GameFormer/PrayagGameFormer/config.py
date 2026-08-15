"""
Configuration for PrayagGameFormer.

Provides configuration for training GameFormer on ChunkedProjectPrayagBEVDataset.
"""

import os
from pathlib import Path


def getConfig(datasetType: str = "10Hz") -> dict:
    """
    Get configuration for training.
    
    Args:
        datasetType: "10Hz" or "30Hz"
        
    Returns:
        Configuration dictionary
    """
    baseDir = Path(__file__).parent.parent
    
    if datasetType == "10Hz":
        datasetDir = baseDir / "ChunkedProjectPrayagBEVDataset10Hz"
        obsHorizon = 10   # 1 second at 10 FPS
        predHorizon = 30  # 3 seconds at 10 FPS
        fps = 10
    else:
        datasetDir = baseDir / "ChunkedProjectPrayagBEVDataset"
        obsHorizon = 30   # 1 second at 30 FPS
        predHorizon = 90  # 3 seconds at 30 FPS
        fps = 30
    
    config = {
        # Dataset paths
        "datasetDir": str(datasetDir),
        "trainAnnotations": str(datasetDir / "train" / "annotations"),
        "trainVideos": str(datasetDir / "train" / "videos"),
        "valAnnotations": str(datasetDir / "val" / "annotations"),
        "valVideos": str(datasetDir / "val" / "videos"),
        "testAnnotations": str(datasetDir / "test" / "annotations"),
        "testVideos": str(datasetDir / "test" / "videos"),
        "chunkList": {
            "train": str(datasetDir / "train_chunks.txt"),
            "val": str(datasetDir / "val_chunks.txt"),
            "test": str(datasetDir / "test_chunks.txt"),
        },
        
        # Temporal settings
        "obsHorizon": obsHorizon,
        "predHorizon": predHorizon,
        "fps": fps,
        "totalHorizon": obsHorizon + predHorizon,
        
        # Agent settings
        "maxAgents": 100,  # Max 100 agents for Indian traffic
        "neighborsToPredict": 31,  # How many neighbors to predict (ego + neighbors)
        "resolution": (1080, 1920),  # Video resolution (H, W)
        
        # Map settings (skeletonized)
        "numLanes": 50,  # Max lane segments after skeletonization
        "numCrosswalks": 10,  # Placeholder (no crosswalk data)
        "lanePoints": 100,  # Points per lane segment
        "crosswalkPoints": 10,  # Points per crosswalk
        
        # Model architecture
        "dim": 256,
        "heads": 8,
        "dropout": 0.1,
        "encoderLayers": 4,
        "decoderLevels": 3,  # Level-k reasoning depth
        "numModes": 6,  # Multi-modal predictions
        
        # Training
        "batchSize": 8,
        "valBatchSize": 8,
        "numEpochs": 20,
        "lr": [1e-4, 1e-5],  # [initial, final]
        "lrEpochs": [10, 15],  # Milestones for LR decay
        "weightDecay": 1e-4,
        
        # Loss weights
        "regWeight": 1.0,
        "clsWeight": 1.0,
        
        # Data loading
        "workers": 4,
        "valWorkers": 2,
        
        # Checkpointing
        "saveDir": str(baseDir / "PrayagGameFormer" / "checkpoints"),
        "logDir": str(baseDir / "PrayagGameFormer" / "logs"),
        
        # Device
        "device": "cuda",
        
        # Seed
        "seed": 42,
    }
    
    return config


def getTestConfig(numChunks: int = 1, datasetType: str = "10Hz") -> dict:
    """
    Get a minimal config for testing with limited chunks.
    
    Args:
        numChunks: Number of chunks to use for testing
        datasetType: "10Hz" or "30Hz" dataset type
        
    Returns:
        Minimal configuration dictionary
    """
    config = getConfig(datasetType)
    
    # Reduce for testing
    config["batchSize"] = 4
    config["valBatchSize"] = 4
    config["numEpochs"] = 5
    config["workers"] = 2
    config["valWorkers"] = 1
    config["testChunks"] = numChunks
    
    return config


def getHPOConfig(datasetType: str = "10Hz") -> dict:
    """
    Get configuration for HPO using Stratified datasets (20% sample).
    
    Uses StratifiedProjectPrayagBEVDataset for faster HPO runs while
    maintaining representative data distribution.
    
    Args:
        datasetType: "10Hz" or "30Hz"
        
    Returns:
        Configuration dictionary pointing to stratified dataset
    """
    baseDir = Path(__file__).parent.parent
    
    if datasetType == "10Hz":
        datasetDir = baseDir / "StratifiedProjectPrayagBEVDataset10Hz"
        obsHorizon = 10   # 1 second at 10 FPS
        predHorizon = 30  # 3 seconds at 10 FPS
        fps = 10
    else:
        datasetDir = baseDir / "StratifiedProjectPrayagBEVDataset"
        obsHorizon = 30   # 1 second at 30 FPS
        predHorizon = 90  # 3 seconds at 30 FPS
        fps = 30
    
    config = {
        # Dataset paths (Stratified for HPO)
        "datasetDir": str(datasetDir),
        "trainAnnotations": str(datasetDir / "train" / "annotations"),
        "trainVideos": str(datasetDir / "train" / "videos"),
        "valAnnotations": str(datasetDir / "val" / "annotations"),
        "valVideos": str(datasetDir / "val" / "videos"),
        "testAnnotations": str(datasetDir / "test" / "annotations"),
        "testVideos": str(datasetDir / "test" / "videos"),
        "chunkList": {
            "train": str(datasetDir / "train_chunks.txt"),
            "val": str(datasetDir / "val_chunks.txt"),
            "test": str(datasetDir / "test_chunks.txt"),
        },
        
        # Temporal settings
        "obsHorizon": obsHorizon,
        "predHorizon": predHorizon,
        "fps": fps,
        "totalHorizon": obsHorizon + predHorizon,
        
        # Agent settings
        "maxAgents": 32,  # Reduced for HPO memory efficiency
        "neighborsToPredict": 31,
        "resolution": (1080, 1920),
        
        # Map settings (skeletonized)
        "numLanes": 50,
        "numCrosswalks": 10,
        "lanePoints": 100,
        "crosswalkPoints": 10,
        
        # Model architecture (fixed for HPO - not tuned)
        "dim": 256,
        "heads": 8,
        "dropout": 0.1,
        "encoderLayers": 4,
        "decoderLevels": 3,
        "numModes": 6,
        
        # HPO training settings
        "batchSize": 16,
        "valBatchSize": 16,
        "numEpochs": 10,  # HPO epochs
        "lr": [1e-4, 1e-5],
        "lrEpochs": [5, 8],
        "weightDecay": 1e-4,
        
        # Loss weights
        "regWeight": 1.0,
        "clsWeight": 1.0,
        
        # Data loading
        "workers": 8,
        "valWorkers": 4,
        
        # Checkpointing
        "saveDir": str(baseDir / "PrayagGameFormer" / "checkpoints"),
        "logDir": str(baseDir / "PrayagGameFormer" / "logs"),
        
        # Device
        "device": "cuda",
        
        # Seed
        "seed": 42,
    }
    
    return config
