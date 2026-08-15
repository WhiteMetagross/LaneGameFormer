#!/usr/bin/env python3
"""
create_manifest.py: Dynamically generates chunk_manifest.json and splits
for contiguous (0-gap, 0-overlap) 30-second chunks of ProjectPrayagTopDownDataset.
"""

import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# Paths
BASE_DIR = Path(__file__).parent.resolve()
SRC_DIR = BASE_DIR / "ProjectPrayagTopDownDataset"
DST_DIR = BASE_DIR / "ChunkedProjectPrayagBEVDataset"

SCENES = {
    "DJI_0910": {"time_of_day": "morning", "chunks_alloc": {"train": 10, "val": 2, "test": 2}},
    "DJI_0911": {"time_of_day": "morning", "chunks_alloc": {"train": 11, "val": 1, "test": 2}},
    "DJI_0912": {"time_of_day": "morning", "chunks_alloc": {"train": 1, "val": 0, "test": 0}},
    "DJI_0914": {"time_of_day": "afternoon", "chunks_alloc": {"train": 10, "val": 1, "test": 1}},
    "DJI_0915": {"time_of_day": "afternoon", "chunks_alloc": {"train": 8, "val": 1, "test": 1}},
    "DJI_0916": {"time_of_day": "unknown", "chunks_alloc": {"train": 1, "val": 0, "test": 0}}
}

CHUNK_FRAMES = 900

def get_density_classification(avg_vehicles):
    if avg_vehicles < 15:
        return "low"
    elif avg_vehicles <= 30:
        return "medium"
    else:
        return "high"

def main():
    print("Dynamically compiling contiguous chunk manifest...")
    
    splits_data = {
        "train": [],
        "val": [],
        "test": []
    }
    
    total_chunks = 0
    train_count = 0
    val_count = 0
    test_count = 0
    
    # Process each scene
    for scene_id, info in SCENES.items():
        print(f"Processing scene: {scene_id}")
        
        # 1. Load stabilized tracks to calculate frame metrics
        tracks_json_path = SRC_DIR / "intermediate_files" / scene_id / f"{scene_id}_tracks_stabilized.json"
        if not tracks_json_path.exists():
            print(f"  Warning: Tracks JSON not found for {scene_id}. Skipping.")
            continue
            
        with open(tracks_json_path, 'r', encoding='utf-8') as f:
            tracks_data = json.load(f)
            
        # Determine total frames in scene
        frame_ids = [int(k) for k in tracks_data.keys()]
        if not frame_ids:
            continue
        max_frame = max(frame_ids)
        
        # 2. Load metadata CSV to extract telemetry
        meta_csv_path = SRC_DIR / "CIRAerialDroneIndianIntersectionsVideoes" / f"{scene_id}_metadata.csv"
        df_meta = pd.read_csv(meta_csv_path) if meta_csv_path.exists() else None
        
        # Calculate number of chunks (contiguous, 0 gaps, 0 overlaps)
        # DJI_0916 has only 240 frames, so if max_frame < CHUNK_FRAMES, retain as single chunk
        if max_frame < CHUNK_FRAMES:
            num_chunks = 1
        else:
            num_chunks = max_frame // CHUNK_FRAMES
            
        # Allocation counts
        alloc = info["chunks_alloc"]
        split_assignments = []
        for split_name, count in alloc.items():
            split_assignments.extend([split_name] * count)
            
        # Ensure split assignments match actual generated chunks length
        if len(split_assignments) < num_chunks:
            # Append remaining to train
            split_assignments.extend(["train"] * (num_chunks - len(split_assignments)))
        elif len(split_assignments) > num_chunks:
            split_assignments = split_assignments[:num_chunks]
            
        # Generate chunks
        for idx in range(num_chunks):
            start_f = idx * CHUNK_FRAMES
            end_f = start_f + CHUNK_FRAMES - 1
            if max_frame < CHUNK_FRAMES:
                start_f = 0
                end_f = max_frame - 1
                
            actual_num_frames = end_f - start_f + 1
            chunk_id = f"{scene_id}_chunk_{idx}"
            assigned_split = split_assignments[idx]
            
            # Slice frame vehicle counts
            vehicles_per_frame = []
            unique_tracks = set()
            
            for t in range(actual_num_frames):
                frame_idx = start_f + t + 1
                frame_key = str(frame_idx)
                if frame_key in tracks_data:
                    vehicles_in_frame = tracks_data[frame_key]
                    vehicles_per_frame.append(len(vehicles_in_frame))
                    for track_id in vehicles_in_frame.keys():
                        unique_tracks.add(int(track_id))
                else:
                    vehicles_per_frame.append(0)
                    
            avg_vehicles = sum(vehicles_per_frame) / len(vehicles_per_frame) if vehicles_per_frame else 0.0
            density = get_density_classification(avg_vehicles)
            
            # Slice metadata metrics
            lat, lon, alt = 0.0, 0.0, 0.0
            if df_meta is not None:
                # filter range: [start_f, end_f]
                df_sliced = df_meta[(df_meta['frame_id'] >= start_f) & (df_meta['frame_id'] <= end_f)]
                if not df_sliced.empty:
                    lat = float(df_sliced['latitude'].mean())
                    lon = float(df_sliced['longitude'].mean())
                    alt = float(df_sliced['altitude'].mean())
            
            chunk_dict = {
                "chunk_id": chunk_id,
                "original_scene_id": scene_id,
                "chunk_index": idx,
                "start_frame": start_f,
                "end_frame": end_f,
                "num_frames": actual_num_frames,
                "start_time_ms": int((start_f / 30.0) * 1000),
                "end_time_ms": int((end_f / 30.0) * 1000),
                "time_of_day": info["time_of_day"],
                "agent_density": density,
                "avg_vehicles_per_frame": avg_vehicles,
                "total_unique_tracks": len(unique_tracks),
                "latitude": lat,
                "longitude": lon,
                "altitude": alt,
                "split": assigned_split
            }
            
            splits_data[assigned_split].append(chunk_dict)
            total_chunks += 1
            if assigned_split == "train":
                train_count += 1
            elif assigned_split == "val":
                val_count += 1
            elif assigned_split == "test":
                test_count += 1
                
    # Create final manifest structure
    manifest_out = {
        "created": datetime.utcnow().isoformat(),
        "parameters": {
            "fps": 30,
            "chunk_duration_seconds": 30,
            "gap_duration_seconds": 0,
            "chunk_frames": CHUNK_FRAMES,
            "gap_frames": 0,
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
            "random_seed": 17
        },
        "statistics": {
            "total_chunks": total_chunks,
            "train_chunks": train_count,
            "val_chunks": val_count,
            "test_chunks": test_count
        },
        "splits": splits_data
    }
    
    # Save files
    DST_DIR.mkdir(parents=True, exist_ok=True)
    with open(DST_DIR / "chunk_manifest.json", 'w', encoding='utf-8') as f:
        json.dump(manifest_out, f, indent=4)
        
    print(f"Generated chunk_manifest.json at {DST_DIR / 'chunk_manifest.json'}")
    print(f"Total Chunks: {total_chunks} (Train: {train_count}, Val: {val_count}, Test: {test_count})")
    
    # Write split .txt files
    for split_name, chunks in splits_data.items():
        txt_path = DST_DIR / f"{split_name}_chunks.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(f"{chunk['chunk_id']}\n")
        print(f"Generated {txt_path}")

if __name__ == "__main__":
    main()
