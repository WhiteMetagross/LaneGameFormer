#!/usr/bin/env python3
"""
chunk_dataset.py: Slices ProjectPrayagTopDownDataset into ChunkedProjectPrayagBEVDataset (30Hz)
using ChunkedProjectPrayagBEVDataset/chunk_manifest.json as the blueprint.
"""

import os
import json
import shutil
import subprocess
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import concurrent.futures

# Configuration Paths
BASE_DIR = Path(__file__).parent.resolve()
SRC_DIR = BASE_DIR / "ProjectPrayagTopDownDataset"
DST_DIR = BASE_DIR / "ChunkedProjectPrayagBEVDataset"
MANIFEST_PATH = DST_DIR / "chunk_manifest.json"

def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)

def find_video_file(scene_id):
    # Search in original dataset videos directory first
    video_path = SRC_DIR / "CIRAerialDroneIndianIntersectionsVideoes" / f"{scene_id}.mp4"
    if video_path.exists():
        return video_path
    # Fallback to temp_videos
    video_path = BASE_DIR / "temp_videos" / f"{scene_id}.mp4"
    if video_path.exists():
        return video_path
    return None

def to_win_path(path):
    p_str = str(Path(path).as_posix())
    if p_str.startswith("/mnt/c/"):
        return "C:/" + p_str[7:]
    return p_str

def process_chunk(chunk_info):
    chunk_id = chunk_info['chunk_id']
    scene_id = chunk_info['original_scene_id']
    start_frame = chunk_info['start_frame']
    end_frame = chunk_info['end_frame']
    num_frames = chunk_info['num_frames']
    split = chunk_info['split']
    
    # Define directories
    split_dir = DST_DIR / split
    annot_dir = split_dir / "annotations"
    video_dir = split_dir / "videos"
    
    ensure_dir(annot_dir)
    ensure_dir(video_dir)
    
    # 1. Slice tracks.json
    # Source JSON is in intermediate_files/[scene_id]/[scene_id]_tracks_stabilized.json
    src_json_path = SRC_DIR / "intermediate_files" / scene_id / f"{scene_id}_tracks_stabilized.json"
    dst_json_path = annot_dir / f"{chunk_id}_tracks.json"
    
    try:
        with open(src_json_path, 'r', encoding='utf-8') as f:
            src_json_data = json.load(f)
            
        chunk_json_data = {}
        for t in range(num_frames):
            src_frame_id = start_frame + t + 1
            src_frame_key = str(src_frame_id)
            if src_frame_key in src_json_data:
                chunk_frame_key = str(t + 1)
                chunk_json_data[chunk_frame_key] = src_json_data[src_frame_key]
                
        with open(dst_json_path, 'w', encoding='utf-8') as f:
            json.dump(chunk_json_data, f, indent=None, separators=(',', ':')) # Minified
    except Exception as e:
        print(f"Error slicing tracks JSON for {chunk_id}: {e}")
        
    # 2. Slice tracks.csv
    # Source CSV is in intermediate_files/[scene_id]/[scene_id]_tracks_stabilized.csv
    src_csv_path = SRC_DIR / "intermediate_files" / scene_id / f"{scene_id}_tracks_stabilized.csv"
    dst_csv_path = annot_dir / f"{chunk_id}_tracks.csv"
    
    try:
        df_tracks = pd.read_csv(src_csv_path)
        # Filter range: [start_frame + 1, end_frame + 1]
        df_sliced = df_tracks[(df_tracks['frame_id'] >= start_frame + 1) & (df_tracks['frame_id'] <= end_frame + 1)].copy()
        # Re-index frame_id: frame_id - start_frame
        df_sliced['frame_id'] = df_sliced['frame_id'] - start_frame
        # Rename scene_id to chunk_id
        df_sliced['scene_id'] = chunk_id
        
        df_sliced.to_csv(dst_csv_path, index=False)
    except Exception as e:
        print(f"Error slicing tracks CSV for {chunk_id}: {e}")
        
    # 3. Slice metadata.csv
    src_meta_path = SRC_DIR / "CIRAerialDroneIndianIntersectionsVideoes" / f"{scene_id}_metadata.csv"
    dst_meta_path = annot_dir / f"{chunk_id}_metadata.csv"
    
    try:
        df_meta = pd.read_csv(src_meta_path)
        # Filter range: [start_frame, end_frame] (0-based)
        df_meta_sliced = df_meta[(df_meta['frame_id'] >= start_frame) & (df_meta['frame_id'] <= end_frame)].copy()
        # Re-index
        df_meta_sliced['frame_id'] = df_meta_sliced['frame_id'] - start_frame
        
        df_meta_sliced.to_csv(dst_meta_path, index=False)
    except Exception as e:
        print(f"Error slicing metadata CSV for {chunk_id}: {e}")
        
    # 4. Copy & rename static road annotation files
    road_files = [
        ("_road_annotation.json", "_road_annotation.json"),
        ("_road_mask.png", "_road_mask.png"),
        ("_road_viz.png", "_road_viz.png")
    ]
    for src_suff, dst_suff in road_files:
        src_f = SRC_DIR / "CIRAerialDroneIndianIntersectionsVideoes" / f"{scene_id}{src_suff}"
        dst_f = annot_dir / f"{chunk_id}{dst_suff}"
        if src_f.exists():
            shutil.copy2(src_f, dst_f)
            
    # 5. Slice Video via FFmpeg
    video_src = find_video_file(scene_id)
    video_dst = video_dir / f"{chunk_id}.mp4"
    
    if video_src:
        start_time_sec = start_frame / 30.0
        duration_sec = num_frames / 30.0
        try:
            cmd = [
                'ffmpeg.exe', '-y',
                '-ss', f"{start_time_sec:.4f}",
                '-t', f"{duration_sec:.4f}",
                '-i', to_win_path(video_src),
                '-c:v', 'libx264',
                '-crf', '18',
                '-preset', 'fast',
                '-an',
                to_win_path(video_dst)
            ]
            # Run quietly
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"Error slicing video for {chunk_id}: {e}")
    else:
        print(f"Warning: Source video not found for scene {scene_id} ({chunk_id})")

def main():
    if not MANIFEST_PATH.exists():
        print(f"Error: Manifest not found at {MANIFEST_PATH}")
        return
        
    print("Reading chunk manifest blueprint...")
    with open(MANIFEST_PATH, 'r') as f:
        manifest = json.load(f)
        
    # Gather chunks across train, val, test splits
    chunks_to_process = []
    splits = manifest.get('splits', {})
    for split_name, chunks in splits.items():
        for chunk in chunks:
            chunk['split'] = split_name
            chunks_to_process.append(chunk)
            
    print(f"Loaded {len(chunks_to_process)} chunks from blueprint manifest.")
    
    # Process in parallel
    print("Chunking trajectories, metadata, road annotations, and video files...")
    # Slicing videos is heavy, so we use max_workers=6 to balance CPU and IO load
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        list(tqdm(executor.map(process_chunk, chunks_to_process), total=len(chunks_to_process), unit="chunk"))
        
    print("Temporal chunking completed successfully!")

if __name__ == "__main__":
    main()
