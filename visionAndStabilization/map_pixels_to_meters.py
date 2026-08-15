#!/usr/bin/env python3
"""
map_pixels_to_meters.py: Dynamically projects pixel tracking coordinates to physical meters.
Appends ego-centric metric coordinates and UTM world-aligned coordinates to both
ChunkedProjectPrayagBEVDataset (30Hz) and ChunkedProjectPrayagBEVDataset10Hz (10Hz).
Regenerates unified dataset CSV and JSON files.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.resolve()
DATASETS = [
    BASE_DIR / "ChunkedProjectPrayagBEVDataset",
    BASE_DIR / "ChunkedProjectPrayagBEVDataset10Hz"
]

# Standard Camera Constants
CX, CY = 960.0, 540.0   # principal point (1080p center)
SENSOR_WIDTH = 13.2    # standard 1-inch sensor type (mm)
IMAGE_WIDTH = 1920.0   # resolution width (pixels)
CALIBRATION_FACTOR = 6.55 # calibration multiplier to align GSD scale to physical vehicle sizes
REF_LAT, REF_LON = 25.436562, 81.841327 # global UTM reference coordinates
R_EARTH = 6378137.0    # meters

def gps_to_utm_meters(lat, lon):
    """UTM local flat projection centered on global reference."""
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    ref_lat_rad = np.radians(REF_LAT)
    ref_lon_rad = np.radians(REF_LON)
    
    dy = R_EARTH * (lat_rad - ref_lat_rad)
    dx = R_EARTH * (lon_rad - ref_lon_rad) * np.cos(ref_lat_rad)
    return dx, dy

def process_dataset(dataset_dir):
    print(f"\n=================== Processing Dataset: {dataset_dir.name} ===================")
    
    splits = ["train", "val", "test"]
    all_chunks = []
    
    for split in splits:
        annot_dir = dataset_dir / split / "annotations"
        if not annot_dir.exists():
            continue
        # Find all tracks CSV files
        tracks_csvs = sorted(list(annot_dir.glob("*_tracks.csv")))
        for f in tracks_csvs:
            chunk_id = f.name.replace("_tracks.csv", "")
            all_chunks.append({
                "chunk_id": chunk_id,
                "split": split,
                "tracks_csv": f,
                "tracks_json": annot_dir / f"{chunk_id}_tracks.json",
                "metadata_csv": annot_dir / f"{chunk_id}_metadata.csv"
            })
            
    print(f"Discovered {len(all_chunks)} chunks to project.")
    
    for chunk in tqdm(all_chunks):
        chunk_id = chunk["chunk_id"]
        tracks_csv = chunk["tracks_csv"]
        tracks_json = chunk["tracks_json"]
        metadata_csv = chunk["metadata_csv"]
        
        if not metadata_csv.exists() or not tracks_csv.exists() or not tracks_json.exists():
            continue
            
        # 1. Load Metadata per frame
        df_meta = pd.read_csv(metadata_csv)
        # Create a dictionary mapping frame_id (0-based in metadata) to metrics
        meta_dict = {}
        for _, row in df_meta.iterrows():
            f_id = int(row['frame_id'])
            meta_dict[f_id] = {
                "lat": float(row['latitude']),
                "lon": float(row['longitude']),
                "alt": float(row['altitude']),
                "focal": float(row['focal_length']) / 10.0 # Convert tenths of mm to mm
            }
            
        # 2. Update CSV tracks data
        df_tracks = pd.read_csv(tracks_csv)
        
        # Convert meta_dict to a DataFrame for vectorized lookup
        meta_df_data = []
        for f_id_0based, m in meta_dict.items():
            meta_df_data.append({
                'frame_id_0based': f_id_0based,
                'alt': m['alt'],
                'focal': m['focal'],
                'lat': m['lat'],
                'lon': m['lon']
            })
        if meta_df_data:
            df_meta_lookup = pd.DataFrame(meta_df_data)
        else:
            df_meta_lookup = pd.DataFrame(columns=['frame_id_0based', 'alt', 'focal', 'lat', 'lon'])
            
        df_tracks['frame_id_0based'] = df_tracks['frame_id'] - 1
        df_merged = pd.merge(df_tracks, df_meta_lookup, on='frame_id_0based', how='left')
        
        # Fill missing values from metadata with defaults
        df_merged['alt'] = df_merged['alt'].fillna(80.0)
        df_merged['focal'] = df_merged['focal'].fillna(31.9)
        df_merged['lat'] = df_merged['lat'].fillna(REF_LAT)
        df_merged['lon'] = df_merged['lon'].fillna(REF_LON)
        
        # Calculate scales vectorized
        # If focal > 0, scale = (alt * SENSOR_WIDTH) / (focal * IMAGE_WIDTH), else 0.05
        df_merged['scale'] = np.where(
            df_merged['focal'] > 0,
            (df_merged['alt'] * SENSOR_WIDTH) / (df_merged['focal'] * IMAGE_WIDTH),
            0.05
        ) * CALIBRATION_FACTOR
        scale = df_merged['scale']
        
        # Local center coordinates
        px_dx = df_merged['center_x'] - CX
        px_dy = CY - df_merged['center_y']
        
        df_tracks['local_center_x_m'] = px_dx * scale
        df_tracks['local_center_y_m'] = px_dy * scale
        
        # Drone UTM coordinates
        lat_rad = np.radians(df_merged['lat'])
        lon_rad = np.radians(df_merged['lon'])
        ref_lat_rad = np.radians(REF_LAT)
        ref_lon_rad = np.radians(REF_LON)
        
        drone_dy = R_EARTH * (lat_rad - ref_lat_rad)
        drone_dx = R_EARTH * (lon_rad - ref_lon_rad) * np.cos(ref_lat_rad)
        
        df_tracks['world_center_x_m'] = drone_dx + df_tracks['local_center_x_m']
        df_tracks['world_center_y_m'] = drone_dy + df_tracks['local_center_y_m']
        
        # Corner projections
        for i in range(1, 5):
            pt_dx = df_merged[f'obb_corner{i}_x'] - CX
            pt_dy = CY - df_merged[f'obb_corner{i}_y']
            
            pt_local_x = pt_dx * scale
            pt_local_y = pt_dy * scale
            
            df_tracks[f'local_obb_corner{i}_x_m'] = pt_local_x
            df_tracks[f'local_obb_corner{i}_y_m'] = pt_local_y
            df_tracks[f'world_obb_corner{i}_x_m'] = drone_dx + pt_local_x
            df_tracks[f'world_obb_corner{i}_y_m'] = drone_dy + pt_local_y
            
        # Clean up temporary column
        df_tracks.drop(columns=['frame_id_0based'], inplace=True)
        
        # Overwrite CSV
        df_tracks.to_csv(tracks_csv, index=False)
        
        # 3. Update JSON tracks data
        # Structure is frame_id_1based -> track_id -> content
        with open(tracks_json, 'r', encoding='utf-8') as f:
            data_json = json.load(f)
            
        new_data_json = {}
        for f_id_str, tracks in data_json.items():
            f_id_0based = int(f_id_str) - 1
            alt = 80.0
            focal = 31.9
            lat = REF_LAT
            lon = REF_LON
            
            if f_id_0based in meta_dict:
                meta = meta_dict[f_id_0based]
                alt = meta["alt"]
                focal = meta["focal"]
                lat = meta["lat"]
                lon = meta["lon"]
                
            scale = ((alt * SENSOR_WIDTH) / (focal * IMAGE_WIDTH) if focal > 0 else 0.05) * CALIBRATION_FACTOR
            drone_dx, drone_dy = gps_to_utm_meters(lat, lon)
            
            new_tracks = {}
            for t_id_str, track_info in tracks.items():
                new_info = track_info.copy()
                
                cx_p = float(track_info["center"][0])
                cy_p = float(track_info["center"][1])
                
                # Local
                local_cx = (cx_p - CX) * scale
                local_cy = (CY - cy_p) * scale
                new_info["local_center_m"] = [local_cx, local_cy]
                new_info["world_center_m"] = [drone_dx + local_cx, drone_dy + local_cy]
                
                local_obb = []
                world_obb = []
                for pt in track_info["obb"]:
                    pt_local_x = (float(pt[0]) - CX) * scale
                    pt_local_y = (CY - float(pt[1])) * scale
                    local_obb.append([pt_local_x, pt_local_y])
                    world_obb.append([drone_dx + pt_local_x, drone_dy + pt_local_y])
                    
                new_info["local_obb_m"] = local_obb
                new_info["world_obb_m"] = world_obb
                new_tracks[t_id_str] = new_info
                
            new_data_json[f_id_str] = new_tracks
            
        with open(tracks_json, 'w', encoding='utf-8') as f:
            json.dump(new_data_json, f, separators=(',', ':')) # minified

    # 4. Compile Unified Tracking CSV and JSON
    print(f"Re-unifying global databases for {dataset_dir.name}...")
    
    all_csv_data = []
    unified_json_data = {}
    
    for split in splits:
        annot_dir = dataset_dir / split / "annotations"
        if not annot_dir.exists():
            continue
            
        tracks_csvs = sorted(list(annot_dir.glob("*_tracks.csv")))
        for csv_f in tracks_csvs:
            df = pd.read_csv(csv_f)
            all_csv_data.append(df)
            
            json_f = annot_dir / csv_f.name.replace(".csv", ".json")
            chunk_id = csv_f.name.replace("_tracks.csv", "")
            with open(json_f, 'r') as fh:
                unified_json_data[chunk_id] = json.load(fh)
                
    # Save CSV
    df_unified = pd.concat(all_csv_data, ignore_index=True)
    unified_csv_path = dataset_dir / "unified_tracking_data.csv"
    df_unified.to_csv(unified_csv_path, index=False)
    print(f"Saved unified CSV: {unified_csv_path} ({len(df_unified)} rows)")
    
    # Save JSON
    unified_json_path = dataset_dir / "unified_tracking_data.json"
    with open(unified_json_path, 'w') as fh:
        json.dump(unified_json_data, fh, separators=(',', ':'))
    print(f"Saved unified JSON: {unified_json_path}")

def main():
    for db in DATASETS:
        if db.exists():
            process_dataset(db)
            
    print("\nMapping metrics computation completed successfully!")

if __name__ == "__main__":
    main()
