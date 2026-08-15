import os
import sys
import json
import csv
import numpy as np
import argparse
import config

def convert_gps_to_local_meters(lat, lon, ref_lat, ref_lon):
    # Quick UTM/flat projection centered at ref_lat, ref_lon
    # R_earth = 6378137 meters
    R = 6378137.0
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    ref_lat_rad = np.radians(ref_lat)
    ref_lon_rad = np.radians(ref_lon)
    
    dy = R * (lat_rad - ref_lat_rad)
    dx = R * (lon_rad - ref_lon_rad) * np.cos(ref_lat_rad)
    return dx, dy

def run_alignment(scene_ids):
    print("\n=================== Running Multi-Video Alignment ===================")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_video_dir = config.INPUT_DIR
    intermediate_dir = config.INTERMEDIATE_DIR
    
    # Select global reference GPS coordinate (DJI_0910 start frame coords)
    ref_lat, ref_lon = 25.436562, 81.841327
    
    unified_global_data = []
    
    # Process each scene
    for scene_id in scene_ids:
        print(f"Processing scene: {scene_id}")
        json_path = os.path.join(intermediate_dir, scene_id, f"{scene_id}_tracks_stabilized.json")
        # Fallback to unstabilized if stabilized doesn't exist
        if not os.path.exists(json_path):
            json_path = os.path.join(intermediate_dir, scene_id, f"{scene_id}_tracks.json")
            print(f"  Warning: Stabilized file not found. Falling back to {json_path}")
            
        metadata_csv_path = os.path.join(input_video_dir, f"{scene_id}_metadata.csv")
        
        if not os.path.exists(json_path) or not os.path.exists(metadata_csv_path):
            print(f"  Skipping scene {scene_id} - missing track data or metadata.")
            continue
            
        # Load tracks
        with open(json_path, 'r') as f:
            tracks_data = json.load(f)
            
        # Load metadata CSV
        metadata = {}
        with open(metadata_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame_id = int(row['frame_id'])
                metadata[frame_id] = {
                    'lat': float(row['latitude']),
                    'lon': float(row['longitude']),
                    'alt': float(row['altitude']),
                    'focal_length': float(row['focal_length']) / 10.0 # Convert tenths of mm to mm
                }
                
        # We align by mapping tracking coordinates into world space meters relative to our reference point.
        # Target image resolution is 4K (3840 x 2160 pixels)
        img_w, img_h = 3840, 2160
        cx, cy = img_w / 2.0, img_h / 2.0
        
        # Camera sensor width for DJI Mavic 3 is ~13.2mm (standard 1-inch sensor size is 13.2mm x 8.8mm)
        sensor_width = 13.2
        
        scene_points_count = 0
        
        for frame_id_str, frame_tracks in tracks_data.items():
            frame_id = int(frame_id_str)
            if frame_id not in metadata:
                continue
                
            frame_meta = metadata[frame_id]
            lat = frame_meta['lat']
            lon = frame_meta['lon']
            alt = frame_meta['alt']
            focal_len = frame_meta['focal_length']
            
            # Calculate pixel to meters scale factor based on camera altitude and focal length
            # Scale = Altitude * sensor_width / (focal_length * image_width)
            if focal_len > 0:
                pixel_scale = (alt * sensor_width) / (focal_len * img_w)
            else:
                pixel_scale = 0.05 # Fallback: 5cm per pixel
                
            # Drone offset relative to reference center in meters
            drone_dx, drone_dy = convert_gps_to_local_meters(lat, lon, ref_lat, ref_lon)
            
            for track_id_str, track_info in frame_tracks.items():
                center = track_info["center"]
                obb = track_info["obb"]
                class_id = track_info["class"]
                
                # Project center relative to image principal point (cx, cy)
                px_dx = center[0] - cx
                px_dy = cy - center[1] # Flip Y to make standard Cartesian space (North is positive Y)
                
                # Transform to meters
                world_x = drone_dx + (px_dx * pixel_scale)
                world_y = drone_dy + (px_dy * pixel_scale)
                
                # Project OBB corners similarly
                world_obb = []
                for pt in obb:
                    pt_dx = pt[0] - cx
                    pt_dy = cy - pt[1]
                    wx = drone_dx + (pt_dx * pixel_scale)
                    wy = drone_dy + (pt_dy * pixel_scale)
                    world_obb.append([wx, wy])
                    
                unified_global_data.append({
                    "scene_id": scene_id,
                    "frame_id": frame_id,
                    "track_id": int(track_id_str),
                    "world_center_x": world_x,
                    "world_center_y": world_y,
                    "world_obb": world_obb,
                    "class": class_id
                })
                scene_points_count += 1
                
        print(f"  Aligned {scene_points_count} track points for scene {scene_id}")
        
    # Save global database
    output_json_path = os.path.join(intermediate_dir, "unified_world_trajectories.json")
    print(f"Saving unified global world trajectories to {output_json_path}")
    with open(output_json_path, 'w') as f:
        json.dump(unified_global_data, f, indent=4)
        
    print("Multi-Video alignment complete.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=str, default="DJI_0910,DJI_0911,DJI_0912,DJI_0914,DJI_0915,DJI_0916")
    args = parser.parse_args()
    scene_list = [s.strip() for s in args.scenes.split(",")]
    run_alignment(scene_list)
