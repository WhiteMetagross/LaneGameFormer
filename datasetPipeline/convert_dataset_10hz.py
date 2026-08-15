import os
import json
import shutil
import subprocess
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import concurrent.futures

# Configuration
SOURCE_DIR = Path("ChunkedProjectPrayagBEVDataset")
TARGET_DIR = Path("ChunkedProjectPrayagBEVDataset10Hz")
DOWNSAMPLE_FACTOR = 3
TARGET_FPS = 10

def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)

def process_metadata_csv(src_path, dst_path):
    """
    Process metadata.csv: Keep frame_id % 3 == 0, reindex frame_id.
    Assumes 0-based indexing.
    """
    try:
        df = pd.read_csv(src_path)
        # Filter
        df_downsampled = df[df['frame_id'] % DOWNSAMPLE_FACTOR == 0].copy()
        # Re-index
        df_downsampled['frame_id'] = df_downsampled['frame_id'] // DOWNSAMPLE_FACTOR
        
        df_downsampled.to_csv(dst_path, index=False)
    except Exception as e:
        print(f"Error processing metadata {src_path}: {e}")

def process_tracks_csv(src_path, dst_path):
    """
    Process tracks.csv: Keep (frame_id - 1) % 3 == 0, reindex frame_id.
    Assumes 1-based indexing.
    """
    try:
        df = pd.read_csv(src_path)
        # Filter: (frame_id - 1) % 3 == 0 implies frames 1, 4, 7...
        df_downsampled = df[(df['frame_id'] - 1) % DOWNSAMPLE_FACTOR == 0].copy()
        # Re-index: 1->1, 4->2, 7->3 => (f-1)//3 + 1
        df_downsampled['frame_id'] = (df_downsampled['frame_id'] - 1) // DOWNSAMPLE_FACTOR + 1
        
        df_downsampled.to_csv(dst_path, index=False)
    except Exception as e:
        print(f"Error processing tracks csv {src_path}: {e}")

def process_tracks_json(src_path, dst_path):
    """
    Process tracks.json: Filter keys, rekey.
    """
    try:
        with open(src_path, 'r') as f:
            data = json.load(f)
        
        new_data = {}
        for frame_id_str, content in data.items():
            frame_id = int(frame_id_str)
            if (frame_id - 1) % DOWNSAMPLE_FACTOR == 0:
                new_frame_id = (frame_id - 1) // DOWNSAMPLE_FACTOR + 1
                new_data[str(new_frame_id)] = content
        
        with open(dst_path, 'w') as f:
            json.dump(new_data, f, indent=None, separators=(',', ':')) # Minified to save space
    except Exception as e:
        print(f"Error processing tracks json {src_path}: {e}")

def process_road_annotation(src_path, dst_path):
    """
    Process road_annotation.json: Update reference frame number.
    """
    try:
        with open(src_path, 'r') as f:
            data = json.load(f)
        
        if 'reference_video_frame_number' in data:
            data['reference_video_frame_number'] = data['reference_video_frame_number'] // DOWNSAMPLE_FACTOR
            
        with open(dst_path, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error processing road annotation {src_path}: {e}")

def to_win_path(path):
    p_str = str(Path(path).as_posix())
    if p_str.startswith("/mnt/c/"):
        return "C:/" + p_str[7:]
    return p_str

def process_video(src_path, dst_path):
    """
    Process video using ffmpeg.
    """
    try:
        # ffmpeg command to change fps
        # -r 10 sets the output frame rate. 
        # We assume input is 30fps. ffmpeg dropping behavior usually aligns with keeping first frame.
        cmd = [
            'ffmpeg.exe', '-y', # Overwrite
            '-i', to_win_path(src_path),
            '-filter:v', f'fps={TARGET_FPS}',
            to_win_path(dst_path)
        ]
        # Run quietly
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"Error processing video {src_path}: {e}")

def process_manifest(src_path, dst_path):
    """
    Update chunk_manifest.json
    """
    try:
        with open(src_path, 'r') as f:
            data = json.load(f)
        
        # Update parameters
        if 'parameters' in data:
            data['parameters']['fps'] = TARGET_FPS
            if 'chunk_frames' in data['parameters']:
                data['parameters']['chunk_frames'] = data['parameters']['chunk_frames'] // DOWNSAMPLE_FACTOR
            if 'gap_frames' in data['parameters']:
                data['parameters']['gap_frames'] = data['parameters']['gap_frames'] // DOWNSAMPLE_FACTOR
        
        # Update splits
        if 'splits' in data:
            for split_name, chunks in data['splits'].items():
                for chunk in chunks:
                    if 'start_frame' in chunk:
                        chunk['start_frame'] = chunk['start_frame'] // DOWNSAMPLE_FACTOR
                    if 'end_frame' in chunk:
                        chunk['end_frame'] = chunk['end_frame'] // DOWNSAMPLE_FACTOR
                    if 'num_frames' in chunk:
                        chunk['num_frames'] = chunk['num_frames'] // DOWNSAMPLE_FACTOR
                    # avg_vehicles_per_frame remains roughly the same, no update needed
        
        with open(dst_path, 'w') as f:
            json.dump(data, f, indent=2)
            
    except Exception as e:
        print(f"Error processing manifest {src_path}: {e}")

def process_file(file_info):
    src_file, dst_file, file_type = file_info
    
    if file_type == 'video':
        process_video(src_file, dst_file)
    elif file_type == 'metadata':
        process_metadata_csv(src_file, dst_file)
    elif file_type == 'tracks_csv':
        process_tracks_csv(src_file, dst_file)
    elif file_type == 'tracks_json':
        process_tracks_json(src_file, dst_file)
    elif file_type == 'road_annotation':
        process_road_annotation(src_file, dst_file)
    elif file_type == 'manifest':
        process_manifest(src_file, dst_file)
    elif file_type == 'copy':
        shutil.copy2(src_file, dst_file)

def main():
    if not SOURCE_DIR.exists():
        print(f"Source directory {SOURCE_DIR} not found.")
        return

    ensure_dir(TARGET_DIR)
    
    tasks = []
    
    # Walk through source directory
    for root, dirs, files in os.walk(SOURCE_DIR):
        rel_root = Path(root).relative_to(SOURCE_DIR)
        target_root = TARGET_DIR / rel_root
        ensure_dir(target_root)
        
        for file in files:
            src_file = Path(root) / file
            dst_file = target_root / file
            
            if file == 'chunk_manifest.json':
                tasks.append((src_file, dst_file, 'manifest'))
            elif file.endswith('.mp4'):
                tasks.append((src_file, dst_file, 'video'))
            elif 'metadata.csv' in file:
                tasks.append((src_file, dst_file, 'metadata'))
            elif 'tracks.csv' in file:
                tasks.append((src_file, dst_file, 'tracks_csv'))
            elif 'tracks.json' in file:
                tasks.append((src_file, dst_file, 'tracks_json'))
            elif 'road_annotation.json' in file:
                tasks.append((src_file, dst_file, 'road_annotation'))
            else:
                # Copy other files (README, txt lists, etc.)
                tasks.append((src_file, dst_file, 'copy'))

    print(f"Found {len(tasks)} files to process.")
    
    # Process in parallel
    # Adjust max_workers based on system capabilities. 
    # Video processing is CPU intensive, so maybe lower workers if many videos.
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        list(tqdm(executor.map(process_file, tasks), total=len(tasks), unit="file"))

    print("Conversion complete.")

if __name__ == "__main__":
    main()
