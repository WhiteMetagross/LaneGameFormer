#This script converts all videos in the ProjectPrayagTopDownDataset/CIRAerialDroneIndianIntersectionsVideoes folder to 30 FPS.
#This script overwrites the original files after successful conversion.
#This script uses OpenCV to read frames and resample them to the target frame rate.
#This script processes videos sequentially to ensure stability and correct file handling.
#This script ensures frame alignment with tracking data annotations by using frame skipping method.

import os
import glob
import cv2
import tempfile
import shutil
import gc
from tqdm import tqdm

import os
import glob
import cv2
import tempfile
import shutil
import gc
from tqdm import tqdm

DATASET_DIR = os.path.join('ProjectPrayagTopDownDataset', 'CIRAerialDroneIndianIntersectionsVideoes')
TARGET_FPS = 30.0
SUPPORTED_EXTS = ['.mp4', '.mov', '.avi', '.MP4', '.MOV', '.AVI']


def find_videos(folder):
    #This function finds video files in the given folder matching supported extensions.
    files = []
    for ext in SUPPORTED_EXTS:
        files.extend(glob.glob(os.path.join(folder, f'*{ext}')))
    #Remove duplicates by using set with lowercase paths then map back to original.
    seen = set()
    unique_files = []
    for f in files:
        f_lower = f.lower()
        if f_lower not in seen:
            seen.add(f_lower)
            unique_files.append(f)
    return sorted(unique_files)


def convert_video_to_30fps(src_path, target_fps=TARGET_FPS):
    #This function converts a single video to target_fps and replaces the original file.
    print(f'Processing: {src_path}')
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        print('  Error: could not open video. Skipping.')
        return False

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if orig_fps <= 0 or frame_count == 0 or width == 0 or height == 0:
        print('  Warning: could not determine video properties. Skipping.')
        cap.release()
        return False

    print(f'  Original FPS: {orig_fps:.2f}, Frames: {frame_count}, Resolution: {width}x{height}')

    if abs(orig_fps - target_fps) < 0.01:
        print('  Already at target FPS. No conversion needed.')
        cap.release()
        return True

    total_dst_frames = int(round(frame_count * (target_fps / orig_fps)))
    if total_dst_frames <= 0:
        print('  Computed zero destination frames. Skipping.')
        cap.release()
        return False

    dir_name, base_name = os.path.split(src_path)
    name, ext = os.path.splitext(base_name)
    fd, tmp_path = tempfile.mkstemp(suffix=ext, prefix=name + '_tmp_', dir=dir_name)
    os.close(fd)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(tmp_path, fourcc, target_fps, (width, height))
    if not writer.isOpened():
        print('  Error: could not create video writer. Skipping.')
        cap.release()
        return False

    #For each destination frame index, use frame skipping to match tracking data alignment.
    #This formula ensures output_frame[i] corresponds to source_frame[floor(i * orig_fps / target_fps)].
    frame_skip = orig_fps / target_fps
    current_src_idx = -1
    current_frame = None
    
    #Create progress bar for this video.
    video_name = os.path.basename(src_path)
    pbar = tqdm(total=total_dst_frames, desc=f'{video_name}', unit='frames', leave=True)
    
    for dst_idx in range(total_dst_frames):
        src_idx = int(dst_idx * frame_skip)
        if src_idx >= frame_count:
            src_idx = frame_count - 1
        
        #Only read a new frame if the source index changed.
        if src_idx != current_src_idx:
            #Release previous frame memory before reading new one.
            current_frame = None
            cap.set(cv2.CAP_PROP_POS_FRAMES, src_idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                current_frame = frame
                current_src_idx = src_idx
            elif current_frame is None:
                #Try to read any frame if we have none yet.
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret2, frame2 = cap.read()
                if ret2 and frame2 is not None:
                    current_frame = frame2
                else:
                    pbar.close()
                    print(f'  Warning: failed to read any frame. Stopping.')
                    break
        
        if current_frame is not None:
            writer.write(current_frame)
        pbar.update(1)
    
    pbar.close()
    
    #Explicitly release frame memory.
    current_frame = None

    writer.release()
    cap.release()
    
    #Force garbage collection to free memory.
    gc.collect()

    #Replace original file with temp file safely.
    backup_path = src_path + '.bak'
    try:
        shutil.move(src_path, backup_path)
        shutil.move(tmp_path, src_path)
        os.remove(backup_path)
        print(f'  Converted and replaced: {src_path} -> {target_fps:.0f} FPS.')
        return True
    except Exception as e:
        print(f'  Error during replacement: {e}')
        #Attempt cleanup and restore original
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if os.path.exists(backup_path):
            shutil.move(backup_path, src_path)
        return False


if __name__ == '__main__':
    #This script processes all supported video files in the dataset folder sequentially.
    if not os.path.isdir(DATASET_DIR):
        print(f'Error: dataset folder not found: {DATASET_DIR}')
        exit(1)

    videos = find_videos(DATASET_DIR)
    if not videos:
        print('No videos found to process.')
        exit(0)

    print(f'Found {len(videos)} video(s) to process sequentially.')
    for i, v in enumerate(videos, 1):
        print(f'  {i}. {os.path.basename(v)}')
    print('=' * 60)
    
    success_count = 0
    failed_videos = []
    
    for video_path in videos:
        video_name = os.path.basename(video_path)
        try:
            result = convert_video_to_30fps(video_path, TARGET_FPS)
            if result:
                success_count += 1
                print(f'[SUCCESS] {video_name}')
            else:
                failed_videos.append(video_name)
                print(f'[FAILED] {video_name}')
        except Exception as e:
            failed_videos.append(video_name)
            print(f'[ERROR] {video_name}: {e}')
        print('-' * 60)
    
    print('=' * 60)
    print(f'Finished. Successful conversions: {success_count}/{len(videos)}.')
    if failed_videos:
        print(f'Failed videos: {", ".join(failed_videos)}')
