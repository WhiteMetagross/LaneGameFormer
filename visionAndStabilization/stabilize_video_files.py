import os
import sys
import cv2
import numpy as np
import argparse
import shutil
import config

def run_video_stabilization(scene_id):
    print(f"\n=================== Stabilizing Video Frames for Scene: {scene_id} ===================")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_video_dir = config.INPUT_DIR
    temp_video_dir = os.path.join(config.BASE_DIR, "temp_videos")
    
    # Locate original video
    video_path = os.path.join(input_video_dir, f"{scene_id}.mp4")
    in_temp = False
    if not os.path.exists(video_path):
        video_path = os.path.join(temp_video_dir, f"{scene_id}.mp4")
        in_temp = True
        
    if not os.path.exists(video_path):
        print(f"Error: Original video file not found at {video_path}")
        return False
        
    print(f"Loading video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return False
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video Info: {w}x{h} @ {fps:.2f} FPS. Total frames: {total_frames}")
    
    # Setup temporary output path for writing stabilized video
    temp_out_path = os.path.join(input_video_dir, f"{scene_id}_stabilized_temp.mp4")
    
    # Use standard mp4v codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_out_path, fourcc, fps, (w, h))
    
    # Read first frame as reference
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        cap.release()
        out.release()
        return False
        
    # Write first frame unchanged
    out.write(prev_frame)
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Detect Harris corner features in reference frame
    feature_params = dict(
        maxCorners=500,
        qualityLevel=0.01,
        minDistance=15,
        blockSize=7,
        useHarrisDetector=True,
        k=0.04
    )
    
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, **feature_params)
    if prev_pts is None or len(prev_pts) < 10:
        print("Error: Too few features detected in the reference frame.")
        cap.release()
        out.release()
        return False
        
    ref_pts = prev_pts.copy()
    
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03)
    )
    
    frame_idx = 1
    last_valid_H = np.eye(3, dtype=np.float32)
    
    print("Stabilizing video frames and writing to temporary file...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Track features
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, prev_pts, None, **lk_params)
        
        valid_prev = prev_pts[status == 1]
        valid_next = next_pts[status == 1]
        valid_ref = ref_pts[status == 1]
        
        H = None
        if len(valid_next) >= 10:
            H, inliers = cv2.findHomography(valid_next, valid_ref, cv2.RANSAC, 5.0)
            
        if H is not None:
            last_valid_H = H
            if len(valid_next) < 150:
                H_inv = np.linalg.inv(H)
                new_ref_pts = cv2.goodFeaturesToTrack(prev_gray, **feature_params)
                if new_ref_pts is not None:
                    ref_pts = new_ref_pts
                    prev_pts = cv2.perspectiveTransform(new_ref_pts, H_inv)
                else:
                    prev_pts = valid_next.reshape(-1, 1, 2)
                    ref_pts = valid_ref.reshape(-1, 1, 2)
            else:
                prev_pts = valid_next.reshape(-1, 1, 2)
                ref_pts = valid_ref.reshape(-1, 1, 2)
        else:
            H = last_valid_H
            prev_pts = valid_next.reshape(-1, 1, 2) if len(valid_next) > 0 else prev_pts
            ref_pts = valid_ref.reshape(-1, 1, 2) if len(valid_ref) > 0 else ref_pts
            
        # Warp the image frame using computed homography
        warped_frame = cv2.warpPerspective(frame, H, (w, h))
        out.write(warped_frame)
        
        prev_gray = frame_gray.copy()
        
        if frame_idx % 100 == 0:
            print(f"  Processed and stabilized frame {frame_idx}/{total_frames}...")
            
    cap.release()
    out.release()
    print(f"Finished stabilizing video frames. Total frames written: {frame_idx}")
    
    # Overwrite the original raw video in config.INPUT_DIR (CIRAerialDroneIndianIntersectionsVideoes)
    final_video_path = os.path.join(input_video_dir, f"{scene_id}.mp4")
    
    # Backup original video if we are overwriting it in place
    if not in_temp:
        bak_video_path = final_video_path + ".bak"
        if not os.path.exists(bak_video_path):
            shutil.copy(final_video_path, bak_video_path)
            print(f"  Backed up original video to {bak_video_path}")
            
    # Copy stabilized video to config.INPUT_DIR
    shutil.copy(temp_out_path, final_video_path)
    os.remove(temp_out_path)
    print(f"Stabilized video successfully written to: {final_video_path}")
    
    # Also overwrite original video in temp_videos if it was originally there
    if in_temp:
        bak_temp_path = video_path + ".bak"
        if not os.path.exists(bak_temp_path):
            shutil.copy(video_path, bak_temp_path)
            print(f"  Backed up original temp video to {bak_temp_path}")
        shutil.copy(final_video_path, video_path)
        print(f"  Overwrote original temp video in: {video_path}")
        
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-Video Frame-Level Stabilization Module")
    parser.add_argument("--scene-id", type=str, default="DJI_0916", help="Scene ID to stabilize")
    args = parser.parse_args()
    
    run_video_stabilization(args.scene_id)
