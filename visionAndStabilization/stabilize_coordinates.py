import os
import sys
import json
import csv
import cv2
import numpy as np
import argparse
import config

def stabilize_points(points, H):
    if H is None:
        return points
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    warped_pts = cv2.perspectiveTransform(pts, H)
    return warped_pts.reshape(-1, 2).tolist()

def run_stabilization(scene_id):
    print(f"\n=================== Stabilizing Scene: {scene_id} ===================")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_video_dir = config.INPUT_DIR
    intermediate_dir = config.INTERMEDIATE_DIR
    
    video_path = os.path.join(input_video_dir, f"{scene_id}.mp4")
    if not os.path.exists(video_path):
        video_path = os.path.join(base_dir, "temp_videos", f"{scene_id}.mp4")
        
    json_path = os.path.join(intermediate_dir, scene_id, f"{scene_id}_tracks.json")
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return False
    if not os.path.exists(json_path):
        print(f"Error: Tracking JSON file not found at {json_path}")
        return False
        
    print(f"Loading video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return False
        
    print(f"Loading tracking data: {json_path}")
    with open(json_path, 'r') as f:
        tracking_data = json.load(f)
        
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame of video.")
        cap.release()
        return False
        
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    print("Detecting background features in the reference frame...")
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
        return False
        
    print(f"Successfully detected {len(prev_pts)} initial features.")
    ref_pts = prev_pts.copy()
    homographies = {1: np.eye(3, dtype=np.float32)}
    
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03)
    )
    
    frame_idx = 1
    last_valid_H = np.eye(3, dtype=np.float32)
    
    print("Tracking features and calculating frame homographies...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, prev_pts, None, **lk_params)
        
        valid_prev = prev_pts[status == 1]
        valid_next = next_pts[status == 1]
        valid_ref = ref_pts[status == 1]
        
        H = None
        if len(valid_next) >= 10:
            H, inliers = cv2.findHomography(valid_next, valid_ref, cv2.RANSAC, 5.0)
            
        if H is not None:
            homographies[frame_idx] = H
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
            homographies[frame_idx] = last_valid_H
            prev_pts = valid_next.reshape(-1, 1, 2) if len(valid_next) > 0 else prev_pts
            ref_pts = valid_ref.reshape(-1, 1, 2) if len(valid_ref) > 0 else ref_pts
            
        prev_gray = frame_gray.copy()
        if frame_idx % 100 == 0:
            print(f"  Processed frame {frame_idx}...")
            
    cap.release()
    print(f"Finished processing video. Total frames: {frame_idx}")
    
    print("Applying homographies to tracking coordinates...")
    stabilized_data = {}
    total_points = 0
    
    for frame_id_str, tracks in tracking_data.items():
        frame_id = int(frame_id_str)
        H = homographies.get(frame_id, last_valid_H)
        stabilized_data[frame_id_str] = {}
        for track_id_str, track_info in tracks.items():
            center = track_info["center"]
            obb = track_info["obb"]
            class_id = track_info["class"]
            
            stabilized_center = stabilize_points([center], H)[0]
            stabilized_obb = stabilize_points(obb, H)
            
            stabilized_data[frame_id_str][track_id_str] = {
                "center": stabilized_center,
                "obb": stabilized_obb,
                "class": class_id
            }
            total_points += 1
            
    stabilized_json_path = os.path.join(intermediate_dir, scene_id, f"{scene_id}_tracks_stabilized.json")
    print(f"Saving stabilized JSON to {stabilized_json_path}")
    with open(stabilized_json_path, 'w') as f:
        json.dump(stabilized_data, f, indent=4)
        
    stabilized_csv_path = os.path.join(intermediate_dir, scene_id, f"{scene_id}_tracks_stabilized.csv")
    print(f"Saving stabilized CSV to {stabilized_csv_path}")
    header = [
        'scene_id', 'frame_id', 'track_id', 
        'center_x', 'center_y', 
        'obb_corner1_x', 'obb_corner1_y',
        'obb_corner2_x', 'obb_corner2_y',
        'obb_corner3_x', 'obb_corner3_y',
        'obb_corner4_x', 'obb_corner4_y'
    ]
    
    with open(stabilized_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for frame_id_str, tracks in stabilized_data.items():
            frame_id = int(frame_id_str)
            for track_id_str, track_info in tracks.items():
                center = track_info["center"]
                obb = track_info["obb"]
                row = [
                    scene_id, frame_id, int(track_id_str), center[0], center[1],
                    obb[0][0], obb[0][1], obb[1][0], obb[1][1],
                    obb[2][0], obb[2][1], obb[3][0], obb[3][1]
                ]
                writer.writerow(row)
                
    print(f"Stabilization complete. Stabilized {total_points} track points.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-id", type=str, default="DJI_0916")
    args = parser.parse_args()
    run_stabilization(args.scene_id)
