import os
import sys
import json
import csv
import numpy as np
import argparse
import config

def obb_to_center_width_height_angle(obb):
    # corners order: BL, BR, TR, TL (bottom-left, bottom-right, top-right, top-left)
    pts = np.array(obb, dtype=np.float32)
    center = np.mean(pts, axis=0)
    
    # Vector from point 0 to point 1 (width direction / heading)
    v01 = pts[1] - pts[0]
    w = np.linalg.norm(v01)
    
    # Vector from point 1 to point 2 (height direction)
    v12 = pts[2] - pts[1]
    h = np.linalg.norm(v12)
    
    angle = np.arctan2(v01[1], v01[0])
    return center, w, h, angle

def reconstruct_obb_corners(center, w, h, angle):
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    half_w = w / 2.0
    half_h = h / 2.0
    corners_local = np.array([
        [-half_w, -half_h],
        [half_w, -half_h],
        [half_w, half_h],
        [-half_w, half_h]
    ], dtype=np.float32)
    
    rotation_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ], dtype=np.float32)
    
    rotated_corners = corners_local @ rotation_matrix.T
    return (rotated_corners + np.array(center)).tolist()

def rts_smooth_centers(coords, dt=1.0/30.0, q_noise=0.1, r_noise=1.5):
    N = len(coords)
    if N < 3:
        return coords # Too short to smooth
        
    # State: [x, y, vx, vy, ax, ay]
    F = np.array([
        [1, 0, dt,  0, 0.5*dt**2,        0],
        [0, 1,  0, dt,        0, 0.5*dt**2],
        [0, 0,  1,  0,       dt,        0],
        [0, 0,  0,  1,        0,       dt],
        [0, 0,  0,  0,        1,        0],
        [0, 0,  0,  0,        0,        1]
    ], dtype=np.float32)
    
    H = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0]
    ], dtype=np.float32)
    
    Q = np.eye(6, dtype=np.float32) * q_noise
    R = np.eye(2, dtype=np.float32) * r_noise
    
    # Forward Kalman Pass
    x = np.zeros(6, dtype=np.float32)
    x[0:2] = coords[0]
    P = np.eye(6, dtype=np.float32) * 10.0
    
    xs_f = np.zeros((N, 6), dtype=np.float32)
    Ps_f = np.zeros((N, 6, 6), dtype=np.float32)
    xs_pred = np.zeros((N, 6), dtype=np.float32)
    Ps_pred = np.zeros((N, 6, 6), dtype=np.float32)
    
    for k in range(N):
        if k == 0:
            x_pred = x
            P_pred = P
        else:
            x_pred = F @ xs_f[k-1]
            P_pred = F @ Ps_f[k-1] @ F.T + Q
            
        xs_pred[k] = x_pred
        Ps_pred[k] = P_pred
        
        z = coords[k]
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x = x_pred + K @ (z - H @ x_pred)
        P = (np.eye(6) - K @ H) @ P_pred
        
        xs_f[k] = x
        Ps_f[k] = P
        
    # Backward RTS smoothing pass
    xs_s = np.zeros((N, 6), dtype=np.float32)
    xs_s[-1] = xs_f[-1]
    Ps_s = np.zeros((N, 6, 6), dtype=np.float32)
    Ps_s[-1] = Ps_f[-1]
    
    for k in range(N-2, -1, -1):
        # Prevent division by zero/invertibility issues
        try:
            inv_P_pred = np.linalg.inv(Ps_pred[k+1])
        except np.linalg.LinAlgError:
            inv_P_pred = np.linalg.pinv(Ps_pred[k+1])
            
        C = Ps_f[k] @ F.T @ inv_P_pred
        xs_s[k] = xs_f[k] + C @ (xs_s[k+1] - xs_pred[k+1])
        Ps_s[k] = Ps_f[k] + C @ (Ps_s[k+1] - Ps_pred[k+1]) @ C.T
        
    return xs_s[:, 0:2]

def smooth_and_align_headings(angles, centers, dt=1.0/30.0):
    N = len(angles)
    if N < 2:
        return angles
        
    # Compute velocity vectors for heading direction check
    velocities = np.zeros((N, 2))
    for i in range(N):
        if i > 0 and i < N - 1:
            velocities[i] = (centers[i+1] - centers[i-1]) / (2.0 * dt)
        elif i > 0:
            velocities[i] = (centers[i] - centers[i-1]) / dt
        else:
            velocities[i] = (centers[i+1] - centers[i]) / dt
            
    corrected_angles = np.zeros(N)
    for i in range(N):
        angle = angles[i]
        vx, vy = velocities[i]
        speed = np.hypot(vx, vy)
        
        # 1. Heading flip resolver (only active when vehicle is moving significantly)
        if speed > 0.5:
            theta_motion = np.arctan2(vy, vx)
            # Dot product to check if heading matches motion direction
            dot = np.cos(angle)*np.cos(theta_motion) + np.sin(angle)*np.sin(theta_motion)
            if dot < 0:
                angle = angle + np.pi
                
        corrected_angles[i] = np.arctan2(np.sin(angle), np.cos(angle)) # wrap to -pi, pi
        
    # 2. Circular statistics rolling average (window length = 5)
    window = 5
    smoothed_angles = np.zeros(N)
    half_win = window // 2
    for i in range(N):
        start = max(0, i - half_win)
        end = min(N, i + half_win + 1)
        win_angles = corrected_angles[start:end]
        mean_x = np.mean(np.cos(win_angles))
        mean_y = np.mean(np.sin(win_angles))
        smoothed_angles[i] = np.arctan2(mean_y, mean_x)
        
    return smoothed_angles

def run_smoothing(scene_id):
    print(f"\n=================== Smoothing Trajectories for Scene: {scene_id} ===================")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    intermediate_dir = config.INTERMEDIATE_DIR
    
    # We load the stabilized track files (which already has camera-motion corrected)
    json_path = os.path.join(intermediate_dir, scene_id, f"{scene_id}_tracks_stabilized.json")
    if not os.path.exists(json_path):
        # Fallback to unstabilized tracks
        json_path = os.path.join(intermediate_dir, scene_id, f"{scene_id}_tracks.json")
        print(f"  Warning: Stabilized tracks not found. Falling back to {json_path}")
        
    if not os.path.exists(json_path):
        print(f"Error: No tracking files found for scene {scene_id}")
        return False
        
    print(f"Loading track data: {json_path}")
    with open(json_path, 'r') as f:
        tracking_data = json.load(f)
        
    # Group records by track_id
    # track_id -> list of dict with frame_id, center, obb, class
    tracks_by_id = {}
    
    for frame_id_str, tracks in tracking_data.items():
        frame_id = int(frame_id_str)
        for track_id_str, track_info in tracks.items():
            track_id = int(track_id_str)
            if track_id not in tracks_by_id:
                tracks_by_id[track_id] = []
                
            center, w, h, angle = obb_to_center_width_height_angle(track_info["obb"])
            tracks_by_id[track_id].append({
                "frame_id": frame_id,
                "center": center,
                "w": w,
                "h": h,
                "angle": angle,
                "class": track_info["class"]
            })
            
    print(f"Found {len(tracks_by_id)} unique tracks to smooth.")
    
    # Process each track sequence
    smoothed_tracks_by_id = {}
    for track_id, seq in tracks_by_id.items():
        # Sort by frame_id
        seq.sort(key=lambda x: x["frame_id"])
        N = len(seq)
        
        centers = np.array([item["center"] for item in seq])
        widths = [item["w"] for item in seq]
        heights = [item["h"] for item in seq]
        angles = [item["angle"] for item in seq]
        
        # 1. Size Locking: use median width and height across the entire sequence
        locked_w, locked_h = np.median(widths), np.median(heights)
        
        # 2. RTS Kalman coordinate smoothing on centers
        smoothed_centers = rts_smooth_centers(centers)
        
        # 3. Heading ambiguity flipping & circular statistical smoothing
        smoothed_angles = smooth_and_align_headings(angles, smoothed_centers)
        
        # Reconstruct smoothed oriented boxes
        smoothed_seq = []
        for idx in range(N):
            reconstructed_corners = reconstruct_obb_corners(
                smoothed_centers[idx],
                locked_w,
                locked_h,
                smoothed_angles[idx]
            )
            smoothed_seq.append({
                "frame_id": seq[idx]["frame_id"],
                "center": smoothed_centers[idx].tolist(),
                "obb": reconstructed_corners,
                "class": seq[idx]["class"]
            })
            
        smoothed_tracks_by_id[track_id] = smoothed_seq
        
    # Reassemble tracking data back into frame-wise format
    smoothed_frame_data = {}
    total_points = 0
    
    for track_id, seq in smoothed_tracks_by_id.items():
        for item in seq:
            frame_id_str = str(item["frame_id"])
            if frame_id_str not in smoothed_frame_data:
                smoothed_frame_data[frame_id_str] = {}
                
            smoothed_frame_data[frame_id_str][str(track_id)] = {
                "center": item["center"],
                "obb": item["obb"],
                "class": item["class"]
            }
            total_points += 1
            
    # Save smoothed tracks over stabilized file to integrate seamlessly
    output_path = os.path.join(intermediate_dir, scene_id, f"{scene_id}_tracks_stabilized.json")
    print(f"Saving smoothed OBB trajectories directly to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(smoothed_frame_data, f, indent=4)
        
    # Also update the companion CSV file
    csv_path = os.path.join(intermediate_dir, scene_id, f"{scene_id}_tracks_stabilized.csv")
    print(f"Saving smoothed OBB CSV to: {csv_path}")
    header = [
        'scene_id', 'frame_id', 'track_id', 
        'center_x', 'center_y', 
        'obb_corner1_x', 'obb_corner1_y',
        'obb_corner2_x', 'obb_corner2_y',
        'obb_corner3_x', 'obb_corner3_y',
        'obb_corner4_x', 'obb_corner4_y'
    ]
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for frame_id_str, tracks in sorted(smoothed_frame_data.items(), key=lambda x: int(x[0])):
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
                
    print(f"OBB Kinematic Smoothing complete. Processed {total_points} tracking coordinates.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-id", type=str, default="DJI_0916")
    args = parser.parse_args()
    run_smoothing(args.scene_id)
