import os
import sys
import json
import csv
import numpy as np
from collections import defaultdict
import argparse

def run_conventional_ttc_mining():
    print("\n=================== Starting Conventional TTC Mining (2nd Baseline) ===================")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    intermediate_dir = os.path.join(base_dir, "ProjectPrayagTopDownDataset", "intermediate_files")
    output_dir = os.path.join(base_dir, "ProjectPrayagTopDownDataset")
    
    world_trajectories_path = os.path.join(intermediate_dir, "unified_world_trajectories.json")
    
    if not os.path.exists(world_trajectories_path):
        print(f"Error: Unified world trajectories not found. Please run telemetry_aligner.py first.")
        return False
        
    print(f"Loading unified world trajectories: {world_trajectories_path}")
    with open(world_trajectories_path, 'r') as f:
        traj_data = json.load(f)
        
    # Organize data by scene -> frame -> track_id
    scene_frame_data = defaultdict(lambda: defaultdict(dict))
    agent_paths = defaultdict(list)
    
    for pt in traj_data:
        scene = pt["scene_id"]
        frame = pt["frame_id"]
        tid = pt["track_id"]
        
        scene_frame_data[scene][frame][tid] = pt
        agent_paths[(scene, tid)].append(pt)
        
    print("Calculating velocity vectors...")
    agent_velocities = {}
    dt = 1.0 / 30.0
    
    for key, path in agent_paths.items():
        path.sort(key=lambda x: x["frame_id"])
        vel_dict = {}
        for idx in range(len(path)):
            frame_id = path[idx]["frame_id"]
            if idx > 0 and idx < len(path) - 1:
                vx = (path[idx+1]["world_center_x"] - path[idx-1]["world_center_x"]) / (2.0 * dt)
                vy = (path[idx+1]["world_center_y"] - path[idx-1]["world_center_y"]) / (2.0 * dt)
            elif idx > 0:
                vx = (path[idx]["world_center_x"] - path[idx-1]["world_center_x"]) / dt
                vy = (path[idx]["world_center_y"] - path[idx-1]["world_center_y"]) / dt
            elif idx < len(path) - 1:
                vx = (path[idx+1]["world_center_x"] - path[idx]["world_center_x"]) / dt
                vy = (path[idx+1]["world_center_y"] - path[idx]["world_center_y"]) / dt
            else:
                vx, vy = 0.0, 0.0
            vel_dict[frame_id] = (vx, vy)
        agent_velocities[key] = vel_dict
        
    print("Mining interactions using Conventional TTC (1D Line-of-Sight projection)...")
    conventional_scenarios = []
    
    TTC_FILTER_THRESHOLD = 1.53 # Same statistical Q3 threshold
    SPATIAL_RADIUS = 30.0 # meters
    interaction_id = 1
    
    for (scene_id, ego_id), ego_path in agent_paths.items():
        ego_class = ego_path[0]["class"]
        # Ego must be HVE (1) or SVE (2)
        if ego_class not in [1, 2]:
            continue
            
        tp_interactions = defaultdict(list)
        
        for ego_pt in ego_path:
            frame = ego_pt["frame_id"]
            ego_x = ego_pt["world_center_x"]
            ego_y = ego_pt["world_center_y"]
            
            ego_vx, ego_vy = agent_velocities[(scene_id, ego_id)].get(frame, (0.0, 0.0))
            
            frame_agents = scene_frame_data[scene_id][frame]
            for tp_id, tp_pt in frame_agents.items():
                if tp_id == ego_id:
                    continue
                    
                tp_x = tp_pt["world_center_x"]
                tp_y = tp_pt["world_center_y"]
                
                dist = np.hypot(tp_x - ego_x, tp_y - ego_y)
                if dist > SPATIAL_RADIUS:
                    continue
                    
                tp_vx, tp_vy = agent_velocities[(scene_id, tp_id)].get(frame, (0.0, 0.0))
                
                # Relative vectors
                dx = tp_x - ego_x
                dy = tp_y - ego_y
                dvx = tp_vx - ego_vx
                dvy = tp_vy - ego_vy
                
                dot_product = dx * dvx + dy * dvy
                
                # If dot_product < 0, they are approaching
                if dot_product < 0:
                    # Closing velocity along the line connecting their centers
                    v_closing = - dot_product / dist
                    
                    if v_closing > 0.01:
                        # Conventional 1D Time-to-Collision (TTC)
                        ttc = dist / v_closing
                        
                        if ttc <= TTC_FILTER_THRESHOLD:
                            tp_interactions[tp_id].append((frame, ttc, dist))
                            
        # Post-process: group interacting frames into scenarios
        for tp_id, frames in tp_interactions.items():
            if len(frames) < 15:
                continue
                
            frames.sort(key=lambda x: x[0])
            segments = []
            current_segment = [frames[0]]
            
            for idx in range(1, len(frames)):
                if frames[idx][0] - frames[idx-1][0] <= 5:
                    current_segment.append(frames[idx])
                else:
                    if len(current_segment) >= 15:
                        segments.append(current_segment)
                    current_segment = [frames[idx]]
            if len(current_segment) >= 15:
                segments.append(current_segment)
                
            for seg in segments:
                start_frame = seg[0][0]
                end_frame = seg[-1][0]
                mean_ttc = np.mean([item[1] for item in seg])
                min_dist = np.min([item[2] for item in seg])
                
                conventional_scenarios.append({
                    "interaction_id": interaction_id,
                    "scene_id": scene_id,
                    "ego_id": ego_id,
                    "tp_id": tp_id,
                    "tp_class": agent_paths[(scene_id, tp_id)][0]["class"],
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "duration_seconds": (end_frame - start_frame) / 30.0,
                    "mean_conventional_ttc": float(mean_ttc),
                    "min_distance_meters": float(min_dist)
                })
                interaction_id += 1
                
    output_json_path = os.path.join(output_dir, "conventional_ttc_scenarios.json")
    print(f"Saving conventional TTC scenarios library to {output_json_path}")
    with open(output_json_path, 'w') as f:
        json.dump(conventional_scenarios, f, indent=4)
        
    print(f"Conventional TTC mining complete. Mined {len(conventional_scenarios)} baseline scenarios using 1D line-of-sight projection.")
    return True

if __name__ == "__main__":
    run_conventional_ttc_mining()
