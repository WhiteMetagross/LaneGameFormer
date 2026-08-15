import os
import sys
import json
import numpy as np
from collections import defaultdict
import argparse

def run_interaction_mining():
    print("\n=================== Starting Interaction Mining (VTTC & CPA) ===================")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    intermediate_dir = os.path.join(base_dir, "ProjectPrayagTopDownDataset", "intermediate_files")
    output_dir = os.path.join(base_dir, "ProjectPrayagTopDownDataset")
    
    world_trajectories_path = os.path.join(intermediate_dir, "unified_world_trajectories.json")
    
    if not os.path.exists(world_trajectories_path):
        print(f"Error: Unified world trajectories not found at {world_trajectories_path}. Please run telemetry_aligner.py first.")
        return False
        
    print(f"Loading unified world trajectories: {world_trajectories_path}")
    with open(world_trajectories_path, 'r') as f:
        traj_data = json.load(f)
        
    # Organize data by scene -> frame -> track_id
    print("Grouping trajectories by scene and frame...")
    scene_frame_data = defaultdict(lambda: defaultdict(dict))
    # Keep track of individual trajectories for smoothing and velocity calculation
    # key: (scene_id, track_id) -> list of (frame_id, x, y, class)
    agent_paths = defaultdict(list)
    
    for pt in traj_data:
        scene = pt["scene_id"]
        frame = pt["frame_id"]
        tid = pt["track_id"]
        wx = pt["world_center_x"]
        wy = pt["world_center_y"]
        cid = pt["class"]
        
        scene_frame_data[scene][frame][tid] = pt
        agent_paths[(scene, tid)].append((frame, wx, wy, cid))
        
    # Step 1: Smooth trajectories and calculate velocity vectors
    print("Calculating velocity vectors...")
    agent_velocities = {} # key: (scene_id, track_id) -> dict of frame_id -> (vx, vy)
    
    for key, path in agent_paths.items():
        scene, tid = key
        # Sort by frame_id
        path.sort(key=lambda x: x[0])
        
        # Calculate velocity using sliding window central difference
        # FPS is 30Hz, so dt = 1/30 seconds
        dt = 1.0 / 30.0
        
        vel_dict = {}
        for idx in range(len(path)):
            frame_id = path[idx][0]
            
            # Central difference
            if idx > 0 and idx < len(path) - 1:
                vx = (path[idx+1][1] - path[idx-1][1]) / (2.0 * dt)
                vy = (path[idx+1][2] - path[idx-1][2]) / (2.0 * dt)
            elif idx > 0: # backward diff
                vx = (path[idx][1] - path[idx-1][1]) / dt
                vy = (path[idx][2] - path[idx-1][2]) / dt
            elif idx < len(path) - 1: # forward diff
                vx = (path[idx+1][1] - path[idx][1]) / dt
                vy = (path[idx+1][2] - path[idx][2]) / dt
            else:
                vx, vy = 0.0, 0.0
                
            vel_dict[frame_id] = (vx, vy)
            
        agent_velocities[key] = vel_dict
        
    # Step 2: Traverse every motorized vehicle (class 1: HVE, class 2: SVE) as Ego
    # Extract interactions using VTTC and CPA
    print("Mining interactions...")
    interaction_scenarios = []
    
    # Paper threshold parameters
    VTTC_FILTER_THRESHOLD = 1.53 # s (Q3 upper quartile cut-off)
    VTTC_CONVERGENCE_THRESHOLD = 0.7 # s (negotiation convergence)
    SPATIAL_RADIUS = 30.0 # meters (surrounding region of interest)
    
    interaction_id = 1
    
    for (scene_id, ego_id), ego_path in agent_paths.items():
        # Ego must be HVE (1) or SVE (2)
        ego_class = ego_path[0][3]
        if ego_class not in [1, 2]:
            continue
            
        ego_path_dict = {item[0]: (item[1], item[2]) for item in ego_path}
        
        # We will collect active interacting frames for each neighboring TP
        # neighboring TP -> list of frames where interaction is highly relevant
        tp_interactions = defaultdict(list)
        
        for frame, ego_x, ego_y, _ in ego_path:
            # Get velocity of ego
            ego_vx, ego_vy = agent_velocities[(scene_id, ego_id)].get(frame, (0.0, 0.0))
            
            # Check all surrounding agents in this frame
            frame_agents = scene_frame_data[scene_id][frame]
            for tp_id, tp_pt in frame_agents.items():
                if tp_id == ego_id:
                    continue
                    
                tp_x = tp_pt["world_center_x"]
                tp_y = tp_pt["world_center_y"]
                
                # Check spatial distance
                dist = np.hypot(tp_x - ego_x, tp_y - ego_y)
                if dist > SPATIAL_RADIUS:
                    continue
                    
                # Calculate relative vectors
                tp_vx, tp_vy = agent_velocities[(scene_id, tp_id)].get(frame, (0.0, 0.0))
                
                # Relative vectors: TP - Ego
                dx = tp_x - ego_x
                dy = tp_y - ego_y
                dvx = tp_vx - ego_vx
                dvy = tp_vy - ego_vy
                
                # Relative velocity dot product with relative position
                dot_product = dx * dvx + dy * dvy
                
                # If dot_product < 0, they are approaching each other
                if dot_product < 0:
                    dv_sq = dvx**2 + dvy**2
                    if dv_sq > 0.001:
                        # Time to closest approach
                        tau_cpa = -dot_product / dv_sq
                        
                        if tau_cpa > 0 and tau_cpa <= 3.0: # Check up to 3.0s in future
                            # Distance at CPA
                            cpa_x = dx + tau_cpa * dvx
                            cpa_y = dy + tau_cpa * dvy
                            dist_cpa = np.hypot(cpa_x, cpa_y)
                            
                            # If they get within 5 meters at closest approach, it's a valid interaction
                            if dist_cpa < 5.0:
                                vttc = tau_cpa
                                if vttc <= VTTC_FILTER_THRESHOLD:
                                    tp_interactions[tp_id].append((frame, vttc, dist_cpa))
                                    
        # Post-process: group interacting frames into scenarios
        for tp_id, frames in tp_interactions.items():
            if len(frames) < 15: # At least 0.5s of persistent interaction
                continue
                
            # Find continuous segments of frames
            frames.sort(key=lambda x: x[0])
            segments = []
            current_segment = [frames[0]]
            
            for idx in range(1, len(frames)):
                # Frame index gap <= 5 frames is allowed
                if frames[idx][0] - frames[idx-1][0] <= 5:
                    current_segment.append(frames[idx])
                else:
                    if len(current_segment) >= 15:
                        segments.append(current_segment)
                    current_segment = [frames[idx]]
            if len(current_segment) >= 15:
                segments.append(current_segment)
                
            # Create a structured scenario for each valid segment
            for seg in segments:
                start_frame = seg[0][0]
                end_frame = seg[-1][0]
                mean_vttc = np.mean([item[1] for item in seg])
                min_cpa_dist = np.min([item[2] for item in seg])
                
                # Check if interaction reaches the critical 0.7s convergence
                has_reached_critical_07s = any(abs(item[1] - 0.7) <= 0.15 for item in seg)
                
                interaction_scenarios.append({
                    "interaction_id": interaction_id,
                    "scene_id": scene_id,
                    "ego_id": ego_id,
                    "tp_id": tp_id,
                    "tp_class": agent_paths[(scene_id, tp_id)][0][3],
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "duration_seconds": (end_frame - start_frame) / 30.0,
                    "mean_vttc": float(mean_vttc),
                    "min_cpa_distance_meters": float(min_cpa_dist),
                    "reaches_critical_negotiation": has_reached_critical_07s
                })
                interaction_id += 1
                
    # Save scenario library
    output_json_path = os.path.join(output_dir, "interaction_scenarios.json")
    print(f"Saving interaction scenarios library to {output_json_path}")
    with open(output_json_path, 'w') as f:
        json.dump(interaction_scenarios, f, indent=4)
        
    print(f"Interaction mining complete. Mined {len(interaction_scenarios)} highly relevant interaction scenarios.")
    return True

if __name__ == "__main__":
    run_interaction_mining()
