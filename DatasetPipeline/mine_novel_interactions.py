import os
import sys
import json
import numpy as np
from collections import defaultdict
import argparse

# OBB math helper to compute minimum distance between two oriented bounding boxes
def min_distance_between_obbs(obb1, obb2):
    # Standard OBB-to-OBB distance is approximated by the minimum distance between their vertices and edges.
    pts1 = np.array(obb1, dtype=np.float32)
    pts2 = np.array(obb2, dtype=np.float32)
    
    min_dist = float('inf')
    # Vertex-to-Vertex distances
    for p1 in pts1:
        for p2 in pts2:
            dist = np.linalg.norm(p1 - p2)
            if dist < min_dist:
                min_dist = dist
                
    # Also evaluate midpoint distances along edges for precision
    for i in range(4):
        mid1 = (pts1[i] + pts1[(i+1)%4]) / 2.0
        for j in range(4):
            mid2 = (pts2[j] + pts2[(j+1)%4]) / 2.0
            dist = np.linalg.norm(mid1 - mid2)
            if dist < min_dist:
                min_dist = dist
                
    return min_dist

# Quadratic formula solver for active acceleration trajectories
# dx(tau) = dx + tau * dvx + 0.5 * tau^2 * dax
# Finds the CPA tau analytically by solving:
# da^2 * tau^3 + 3*(dv.da)*tau^2 + 2*(||dv||^2 + dp.da)*tau + 2*(dp.dv) = 0
def solve_quadratic_cpa(dx, dy, dvx, dvy, dax, day):
    # Coefficients of the cubic equation: A*t^3 + B*t^2 + C*t + D = 0
    A = dax**2 + day**2
    B = 3.0 * (dvx * dax + dvy * day)
    C = 2.0 * (dvx**2 + dvy**2 + dx * dax + dy * day)
    D = 2.0 * (dx * dvx + dy * dvy)
    
    if abs(A) < 1e-4:
        # Fallback to linear constant-velocity if acceleration is negligible
        dv_sq = dvx**2 + dvy**2
        if dv_sq > 1e-4:
            return - (dx * dvx + dy * dvy) / dv_sq
        return 0.0
        
    # Solve cubic equation using numpy roots
    coeffs = [A, B, C, D]
    roots = np.roots(coeffs)
    
    # Filter real roots
    real_roots = roots[np.isreal(roots)].real
    
    # We are interested in positive times in the near future (e.g. up to 3.0s)
    valid_roots = [r for r in real_roots if r > 0 and r <= 3.0]
    if valid_roots:
        return min(valid_roots) # First point of approach
        
    # Standard constant velocity fallback
    dv_sq = dvx**2 + dvy**2
    if dv_sq > 1e-4:
        return max(0.0, - (dx * dvx + dy * dvy) / dv_sq)
    return 0.0

def run_novel_interaction_mining():
    print("\n=================== Starting NOVEL Interaction Mining (OB-VTC, A-VTC, SCI) ===================")
    
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
        
    print("Calculating velocity and acceleration vectors via sliding central differences...")
    # key: (scene_id, track_id) -> dict of frame_id -> {"vel": (vx, vy), "accel": (ax, ay)}
    agent_kinematics = {}
    dt = 1.0 / 30.0
    
    for key, path in agent_paths.items():
        # Sort by frame_id
        path.sort(key=lambda x: x["frame_id"])
        N = len(path)
        
        kin_dict = {}
        # 1. First pass: compute velocities
        vels = []
        for idx in range(N):
            if idx > 0 and idx < N - 1:
                vx = (path[idx+1]["world_center_x"] - path[idx-1]["world_center_x"]) / (2.0 * dt)
                vy = (path[idx+1]["world_center_y"] - path[idx-1]["world_center_y"]) / (2.0 * dt)
            elif idx > 0:
                vx = (path[idx]["world_center_x"] - path[idx-1]["world_center_x"]) / dt
                vy = (path[idx]["world_center_y"] - path[idx-1]["world_center_y"]) / dt
            elif idx < N - 1:
                vx = (path[idx+1]["world_center_x"] - path[idx]["world_center_x"]) / dt
                vy = (path[idx+1]["world_center_y"] - path[idx]["world_center_y"]) / dt
            else:
                vx, vy = 0.0, 0.0
            vels.append((vx, vy))
            
        # 2. Second pass: compute accelerations
        for idx in range(N):
            if idx > 0 and idx < N - 1:
                ax = (vels[idx+1][0] - vels[idx-1][0]) / (2.0 * dt)
                ay = (vels[idx+1][1] - vels[idx-1][1]) / (2.0 * dt)
            elif idx > 0:
                ax = (vels[idx][0] - vels[idx-1][0]) / dt
                ay = (vels[idx][1] - vels[idx-1][1]) / dt
            elif idx < N - 1:
                ax = (vels[idx+1][0] - vels[idx][0]) / dt
                ay = (vels[idx+1][1] - vels[idx][1]) / dt
            else:
                ax, ay = 0.0, 0.0
                
            frame_id = path[idx]["frame_id"]
            kin_dict[frame_id] = {
                "vel": vels[idx],
                "accel": (ax, ay)
            }
            
        agent_kinematics[key] = kin_dict
        
    print("Mining advanced interactions & calculating Swarm Complexity Indexes (SCI)...")
    novel_scenarios = []
    
    VTTC_FILTER_THRESHOLD = 1.53
    SPATIAL_RADIUS = 30.0 # meters
    interaction_id = 1
    
    # Class-dependent vulnerability weights
    # 0: HPE (pedestrians/cows/vulnerables) -> 2.5
    # 1: LVE (buses/trucks/large) -> 1.0
    # 2: SVE (three-wheelers/motorcycles) -> 1.8
    vulner_weights = {0: 2.5, 1: 1.0, 2: 1.8}
    
    for (scene_id, ego_id), ego_path in agent_paths.items():
        ego_class = ego_path[0]["class"]
        # Ego must be LVE (1) or SVE (2)
        if ego_class not in [1, 2]:
            continue
            
        # track interactions with TPs
        tp_interactions = defaultdict(list)
        
        # Sort path
        ego_path.sort(key=lambda x: x["frame_id"])
        
        for ego_pt in ego_path:
            frame = ego_pt["frame_id"]
            ego_x = ego_pt["world_center_x"]
            ego_y = ego_pt["world_center_y"]
            ego_obb = ego_pt["world_obb"]
            
            # Kinematics
            ego_kin = agent_kinematics[(scene_id, ego_id)].get(frame, {"vel": (0.0, 0.0), "accel": (0.0, 0.0)})
            ego_vx, ego_vy = ego_kin["vel"]
            ego_ax, ego_ay = ego_kin["accel"]
            
            # 1. Swarm Complexity Index calculation
            local_sci = 0.0
            
            frame_agents = scene_frame_data[scene_id][frame]
            for tp_id, tp_pt in frame_agents.items():
                if tp_id == ego_id:
                    continue
                    
                tp_x = tp_pt["world_center_x"]
                tp_y = tp_pt["world_center_y"]
                tp_obb = tp_pt["world_obb"]
                tp_class = tp_pt["class"]
                
                dist = np.hypot(tp_x - ego_x, tp_y - ego_y)
                if dist > SPATIAL_RADIUS:
                    continue
                    
                tp_kin = agent_kinematics[(scene_id, tp_id)].get(frame, {"vel": (0.0, 0.0), "accel": (0.0, 0.0)})
                tp_vx, tp_vy = tp_kin["vel"]
                tp_ax, tp_ay = tp_kin["accel"]
                
                # Relative parameters
                dx = tp_x - ego_x
                dy = tp_y - ego_y
                dvx = tp_vx - ego_vx
                dvy = tp_vy - ego_vy
                dax = tp_ax - ego_ax
                day = tp_ay - ego_ay
                
                # Proximity check
                dot = dx * dvx + dy * dvy
                if dot < 0: # Approaching
                    # Novelty 2: Acceleration-Aware Quadratic VTC (A-VTC)
                    a_vtc = solve_quadratic_cpa(dx, dy, dvx, dvy, dax, day)
                    
                    if a_vtc > 0 and a_vtc <= VTTC_FILTER_THRESHOLD:
                        # Novelty 1: Oriented Boundary-to-Boundary VTC (OB-VTC)
                        d_edge = min_distance_between_obbs(ego_obb, tp_obb)
                        
                        # Project range rate from relative velocity vector onto OBB edge normal
                        rel_speed = np.hypot(dvx, dvy)
                        if rel_speed > 0.01:
                            range_rate = dot / dist # rate of change of distance
                            ob_vtc = - d_edge / range_rate if range_rate < 0 else a_vtc
                        else:
                            ob_vtc = a_vtc
                            
                        # Ensure reasonable bounds
                        ob_vtc = max(0.05, min(ob_vtc, 3.0))
                        
                        # Update Swarm Complexity node weight
                        node_weight = vulner_weights.get(tp_class, 1.0)
                        local_sci += node_weight / ob_vtc
                        
                        # Store interaction statistics
                        tp_interactions[tp_id].append({
                            "frame": frame,
                            "a_vtc": float(a_vtc),
                            "ob_vtc": float(ob_vtc),
                            "edge_dist": float(d_edge),
                            "sci": float(local_sci)
                        })
                        
        # 2. Time-series segmenting and persistence verification
        for tp_id, records in tp_interactions.items():
            if len(records) < 15:
                continue
                
            records.sort(key=lambda x: x["frame"])
            segments = []
            current_segment = [records[0]]
            
            for idx in range(1, len(records)):
                if records[idx]["frame"] - records[idx-1]["frame"] <= 5:
                    current_segment.append(records[idx])
                else:
                    if len(current_segment) >= 15:
                        segments.append(current_segment)
                    current_segment = [records[idx]]
            if len(current_segment) >= 15:
                segments.append(current_segment)
                
            for seg in segments:
                start_frame = seg[0]["frame"]
                end_frame = seg[-1]["frame"]
                
                mean_a_vtc = np.mean([item["a_vtc"] for item in seg])
                mean_ob_vtc = np.mean([item["ob_vtc"] for item in seg])
                min_edge_dist = np.min([item["edge_dist"] for item in seg])
                max_swarm_sci = np.max([item["sci"] for item in seg])
                
                novel_scenarios.append({
                    "interaction_id": interaction_id,
                    "scene_id": scene_id,
                    "ego_id": ego_id,
                    "tp_id": tp_id,
                    "tp_class": agent_paths[(scene_id, tp_id)][0]["class"],
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "duration_seconds": (end_frame - start_frame) / 30.0,
                    
                    # Novel Safety Measures
                    "mean_acceleration_aware_vtc": float(mean_a_vtc),
                    "mean_oriented_boundary_vtc": float(mean_ob_vtc),
                    "min_boundary_distance_meters": float(min_edge_dist),
                    "max_swarm_complexity_index": float(max_swarm_sci)
                })
                interaction_id += 1
                
    output_json_path = os.path.join(output_dir, "novel_interaction_scenarios.json")
    print(f"Saving NOVEL interaction scenarios library to {output_json_path}")
    with open(output_json_path, 'w') as f:
        json.dump(novel_scenarios, f, indent=4)
        
    print(f"Novel interaction mining complete. Mined {len(novel_scenarios)} advanced scenarios using OB-VTC, A-VTC, and SCI metrics.")
    return True

if __name__ == "__main__":
    run_novel_interaction_mining()
