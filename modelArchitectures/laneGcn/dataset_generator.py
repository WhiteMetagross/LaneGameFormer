import config

def generate_samples(tracking_data, scene_id):
    print(f"--- Stage 3: Generating Dataset Samples for Scene: {scene_id} ---")
    samples = []
    h_frames = int(config.HISTORY_SECS * config.FPS)
    f_frames = int(config.PREDICTION_SECS * config.FPS)
    agent_ids = list(tracking_data.keys())

    for agent_id in agent_ids:
        track = tracking_data[agent_id]
        frames = sorted(track.keys())
        if len(frames) < h_frames + f_frames:
            continue

        for i in range(h_frames, len(frames) - f_frames):
            curr_f = frames[i]
            hist_start_f = frames[i - h_frames]
            
            if curr_f - hist_start_f > h_frames * 1.5:
                continue
            
            agent_hist = [track[f]['center'] for f in frames[i-h_frames:i]]
            agent_fut = [track[f]['center'] for f in frames[i:i+f_frames]]
            
            if len(agent_hist) != h_frames or len(agent_fut) != f_frames:
                continue
            
            curr_pos = track[curr_f]['center']
            neighbors = []
            for neighbor_id in agent_ids:
                if agent_id == neighbor_id:
                    continue
                
                n_track = tracking_data[neighbor_id]
                if curr_f in n_track:
                    n_pos = n_track[curr_f]['center']
                    dist = ((curr_pos[0] - n_pos[0])**2 + (curr_pos[1] - n_pos[1])**2)**0.5
                    
                    if dist < config.SCENE_RADIUS:
                        n_hist_frames = [f for f in frames[i-h_frames:i] if f in n_track]
                        if len(n_hist_frames) > h_frames / 2:
                            neighbors.append({'id': neighbor_id, 'history': [n_track[f]['center'] for f in n_hist_frames]})
            
            samples.append({
                'scene_id': scene_id,
                'agent_id': agent_id,
                'start_frame': curr_f,
                'agent_history': agent_hist,
                'agent_future_gt': agent_fut,
                'neighbors': sorted(neighbors, key=lambda n: n['id'])[:config.MAX_NEIGHBORS]
            })
    
    print(f"  [{scene_id}] Generated {len(samples)} training samples.")
    return samples

