import config
import utils

def process_lanes_into_graph(polylines, scene_id):
    print(f"--- Stage 2: Processing Lanes into Graph for Scene: {scene_id} ---")
    nodes, node_counter, lane_to_nodes = {}, 0, {}
    
    for lane_idx, polyline in enumerate(polylines):
        if len(polyline) < 2: continue
        lane_nodes, dist = [], 0.0
        for i in range(len(polyline) - 1):
            s, e = polyline[i], polyline[i+1]
            seg_len = utils.calculate_distance(s, e)
            if seg_len == 0: continue
            vec = utils.get_normalized_direction_vector(s, e)
            while dist <= seg_len:
                p = [s[0] + dist*vec[0], s[1] + dist*vec[1]]
                node_id = f"{scene_id}_{node_counter}"
                nodes[node_id] = {'id': node_id, 'coords': p, 'lane_id': lane_idx, 'scene_id': scene_id, 'successors': [], 'predecessors': [], 'left_neighbors': [], 'right_neighbors': []}
                lane_nodes.append(node_id)
                node_counter += 1
                dist += config.LANE_DISCRETIZATION_DISTANCE
            dist -= seg_len
        lane_to_nodes[lane_idx] = lane_nodes
    
    for node_ids in lane_to_nodes.values():
        for i, node_id in enumerate(node_ids):
            if i > 0: nodes[node_id]['predecessors'].append(node_ids[i-1])
            if i < len(node_ids) - 1: nodes[node_id]['successors'].append(node_ids[i+1])
    
    node_list = list(nodes.values())
    for i, n_a in enumerate(node_list):
        for n_b in node_list[i+1:]:
            if n_a['lane_id'] == n_b['lane_id']: continue
            if utils.calculate_distance(n_a['coords'], n_b['coords']) < config.NEIGHBOR_LANE_SEARCH_RADIUS:
                vec_ab = (n_b['coords'][0]-n_a['coords'][0], n_b['coords'][1]-n_a['coords'][1])
                dir_a = utils.get_normalized_direction_vector(n_a['coords'], nodes[n_a['successors'][0]]['coords']) if n_a['successors'] else (0,0)
                cross_z = dir_a[0]*vec_ab[1] - dir_a[1]*vec_ab[0]
                if cross_z > 0: n_a['left_neighbors'].append(n_b['id']); n_b['right_neighbors'].append(n_a['id'])
                else: n_a['right_neighbors'].append(n_b['id']); n_b['left_neighbors'].append(n_a['id'])
    
    print(f"  [{scene_id}] Processed into {len(nodes)} graph nodes.")
    return nodes

