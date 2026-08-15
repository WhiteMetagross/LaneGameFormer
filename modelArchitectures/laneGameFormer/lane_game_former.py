import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Union

from model.lanegcn_encoder import ActorNet, MapNet, A2M, M2M, M2A, A2A, actor_gather, graph_gather
from model.gameformer_decoder import GameFormerDecoder


class LaneGameFormer(nn.Module):
    """
    LaneGameFormer: Hybrid Trajectory Prediction Model.
    
    This model combines:
        - LaneGCN Encoder: Efficient graph-based encoding of agents and lane topology
        - GameFormer Decoder: Game-theoretic Level-k reasoning for multi-agent prediction
    
    Architecture:
        1. ActorNet: Encodes agent trajectories via 1D convolutions
        2. MapNet: Encodes lane graph via graph neural networks
        3. A2M → M2M → M2A → A2A: Bidirectional agent-map fusion
        4. GameFormerDecoder: Multi-level decoding with interaction modeling
    
    Key Features:
        - GMM output for uncertainty quantification
        - Rich trajectory encoding (velocity, heading, size)
        - Agent-specific query embeddings
        - Self-masking in interaction stages
        - World coordinate transformation
    
    Reference:
        LaneGCN: Liang et al., ECCV 2020
        GameFormer: Huang et al., ICCV 2023
    """
    
    def __init__(self, config):
        super(LaneGameFormer, self).__init__()
        self.config = config
        
        # === Encoder (LaneGCN) ===
        self.actor_net = ActorNet(config)
        self.map_net = MapNet(config)
        
        # Fusion Layers
        self.a2m = A2M(config)
        self.m2m = M2M(config)
        self.m2a = M2A(config)
        self.a2a = A2A(config)
        
        # === Projections for Decoder ===
        agent_dim = config['model']['encoder']['agent_dim']
        map_dim = config['model']['encoder']['map_dim']
        decoder_dim = config['model']['decoder']['d_model']
        
        self.agent_proj = nn.Linear(agent_dim, decoder_dim) if agent_dim != decoder_dim else nn.Identity()
        self.map_proj = nn.Linear(map_dim, decoder_dim) if map_dim != decoder_dim else nn.Identity()
        
        # === Decoder (GameFormer) ===
        self.decoder = GameFormerDecoder(config)
        
    def forward(self, data: Dict) -> Dict:
        """
        Forward pass through LaneGameFormer.
        
        Args:
            data: Dict containing:
                - 'feats': List of agent trajectories [(N_i, T, 4), ...] with (x, y, mask, priority)
                - 'ctrs': List of agent centers [(N_i, 2), ...]
                - 'graph': List of graph dicts with lane topology
                - 'rot', 'orig': Coordinate transformation matrices
                - 'vehicle_sizes': Optional (B, N, 3) vehicle dimensions
                
        Returns:
            outputs: Dict with keys 'level_0', 'level_1', ..., each containing:
                - 'traj': (B, N, K, T, 2) mean trajectories
                - 'gmm_params': (B, N, K, T, 4) GMM parameters (μx, μy, log_σx, log_σy)
                - 'scores': (B, N, K) mode scores
        """
        # ==================== 1. Encoder ====================
        # Construct actor features
        actors, actor_idcs = actor_gather(data["feats"])
        actor_ctrs = data["ctrs"]
        actors = self.actor_net(actors)  # (N_total, D)

        # Construct map features
        graph = graph_gather(data["graph"])
        nodes, node_idcs, node_ctrs = self.map_net(graph)  # (M_total, D)

        # Actor-Map Fusion Cycle 
        # Only run fusion if we have map nodes (otherwise keep actors as-is) and map less ablation is not active
        if nodes.shape[0] > 0 and not self.config.get('ablation', {}).get('bypass_map', False):
            nodes = self.a2m(nodes, graph, actors, actor_idcs, actor_ctrs)
            nodes = self.m2m(nodes, graph)
            actors = self.m2a(actors, actor_idcs, actor_ctrs, nodes, node_idcs, node_ctrs)
        actors = self.a2a(actors, actor_idcs, actor_ctrs)
        
        # ==================== 2. Prepare for Decoder ====================
        batch_size = len(actor_idcs)
        max_actors = max([len(x) for x in actor_idcs]) if actor_idcs else 1
        max_nodes = max([len(x) for x in node_idcs]) if node_idcs else 1
        feature_dim = actors.shape[1]
        device = actors.device
        
        # Pad Actors: (B, N_max, D)
        padded_actors = torch.zeros(batch_size, max_actors, feature_dim, device=device)
        actor_mask = torch.zeros(batch_size, max_actors, dtype=torch.bool, device=device)
        
        for i, idcs in enumerate(actor_idcs):
            num = len(idcs)
            padded_actors[i, :num] = actors[idcs]
            actor_mask[i, :num] = True
            
        # Pad Nodes (Context): (B, S_max, D)
        padded_nodes = torch.zeros(batch_size, max_nodes, feature_dim, device=device)
        node_mask = torch.ones(batch_size, max_nodes, dtype=torch.bool, device=device)  # True = ignore
        
        # Only fill in nodes if we have some and map less ablation is not active
        if nodes.shape[0] > 0 and not self.config.get('ablation', {}).get('bypass_map', False):
            for i, idcs in enumerate(node_idcs):
                num = len(idcs)
                if num > 0:
                    padded_nodes[i, :num] = nodes[idcs]
                    node_mask[i, :num] = False  # False = attend
            
        # Project to Decoder Dimension
        padded_actors = self.agent_proj(padded_actors)
        padded_nodes = self.map_proj(padded_nodes)
        
        # Extract current positions for trajectory offset
        # actor_ctrs is a list of (N_i, 2) tensors
        current_states = torch.zeros(batch_size, max_actors, 2, device=device)
        for i, ctrs in enumerate(actor_ctrs):
            if isinstance(ctrs, torch.Tensor):
                num = min(len(ctrs), max_actors)
                current_states[i, :num] = ctrs[:num].to(device)
        
        # ===== Constant-Velocity Prior =====
        # Compute velocity from observation history and extrapolate into future.
        # This gives a per-timestep trajectory baseline for each agent —
        # the model then only needs to predict residuals from CV.
        # This also fixes the zero-priors bug where Level-0 had no position offset.
        T_future = self.config['model']['decoder']['future_steps']
        K_modes = self.config['model']['decoder']['num_modes']
        dt = 1.0 / 10.0  # 10 Hz dataset
        
        cv_prior = torch.zeros(batch_size, max_actors, T_future, 2, device=device)
        for i, feats_i in enumerate(data['feats']):
            # feats_i: (N_i, T_obs, 4) with (x, y, mask, priority)
            if isinstance(feats_i, torch.Tensor):
                feats_i = feats_i.to(device)
                n_agents = min(feats_i.shape[0], max_actors)
                for a in range(n_agents):
                    obs = feats_i[a]  # (T_obs, 4)
                    mask = obs[:, 2]  # (T_obs,) — 1.0 if valid
                    valid_idx = torch.where(mask > 0.5)[0]
                    if len(valid_idx) >= 2:
                        last = valid_idx[-1]
                        prev = valid_idx[-2]
                        velocity = (obs[last, :2] - obs[prev, :2]) / ((last - prev).float() * dt)
                        last_pos = obs[last, :2]
                    elif len(valid_idx) == 1:
                        velocity = torch.zeros(2, device=device)
                        last_pos = obs[valid_idx[0], :2]
                    else:
                        velocity = torch.zeros(2, device=device)
                        last_pos = current_states[i, a]
                    # Extrapolate: cv[t] = last_pos + velocity * (t+1) * dt
                    t_steps = torch.arange(1, T_future + 1, device=device, dtype=torch.float32)
                    cv_prior[i, a] = last_pos.unsqueeze(0) + velocity.unsqueeze(0) * (t_steps * dt).unsqueeze(-1)
        
        # Expand to all modes: (B, N, K, T, 2)
        priors = cv_prior.unsqueeze(2).expand(-1, -1, K_modes, -1, -1).contiguous()
        
        # Extract vehicle sizes if available
        vehicle_sizes = data.get('vehicle_sizes', None)
        if vehicle_sizes is not None:
            if isinstance(vehicle_sizes, list):
                # Pad to (B, N_max, 3)
                padded_sizes = torch.zeros(batch_size, max_actors, 3, device=device)
                for i, sizes in enumerate(vehicle_sizes):
                    if isinstance(sizes, torch.Tensor):
                        num = min(len(sizes), max_actors)
                        padded_sizes[i, :num] = sizes[:num].to(device)
                vehicle_sizes = padded_sizes
            
        # ==================== 3. Decoder ====================
        outputs = self.decoder(
            agent_features=padded_actors,
            context=padded_nodes,
            context_mask=node_mask,
            agent_mask=actor_mask,
            current_states=current_states,
            vehicle_sizes=vehicle_sizes,
            priors=priors
        )
        
        # ==================== 4. Post-process ====================
        # Keep predictions in ego-centric space for loss/eval consistency.
                
        return outputs
    
    def _transform_to_world(self, outputs: Dict, data: Dict, device: torch.device) -> Dict:
        """
        Transform predictions from ego-centric to world coordinates.
        
        Args:
            outputs: Decoder outputs with 'traj' and 'gmm_params'
            data: Dict with 'rot' and 'orig' transformation matrices
            device: Target device
            
        Returns:
            outputs: Transformed outputs
        """
        # Stack rot and orig
        rot = data["rot"]
        if isinstance(rot, list):
            rot = torch.stack(rot).to(device)
            
        orig = data["orig"]
        if isinstance(orig, list):
            orig = torch.stack(orig).to(device)
        
        for level in outputs:
            # Transform mean trajectory
            traj = outputs[level]['traj']  # (B, N, K, T, 2)
            B, N, K, T, _ = traj.shape
            
            # Reshape for matmul: (B, N*K*T, 2)
            traj_flat = traj.view(B, -1, 2)
            
            # Rotate: (B, P, 2) @ (B, 2, 2) -> (B, P, 2)
            traj_rot = torch.bmm(traj_flat, rot)
            
            # Translate
            traj_world = traj_rot + orig.unsqueeze(1)
            
            outputs[level]['traj'] = traj_world.view(B, N, K, T, 2)
            
            # Transform GMM params if present
            if 'gmm_params' in outputs[level]:
                gmm = outputs[level]['gmm_params']  # (B, N, K, T, 4)
                
                # Only rotate the mean (first 2 dims)
                # Std (log_σ) is rotation-invariant
                mu = gmm[..., :2]  # (B, N, K, T, 2)
                log_std = gmm[..., 2:4]  # (B, N, K, T, 2)
                
                mu_flat = mu.view(B, -1, 2)
                mu_rot = torch.bmm(mu_flat, rot)
                mu_world = mu_rot + orig.unsqueeze(1)
                mu_world = mu_world.view(B, N, K, T, 2)
                
                outputs[level]['gmm_params'] = torch.cat([mu_world, log_std], dim=-1)
                
        return outputs
    
    def get_uncertainty(self, outputs: Dict, level: str = None) -> Dict:
        """
        Extract uncertainty estimates from GMM parameters.
        
        Args:
            outputs: Model outputs with 'gmm_params'
            level: Specific level to extract (default: last level)
            
        Returns:
            uncertainty: Dict with 'std' and 'confidence'
        """
        if level is None:
            # Use last level
            level = f'level_{self.decoder.num_levels}'
        
        if level not in outputs or 'gmm_params' not in outputs[level]:
            return None
            
        gmm = outputs[level]['gmm_params']  # (B, N, K, T, 4)
        scores = outputs[level]['scores']  # (B, N, K)
        
        # Extract std from log_std
        log_std = gmm[..., 2:4]
        std = torch.exp(log_std)  # (B, N, K, T, 2)
        
        # Mode probabilities
        probs = torch.softmax(scores, dim=-1)  # (B, N, K)
        
        return {
            'std': std,
            'mode_probs': probs,
            'gmm_params': gmm
        }
