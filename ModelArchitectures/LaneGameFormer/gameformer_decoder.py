"""
GameFormer Decoder for LaneGameFormer.

This module implements the full GameFormer decoder with Level-k game-theoretic
reasoning, extending the ICCV'23 GameFormer with three novel contributions:

Key Components:
    1. GMMPredictor: Outputs Gaussian Mixture Model parameters (μ, log_σ) for
       uncertainty quantification instead of deterministic point predictions.
    
    2. FutureEncoder: Rich state processing with Hierarchical Temporal Fusion
       (HTF) — dual-scale encoding capturing both short-term dynamics and
       long-term intent through a gated fusion mechanism.
    
    3. InitialDecoder: Agent-specific query embeddings for Level-0 predictions,
       enhanced with Lane-Conditioned Mode Anchoring (LCMA) — mode queries
       derived from lane graph topology for semantically meaningful multimodality.
    
    4. InteractionDecoder: Self-masking during interaction with Social Potential
       Attention Bias (SPAB) — physics-informed attention using multi-band
       repulsive potential fields from social force theory.

Novel Contributions:
    1. SPAB (Social Potential Attention Bias): Injects pairwise multi-band
       social potential as a learnable additive attention bias into the
       interaction self-attention. Encodes implicit safety zones and traffic
       conventions at inference time (not just as a training loss).
    
    2. LCMA (Lane-Conditioned Mode Anchoring): Replaces generic learned modal
       embeddings with agent-specific lane-derived mode anchors selected from
       the LaneGCN-encoded context. Each mode naturally corresponds to a
       different feasible lane/path, producing semantically diverse predictions.
    
    3. HTF (Hierarchical Temporal Fusion): Replaces simple max-pooling in
       trajectory encoding with a dual-scale gated fusion of:
       - Short-term recent dynamics (last 1/3 of timesteps, mean-pooled)
       - Long-term full-sequence intent (max-pooled over all timesteps)
       A learned gate adaptively balances both scales per sample.

Mathematical Formulation:
    Level-k Reasoning:
        - Level 0: Independent predictions based on map context.
        - Level k: Refine predictions considering other agents' Level-(k-1) intents.
    
    GMM Output:
        P(y|x) = Σᵢ πᵢ · N(y | μᵢ, Σᵢ)
        
        Where πᵢ are mode probabilities (softmax of scores),
        and each mode outputs (μx, μy, log_σx, log_σy).

    SPAB: For agents i, j at distance d:
        φ(d) = max_b { D_b · exp(-d / (0.3·R_b)) }   (multi-band potential)
        bias_{h}(i,j) = w_h · φ(d_{ij}) + b_h        (per-head learned projection)
        Attn(Q,K,V) = softmax(QKᵀ/√d + bias) · V     (bias-augmented attention)

Reference:
    Huang, Z., et al. "GameFormer: Game-theoretic Modeling and Learning of
    Transformer-based Interactive Prediction and Planning for Autonomous Driving."
    ICCV 2023.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Novel Contribution 1: Social Potential Attention Bias (SPAB)
# =============================================================================

class SocialPotentialAttentionBias(nn.Module):
    """
    Physics-informed attention bias from multi-band social potential fields.
    
    Computes pairwise repulsive potentials between agents using concentric
    distance bands (critical/high/medium/awareness) from social force theory,
    then projects to per-head attention biases via a learned linear layer.
    
    This allows the interaction decoder to be aware of safety zones and
    spatial proximity at the attention level, not just through loss penalties.
    """
    def __init__(self, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        # Fixed multi-band potential radii (from social force theory, in BEV pixels)
        self.bands = [
            (30.0, 1.0),    # Critical zone
            (60.0, 0.7),    # High risk
            (100.0, 0.4),   # Medium risk
            (150.0, 0.15),  # Awareness zone
        ]
        # Per-head learnable projection: maps scalar potential → per-head bias
        self.head_proj = nn.Linear(1, num_heads, bias=True)
        # Initialize near-zero so SPAB grows gradually during training
        nn.init.normal_(self.head_proj.weight, std=0.01)
        nn.init.zeros_(self.head_proj.bias)

    def forward(self, positions, N_target):
        """
        Compute social potential attention bias for MultiheadAttention.
        
        Args:
            positions: (B, N, 2) agent positions in BEV pixel coordinates
            N_target: int, number of target agents (for expanding the bias)
            
        Returns:
            attn_bias: (B * N_target * num_heads, N, N) additive attention bias
                       compatible with nn.MultiheadAttention's attn_mask parameter
        """
        B, N, _ = positions.shape
        device = positions.device

        # Pairwise L2 distances: (B, N, N)
        diff = positions.unsqueeze(2) - positions.unsqueeze(1)
        dist = torch.norm(diff, dim=-1).clamp(min=1e-6)

        # Multi-band potential: φ(d) = max_b { D_b · exp(-d / (0.3·R_b)) }
        potential = torch.zeros(B, N, N, device=device)
        for R, D in self.bands:
            potential = torch.max(potential, D * torch.exp(-dist / (R * 0.3)))

        # Per-head projection: (B, N, N) → (B, N, N, H) → (B, H, N, N)
        bias = self.head_proj(potential.unsqueeze(-1)).permute(0, 3, 1, 2)

        # Expand for each target agent (bias is the same for all targets)
        # (B, H, N, N) → (B, N_target, H, N, N) → (B*N_target*H, N, N)
        bias = bias.unsqueeze(1).expand(B, N_target, self.num_heads, N, N)
        return bias.reshape(B * N_target * self.num_heads, N, N)


# =============================================================================
# Novel Contribution 2: Lane-Conditioned Mode Anchoring (LCMA)
# =============================================================================

class LaneConditionedModeAnchoring(nn.Module):
    """
    Generate mode-specific query embeddings from lane graph topology.
    
    Instead of using K generic learned modal embeddings, this module:
    1. Scores each lane node's relevance to each agent
    2. Selects the top-K most relevant lanes as mode anchors
    3. Projects them into mode-specific embeddings
    
    This produces semantically meaningful modes: each mode naturally
    corresponds to a different feasible road path, eliminating arbitrary
    mode assignments and directly combating mode collapse.
    """
    def __init__(self, hidden_dim, num_modes):
        super().__init__()
        self.num_modes = num_modes
        self.hidden_dim = hidden_dim

        # Agent-lane relevance scoring
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )
        # Project selected lane features into mode-specific space
        self.mode_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Fallback learned embeddings (when fewer lanes than modes)
        self.fallback = nn.Embedding(num_modes, hidden_dim)

    def forward(self, agent_features, context, context_mask=None):
        """
        Args:
            agent_features: (BN, D) encoded agent features (batch*agents flattened)
            context: (BN, S, D) lane/context features
            context_mask: (BN, S) True = padding position to ignore
            
        Returns:
            mode_queries: (BN, K, D) mode-specific query embeddings
        """
        BN, D = agent_features.shape
        S = context.shape[1]
        K = self.num_modes
        device = agent_features.device

        # If no context available, use fallback
        if S == 0 or (context_mask is not None and context_mask.all()):
            return self.fallback.weight.unsqueeze(0).expand(BN, K, D)

        # Score each lane for each agent: (BN, S)
        agent_exp = agent_features.unsqueeze(1).expand(BN, S, D)
        combined = torch.cat([agent_exp, context], dim=-1)  # (BN, S, 2D)
        scores = self.scorer(combined).squeeze(-1)  # (BN, S)

        if context_mask is not None:
            scores = scores.masked_fill(context_mask, float('-inf'))

        # Determine how many valid lanes are available
        if context_mask is not None:
            valid_per_sample = (~context_mask).sum(dim=1)  # (BN,)
            min_valid = valid_per_sample.min().item()
        else:
            min_valid = S
        actual_k = min(K, max(1, int(min_valid)))

        if actual_k < K:
            # Fewer lanes than modes: use available lanes + fallback embeddings
            topk_scores, topk_idx = torch.topk(scores, actual_k, dim=-1)
            topk_idx_exp = topk_idx.unsqueeze(-1).expand(BN, actual_k, D)
            selected = torch.gather(context, 1, topk_idx_exp)
            selected = self.mode_proj(selected)  # (BN, actual_k, D)
            fallback = self.fallback.weight[actual_k:K].unsqueeze(0).expand(BN, K - actual_k, D)
            mode_queries = torch.cat([selected, fallback], dim=1)
        else:
            topk_scores, topk_idx = torch.topk(scores, K, dim=-1)
            topk_idx_exp = topk_idx.unsqueeze(-1).expand(BN, K, D)
            selected = torch.gather(context, 1, topk_idx_exp)
            mode_queries = self.mode_proj(selected)

        return mode_queries


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal sequences.
    """
    def __init__(self, d_model=256, max_len=100, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Args:
            x: (B, T, D) input tensor
        Returns:
            (B, T, D) with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class GMMPredictor(nn.Module):
    """
    Gaussian Mixture Model Predictor.
    
    Outputs GMM parameters for each mode:
        - μx, μy: Mean position
        - log_σx, log_σy: Log standard deviations (for numerical stability)
    
    This enables uncertainty quantification via NLL loss during training
    and probabilistic sampling during inference.
    """
    def __init__(self, hidden_dim=256, future_len=30):
        super(GMMPredictor, self).__init__()
        self.future_len = future_len
        
        # GMM parameters: (μx, μy, log_σx, log_σy) per timestep
        self.gaussian_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, future_len * 4)  # 4 params per timestep
        )
        
        # Small weights → small initial mu deltas (model predicts offsets from current pos)
        nn.init.normal_(self.gaussian_head[-1].weight, mean=0, std=0.01)
        nn.init.zeros_(self.gaussian_head[-1].bias)

        # CRITICAL: initialise log_sigma to log(50px) ≈ 3.91 so that the initial
        # predicted uncertainty matches the typical BEV pixel-scale error (~50 px).
        # Without this, sigma=1px at init → NLL = (50px)²/2 = 1250 per step →
        # total loss ≈ 150 000 per agent → NaN gradients → all training batches skipped.
        with torch.no_grad():
            bias = self.gaussian_head[-1].bias   # (future_len * 4,)
            bias.view(-1)[2::4].fill_(math.log(50.0))   # log_sig_x per timestep
            bias.view(-1)[3::4].fill_(math.log(50.0))   # log_sig_y per timestep
        
        # Mode score predictor
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, K, D) mode embeddings
            
        Returns:
            gmm_params: (B, K, T, 4) - (μx, μy, log_σx, log_σy)
            scores: (B, K) - mode scores (logits)
        """
        B, K, D = x.shape
        
        # GMM parameters
        gmm_params = self.gaussian_head(x)  # (B, K, T*4)
        gmm_params = gmm_params.view(B, K, self.future_len, 4)
        
        # Mode scores
        scores = self.score_head(x).squeeze(-1)  # (B, K)
        
        return gmm_params, scores
    
    def get_trajectory(self, gmm_params):
        """
        Extract mean trajectory from GMM parameters.
        
        Args:
            gmm_params: (B, K, T, 4)
            
        Returns:
            traj: (B, K, T, 2) - mean positions
        """
        return gmm_params[..., :2]
    
    def get_uncertainty(self, gmm_params):
        """
        Extract uncertainty (std) from GMM parameters.
        
        Args:
            gmm_params: (B, K, T, 4)
            
        Returns:
            std: (B, K, T, 2) - standard deviations
        """
        log_std = gmm_params[..., 2:4]
        return torch.exp(log_std)


class MultiModeGMMPredictor(nn.Module):
    """
    Per-mode GMM Predictor — each mode has its own independent prediction MLP.
    
    This is the key architectural fix for mode collapse: when all modes share
    the same predictor weights, identical (or near-identical) mode embeddings
    produce identical outputs. Separate heads ensure each mode develops its
    own specialised prediction function, guaranteeing output diversity even
    when embeddings are similar.
    
    Each mode head is initialised with different random weights via small
    perturbations, seeding different prediction behaviours from epoch 1.
    """
    def __init__(self, hidden_dim=256, future_len=30, num_modes=6):
        super().__init__()
        self.future_len = future_len
        self.num_modes = num_modes
        
        # Separate gaussian head per mode
        self.mode_heads = nn.ModuleList()
        for k in range(num_modes):
            head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim * 2, future_len * 4)
            )
            # Small init for mu (near-zero offsets from current pos)
            nn.init.normal_(head[-1].weight, mean=0, std=0.01)
            nn.init.zeros_(head[-1].bias)
            # Initialise log_sigma to log(50px) for stable NLL
            with torch.no_grad():
                bias = head[-1].bias
                bias.view(-1)[2::4].fill_(math.log(50.0))
                bias.view(-1)[3::4].fill_(math.log(50.0))
            # Per-mode perturbation: shift initial mu bias slightly differently
            # so each mode starts with a unique spatial tendency
            with torch.no_grad():
                angle = 2 * math.pi * k / num_modes
                bias.view(-1)[0::4].add_(0.05 * math.cos(angle))  # mu_x offset
                bias.view(-1)[1::4].add_(0.05 * math.sin(angle))  # mu_y offset
            self.mode_heads.append(head)
        
        # Shared score head (mode selection should still be a single comparator)
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, K, D) mode embeddings
        Returns:
            gmm_params: (B, K, T, 4)
            scores: (B, K)
        """
        B, K, D = x.shape
        gmm_list = []
        for k in range(K):
            out_k = self.mode_heads[k](x[:, k])  # (B, T*4)
            gmm_list.append(out_k.view(B, 1, self.future_len, 4))
        gmm_params = torch.cat(gmm_list, dim=1)  # (B, K, T, 4)
        scores = self.score_head(x).squeeze(-1)   # (B, K)
        return gmm_params, scores
    
    def get_trajectory(self, gmm_params):
        return gmm_params[..., :2]
    
    def get_uncertainty(self, gmm_params):
        return torch.exp(gmm_params[..., 2:4])


class FutureEncoder(nn.Module):
    """
    Rich Future Trajectory Encoder with Hierarchical Temporal Fusion (HTF).
    
    Processes predicted trajectories into embeddings by computing:
        - Positions (x, y)
        - Heading angle θ = atan2(Δy, Δx)
        - Velocities (vx, vy) = Δ(x,y) / Δt
        - Vehicle size (w, l, h) from current state
    
    Novel HTF: Instead of simple max-pooling over time, uses a dual-scale
    gated fusion:
        - Short-term branch: mean of last 1/3 timesteps (recent dynamics)
        - Long-term branch: max over full sequence (overall intent)
        - Learnable sigmoid gate adaptively balances both scales
    
    This captures both immediate collision-avoidance dynamics and
    longer-term route-following behavior in a single embedding.
    """
    def __init__(self, hidden_dim=256, dt=0.1):
        super(FutureEncoder, self).__init__()
        self.dt = dt  # Time step (0.1s for 10Hz)
        
        # 8-dim input: (x, y, θ, vx, vy, w, l, h)
        self.mlp = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim)
        )
        
        # HTF: Dual-scale gated temporal fusion (replaces simple max pooling)
        self.temporal_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
    
    def forward(self, trajs, current_states=None, vehicle_sizes=None):
        """
        Process trajectories into rich state representations.
        
        Args:
            trajs: (B, N, K, T, 2) predicted trajectories (positions only)
            current_states: (B, N, 2) current positions (optional)
            vehicle_sizes: (B, N, 3) vehicle dimensions (w, l, h) (optional)
            
        Returns:
            embeddings: (B, N, K, D) trajectory embeddings
        """
        B, N, K, T, _ = trajs.shape
        device = trajs.device
        
        # Compute velocities from position differences
        if current_states is not None:
            start = current_states.unsqueeze(2).unsqueeze(3).expand(-1, -1, K, 1, -1)
        else:
            start = trajs[..., :1, :]
        
        full_traj = torch.cat([start, trajs], dim=-2)
        dxy = full_traj[..., 1:, :] - full_traj[..., :-1, :]
        vel = dxy / self.dt
        theta = torch.atan2(dxy[..., 1:2], dxy[..., 0:1].clamp(min=1e-6))
        
        if vehicle_sizes is not None:
            sizes = vehicle_sizes.unsqueeze(2).unsqueeze(3).expand(-1, -1, K, T, -1)
        else:
            sizes = torch.zeros(B, N, K, T, 3, device=device)
        
        rich_states = torch.cat([trajs, theta, vel, sizes], dim=-1)
        embeddings = self.mlp(rich_states)  # (B, N, K, T, D)
        
        # === Hierarchical Temporal Fusion (HTF) ===
        D = embeddings.shape[-1]
        flat = embeddings.view(B * N * K, T, D)
        split = max(1, T // 3)
        
        # Short-term: recent dynamics (last 1/3 of timesteps)
        short_embed = flat[:, -split:].mean(dim=1)   # (BNK, D)
        # Long-term: full-sequence intent (max over all timesteps)
        long_embed = flat.max(dim=1)[0]               # (BNK, D)
        
        # Gated fusion: learned balance between scales
        gate = self.temporal_gate(torch.cat([short_embed, long_embed], dim=-1))
        fused = gate * short_embed + (1 - gate) * long_embed
        
        return fused.view(B, N, K, D)


class SelfTransformer(nn.Module):
    """
    Self-Attention Transformer Block with LayerNorm and FFN.
    
    Supports optional additive attention bias (for SPAB integration).
    """
    def __init__(self, hidden_dim=256, num_heads=8, dropout=0.1):
        super(SelfTransformer, self).__init__()
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_1 = nn.LayerNorm(hidden_dim)
        self.norm_2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None, attn_bias=None):
        """
        Args:
            x: (B, S, D) input sequence
            mask: (B, S) key padding mask (True = ignore)
            attn_bias: Optional (B*num_heads, S, S) additive attention bias
                       (e.g., from SocialPotentialAttentionBias)
            
        Returns:
            (B, S, D) output
        """
        # Convert bool key_padding_mask to float when attn_mask is float,
        # to avoid PyTorch deprecation warning about mismatched types.
        kpm = mask
        if kpm is not None and attn_bias is not None and kpm.dtype == torch.bool:
            kpm = torch.zeros_like(kpm, dtype=attn_bias.dtype)
            kpm.masked_fill_(mask, float('-inf'))
        attn_out, _ = self.self_attention(
            x, x, x, key_padding_mask=kpm, attn_mask=attn_bias
        )
        x = self.norm_1(attn_out + x)
        x = self.norm_2(self.ffn(x) + x)
        return x


class CrossTransformer(nn.Module):
    """
    Cross-Attention Transformer Block with LayerNorm and FFN.
    
    Used for attending to map context.
    """
    def __init__(self, hidden_dim=256, num_heads=8, dropout=0.1):
        super(CrossTransformer, self).__init__()
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_1 = nn.LayerNorm(hidden_dim)
        self.norm_2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (B, Q, D) query tensor
            key, value: (B, S, D) key-value tensors
            mask: (B, S) key padding mask (True = ignore)
            
        Returns:
            (B, Q, D) output
        """
        attn_out, _ = self.cross_attention(query, key, value, key_padding_mask=mask)
        x = self.norm_1(attn_out)
        x = self.norm_2(self.ffn(x) + x)
        return x


class InitialDecoder(nn.Module):
    """
    Level-0 Initial Decoder.
    
    Generates initial trajectory predictions using:
        1. Multi-modal query embeddings (K modes)
        2. Agent-specific query embeddings (per agent identity)
        3. Cross-attention to map context
        4. GMM prediction head
    
    This produces the first set of predictions WITHOUT interaction modeling.
    """
    def __init__(self, num_modes, max_agents, future_len, hidden_dim=256):
        super(InitialDecoder, self).__init__()
        self.num_modes = num_modes
        self.max_agents = max_agents
        self.future_len = future_len
        self.hidden_dim = hidden_dim
        
        # Multi-modal query embeddings (one per mode)
        self.modal_embedding = nn.Embedding(num_modes, hidden_dim)
        
        # Agent-specific query embeddings (one per agent slot)
        self.agent_embedding = nn.Embedding(max_agents, hidden_dim)
        
        # Cross-attention to context
        self.context_attn = CrossTransformer(hidden_dim)
        
        # Per-mode GMM prediction heads (prevents mode collapse)
        self.predictor = MultiModeGMMPredictor(hidden_dim, future_len, num_modes)
        
        # Register buffers for indexing
        self.register_buffer('mode_indices', torch.arange(num_modes))
        self.register_buffer('agent_indices', torch.arange(max_agents))

    def forward(self, agent_idx, agent_state, context, context_mask):
        """
        Generate Level-0 predictions for a single agent.
        
        Args:
            agent_idx: int, index of the agent
            agent_state: (B, D) encoded agent state
            context: (B, S, D) map context
            context_mask: (B, S) context padding mask (True = ignore)
            
        Returns:
            content: (B, K, D) query content embeddings
            gmm_params: (B, K, T, 4) GMM parameters
            scores: (B, K) mode scores
        """
        B = agent_state.shape[0]
        
        # Get modal embeddings: (K, D)
        modal_embed = self.modal_embedding(self.mode_indices)
        
        # Get agent-specific embedding: (D,)
        agent_embed = self.agent_embedding(
            self.agent_indices[agent_idx].expand(B)
        )  # (B, D)
        
        # Combine: modal + agent + encoded state
        # (B, K, D) = (1, K, D) + (B, 1, D) + (B, 1, D)
        query = (
            modal_embed.unsqueeze(0) + 
            agent_embed.unsqueeze(1) + 
            agent_state.unsqueeze(1)
        )
        
        # Cross-attention to context
        content = self.context_attn(query, context, context, context_mask)
        
        # Predict GMM parameters
        gmm_params, scores = self.predictor(content)
        
        return content, gmm_params, scores


class InteractionDecoder(nn.Module):
    """
    Level-k Interaction Decoder.
    
    Refines trajectory predictions by:
        1. Encoding other agents' trajectories from Level-(k-1)
        2. Self-attention for agent-agent interaction (with self-masking)
        3. Cross-attention to map context
        4. GMM prediction head
    
    Key Features:
        - Self-masking: Target agent cannot attend to its own previous prediction
        - Trajectory encoding: Uses rich 8-dim states (pos, vel, heading, size)
        - Residual refinement: Adds previous level's content embedding
    """
    def __init__(self, num_modes, future_len, hidden_dim=256, dt=0.1):
        super(InteractionDecoder, self).__init__()
        self.num_modes = num_modes
        self.future_len = future_len
        self.hidden_dim = hidden_dim
        
        # Rich future trajectory encoder
        self.future_encoder = FutureEncoder(hidden_dim, dt)
        
        # Interaction modeling (self-attention among agents)
        self.interaction_attn = SelfTransformer(hidden_dim)
        
        # Context attention
        self.context_attn = CrossTransformer(hidden_dim)
        
        # Per-mode GMM prediction heads (prevents mode collapse)
        self.predictor = MultiModeGMMPredictor(hidden_dim, future_len, num_modes)

    def forward(
        self,
        agent_idx,
        current_states,
        prev_trajs,
        prev_scores,
        prev_content,
        context,
        context_mask,
        agent_mask=None,
        vehicle_sizes=None
    ):
        """
        Generate Level-k predictions for a single agent with interaction.
        
        Args:
            agent_idx: int, index of the target agent
            current_states: (B, N, 2) current positions of all agents
            prev_trajs: (B, N, K, T, 2) trajectories from Level-(k-1)
            prev_scores: (B, N, K) mode scores from Level-(k-1)
            prev_content: (B, K, D) previous content embedding for target agent
            context: (B, S, D) map context
            context_mask: (B, S) context padding mask (True = ignore)
            agent_mask: (B, N) agent validity mask (True = valid)
            vehicle_sizes: (B, N, 3) vehicle dimensions (optional)
            
        Returns:
            content: (B, K, D) updated query content
            gmm_params: (B, K, T, 4) GMM parameters  
            scores: (B, K) mode scores
        """
        B, N, K, T, _ = prev_trajs.shape
        
        # 1. Encode all agents' trajectories from previous level
        # (B, N, K, D)
        traj_embeddings = self.future_encoder(
            prev_trajs, current_states, vehicle_sizes
        )
        
        # 2. Aggregate across modes using score-weighted averaging
        # prev_scores: (B, N, K) -> softmax -> (B, N, K, 1)
        mode_weights = F.softmax(prev_scores, dim=-1).unsqueeze(-1)
        
        # Weighted average: (B, N, D)
        agent_futures = (traj_embeddings * mode_weights).sum(dim=2)
        
        # 3. Self-attention for interaction modeling (with self-masking)
        # Create interaction mask: mask out the target agent and invalid agents
        if agent_mask is not None:
            # (B, N) -> True means IGNORE in attention
            interaction_mask = ~agent_mask  # Invert: False=valid -> True=ignore
        else:
            interaction_mask = torch.zeros(B, N, dtype=torch.bool, device=prev_trajs.device)
        
        # CRITICAL: Mask out the target agent's own future (self-masking)
        interaction_mask = interaction_mask.clone()
        interaction_mask[:, agent_idx] = True  # Ignore self
        
        # Check if ALL agents are masked (single-agent case or no other valid agents)
        # In this case, skip interaction attention and use zeros
        all_masked = interaction_mask.all(dim=1)  # (B,)
        
        if all_masked.all():
            # No other agents to interact with - use zeros for interaction
            interaction_out = torch.zeros_like(agent_futures)
        else:
            # Apply interaction attention
            interaction_out = self.interaction_attn(agent_futures, interaction_mask)  # (B, N, D)
            
            # Handle NaN from partially masked batches (some batches have no valid agents)
            if torch.isnan(interaction_out).any():
                # Replace NaN with zeros for batches where all agents are masked
                nan_mask = torch.isnan(interaction_out)
                interaction_out = torch.where(nan_mask, torch.zeros_like(interaction_out), interaction_out)
        
        # 4. Append interaction context to map context
        # Combined context: [interaction; map]
        combined_context = torch.cat([interaction_out, context], dim=1)  # (B, N+S, D)
        
        # Combined mask
        combined_mask = torch.cat([
            interaction_mask, 
            context_mask
        ], dim=1)  # (B, N+S)
        
        # 5. Update query with previous level's trajectory encoding
        # Get target agent's trajectory embedding
        target_traj_embed = traj_embeddings[:, agent_idx]  # (B, K, D)
        
        # Add to previous content
        query = prev_content + target_traj_embed  # (B, K, D)
        
        # 6. Cross-attention to combined context
        content = self.context_attn(query, combined_context, combined_context, combined_mask)
        
        # 7. Predict GMM parameters
        gmm_params, scores = self.predictor(content)
        
        return content, gmm_params, scores


class GameFormerDecoder(nn.Module):
    """
    Full GameFormer Decoder with Level-k Game-Theoretic Reasoning.
    
    Architecture:
        Level 0: InitialDecoder (independent predictions)
        Level 1..L: InteractionDecoder (interactive refinement)
    
    Features:
        - Batched agent processing (all agents in parallel, not sequential loops)
        - Gated encoder residual for direct encoder information flow
        - GMM output for uncertainty quantification
        - Rich trajectory encoding (8-dim states)
        - Agent-specific query embeddings
        - Self-masking in interaction stages
        - Multi-level reasoning for game-theoretic predictions
    
    The output for each level contains:
        - 'traj': (B, N, K, T, 2) mean trajectories
        - 'gmm_params': (B, N, K, T, 4) full GMM parameters
        - 'scores': (B, N, K) mode scores
    """
    def __init__(self, config):
        super(GameFormerDecoder, self).__init__()
        self.config = config
        
        # Extract configuration
        decoder_config = config['model']['decoder']
        self.num_modes = decoder_config['num_modes']
        self.future_len = decoder_config['future_steps']
        self.hidden_dim = decoder_config['d_model']
        self.num_levels = decoder_config.get('num_decoder_layers', 3)
        
        # Maximum agents (for agent embeddings)
        self.max_agents = config['data'].get('max_agents', 32)
        
        # Time step for velocity computation
        obs_hz = 10  # 10 Hz
        self.dt = 1.0 / obs_hz
        
        # Novel feature flags
        novelty_config = config.get('novelty', {})
        self.use_spab = novelty_config.get('use_spab', True)
        self.use_lcma = novelty_config.get('use_lcma', True)
        self.use_gated_residual = decoder_config.get('use_gated_residual', True)
        
        # Level-0: Initial decoder
        self.initial_decoder = InitialDecoder(
            num_modes=self.num_modes,
            max_agents=self.max_agents,
            future_len=self.future_len,
            hidden_dim=self.hidden_dim
        )
        
        # LCMA: Lane-Conditioned Mode Anchoring (Novel Contribution 2)
        if self.use_lcma:
            self.lcma = LaneConditionedModeAnchoring(
                hidden_dim=self.hidden_dim,
                num_modes=self.num_modes
            )
        
        # Level 1..L: Interaction decoders
        self.interaction_decoders = nn.ModuleList([
            InteractionDecoder(
                num_modes=self.num_modes,
                future_len=self.future_len,
                hidden_dim=self.hidden_dim,
                dt=self.dt
            )
            for _ in range(self.num_levels)
        ])
        
        # SPAB: Social Potential Attention Bias (Novel Contribution 1)
        if self.use_spab:
            nhead = decoder_config.get('nhead', 8)
            self.spab = SocialPotentialAttentionBias(num_heads=nhead)
        
        # Gated encoder residual connections (one gate per level)
        self.encoder_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Sigmoid()
            )
            for _ in range(self.num_levels + 1)
        ])

    def forward(
        self,
        agent_features,
        context,
        context_mask,
        agent_mask=None,
        current_states=None,
        vehicle_sizes=None,
        priors=None
    ):
        """
        Forward pass through all decoder levels — batched across agents.
        
        Args:
            agent_features: (B, N, D) encoded agent features from encoder
            context: (B, S, D) map context features
            context_mask: (B, S) True for padding (ignored in attention)
            agent_mask: (B, N) True for valid agents
            current_states: (B, N, 2) current agent positions (optional)
            vehicle_sizes: (B, N, 3) vehicle dimensions (optional)
            priors: (B, N, K, T, 2) CV extrapolation priors (used at all levels)
            
        Returns:
            outputs: Dict with keys 'level_0', 'level_1', ..., each containing:
                - 'traj': (B, N, K, T, 2) mean trajectories
                - 'gmm_params': (B, N, K, T, 4) full GMM parameters
                - 'scores': (B, N, K) mode scores
        """
        B, N, D = agent_features.shape
        device = agent_features.device
        K = self.num_modes
        T = self.future_len
        S = context.size(1)
        
        # Default agent mask (all valid)
        if agent_mask is None:
            agent_mask = torch.ones(B, N, dtype=torch.bool, device=device)
        
        outputs = {}
        
        # ===== Level 0: Batched Initial Predictions =====
        # LCMA: Use lane-conditioned mode queries instead of generic embeddings
        if self.use_lcma:
            # Generate mode queries from lane graph context
            agent_flat = agent_features.reshape(B * N, D)
            ctx_flat = context.unsqueeze(1).expand(B, N, S, D).reshape(B * N, S, D)
            mask_flat = context_mask.unsqueeze(1).expand(B, N, S).reshape(B * N, S)
            
            # LCMA produces lane-anchored mode queries: (B*N, K, D)
            lcma_queries = self.lcma(agent_flat, ctx_flat, mask_flat)
            
            # Add agent-specific embeddings
            agent_embeds = self.initial_decoder.agent_embedding(
                torch.arange(N, device=device).clamp(max=self.initial_decoder.max_agents - 1)
            )
            queries = (
                lcma_queries.view(B, N, K, D) +
                agent_embeds.view(1, N, 1, D) +
                agent_features.unsqueeze(2)
            )
        else:
            modal_embed = self.initial_decoder.modal_embedding(
                self.initial_decoder.mode_indices
            )  # (K, D)
            agent_embeds = self.initial_decoder.agent_embedding(
                torch.arange(N, device=device).clamp(max=self.initial_decoder.max_agents - 1)
            )  # (N, D)
            queries = (
                modal_embed.view(1, 1, K, D) +
                agent_embeds.view(1, N, 1, D) +
                agent_features.unsqueeze(2)
            )
        
        # Training-time noise injection: add Gaussian noise to mode queries to
        # break symmetry and force different modes to explore different predictions.
        # This prevents mode collapse where WTA drives all modes to the same output.
        if self.training:
            noise_scale = 0.1 * (queries.std(dim=-1, keepdim=True).clamp(min=0.01))
            queries = queries + noise_scale * torch.randn_like(queries)
        
        # Expand shared context for all agents: (B*N, S, D)
        ctx_exp = context.unsqueeze(1).expand(B, N, S, D).reshape(B * N, S, D)
        mask_exp = context_mask.unsqueeze(1).expand(B, N, S).reshape(B * N, S)
        
        # Run batched cross-attention + prediction
        q_flat = queries.reshape(B * N, K, D)
        content_flat = self.initial_decoder.context_attn(q_flat, ctx_exp, ctx_exp, mask_exp)
        
        # Gated encoder residual
        if self.use_gated_residual:
            gate_val = self.encoder_gates[0](agent_features.reshape(B * N, D))  # (B*N, D)
            content_flat = content_flat + gate_val.unsqueeze(1) * agent_features.reshape(B * N, 1, D)
        
        gmm_flat, scores_flat = self.initial_decoder.predictor(content_flat)
        
        # Add position offsets or priors
        if priors is not None:
            # Hybrid Initialization: Add PV3 priors (Residual Learning)
            # priors: (B, N, K, T, 2)
            prior_flat = priors.reshape(B * N, K, T, 2)
            gmm_flat = gmm_flat.clone()
            gmm_flat[..., :2] = gmm_flat[..., :2] + prior_flat
            
        elif current_states is not None:
            # Default: Add current state offset (Stationary Prior)
            offset = current_states.reshape(B * N, 1, 1, 2)
            gmm_flat = gmm_flat.clone()
            gmm_flat[..., :2] = gmm_flat[..., :2] + offset
        
        # Reshape
        prev_contents = content_flat.view(B, N, K, D)
        prev_gmm = gmm_flat.view(B, N, K, T, 4)
        prev_scores = scores_flat.view(B, N, K)
        prev_traj = prev_gmm[..., :2]
        
        outputs['level_0'] = {
            'traj': prev_traj,
            'gmm_params': prev_gmm,
            'scores': prev_scores
        }
        
        # Bypassing recursive interaction decoders if ablation active
        if self.config.get('ablation', {}).get('bypass_interaction_decoder', False):
            return outputs
        
        # ===== Levels 1..L: Batched Interaction =====
        for level_idx, decoder in enumerate(self.interaction_decoders):
            # 1. FutureEncoder — computed ONCE (shared across all target agents)
            traj_embeddings = decoder.future_encoder(
                prev_traj, current_states, vehicle_sizes
            )  # (B, N, K, D)
            
            mode_weights = F.softmax(prev_scores, dim=-1).unsqueeze(-1)  # (B, N, K, 1)
            agent_futures = (traj_embeddings * mode_weights).sum(dim=2)  # (B, N, D)
            
            # 2. Batched self-attention with per-agent self-masking
            base_int_mask = ~agent_mask  # (B, N) True=ignore
            
            # Create N copies with different self-masks: (B, N, N)
            per_agent_masks = base_int_mask.unsqueeze(1).expand(B, N, N).clone()
            per_agent_masks[:, torch.arange(N), torch.arange(N)] = True  # diagonal self-mask
            
            # Expand agent_futures for each target: (B*N, N, D)
            futures_exp = agent_futures.unsqueeze(1).expand(B, N, N, D).reshape(B * N, N, D)
            flat_int_masks = per_agent_masks.reshape(B * N, N)
            
            # SPAB: Compute social potential attention bias
            spab_bias = None
            if self.use_spab and current_states is not None:
                spab_bias = self.spab(current_states, N)
            
            # Check for all-masked
            all_masked = flat_int_masks.all(dim=1)
            if all_masked.all():
                interaction_out = torch.zeros(B, N, N, D, device=device)
            else:
                int_flat = decoder.interaction_attn(futures_exp, flat_int_masks, attn_bias=spab_bias)
                int_flat = torch.nan_to_num(int_flat, 0.0)
                interaction_out = int_flat.view(B, N, N, D)
            
            # 3. Build per-agent combined contexts
            ctx_expanded = context.unsqueeze(1).expand(B, N, S, D)
            combined_ctx = torch.cat([interaction_out, ctx_expanded], dim=2)  # (B, N, N+S, D)
            
            # Build per-agent masks with self-masking
            base_cross = torch.cat([base_int_mask, context_mask], dim=1)  # (B, N+S)
            all_masks = base_cross.unsqueeze(1).expand(B, N, -1).clone()  # (B, N, N+S)
            all_masks[:, torch.arange(N), torch.arange(N)] = True  # self-mask in interaction portion
            
            # 4. Build queries: prev_content + own traj embedding
            all_queries = prev_contents + traj_embeddings  # (B, N, K, D)
            
            # 5. Flatten and run batched cross-attention
            NS = combined_ctx.size(2)
            flat_q = all_queries.reshape(B * N, K, D)
            flat_ctx = combined_ctx.reshape(B * N, NS, D)
            flat_masks = all_masks.reshape(B * N, NS)
            
            content_flat = decoder.context_attn(flat_q, flat_ctx, flat_ctx, flat_masks)
            
            # Gated encoder residual
            if self.use_gated_residual:
                gate_val = self.encoder_gates[level_idx + 1](agent_features.reshape(B * N, D))
                content_flat = content_flat + gate_val.unsqueeze(1) * agent_features.reshape(B * N, 1, D)
            
            gmm_flat, scores_flat = decoder.predictor(content_flat)
            
            # Add CV prior offsets (same as Level 0 for consistency)
            if priors is not None:
                prior_flat = priors.reshape(B * N, K, T, 2)
                gmm_flat = gmm_flat.clone()
                gmm_flat[..., :2] = gmm_flat[..., :2] + prior_flat
            elif current_states is not None:
                offset = current_states.reshape(B * N, 1, 1, 2)
                gmm_flat = gmm_flat.clone()
                gmm_flat[..., :2] = gmm_flat[..., :2] + offset
            
            # Reshape
            prev_contents = content_flat.view(B, N, K, D)
            level_gmm = gmm_flat.view(B, N, K, T, 4)
            level_scores = scores_flat.view(B, N, K)
            prev_traj = level_gmm[..., :2]
            prev_scores = level_scores
            
            outputs[f'level_{level_idx + 1}'] = {
                'traj': prev_traj,
                'gmm_params': level_gmm,
                'scores': level_scores
            }
        
        return outputs
