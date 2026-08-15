"""
GameFormer modules for PrayagGameFormer.

Implements the encoder and decoder modules following the original
GameFormer architecture with minor adaptations for Prayag dataset.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence data."""
    
    def __init__(self, dim: int = 256, maxLen: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(maxLen).unsqueeze(1)
        divTerm = torch.exp(
            torch.arange(0, dim, 2) * (-math.log(10000.0) / dim)
        )
        
        pe = torch.zeros(1, maxLen, dim)
        pe[0, :, 0::2] = torch.sin(position * divTerm)
        pe[0, :, 1::2] = torch.cos(position * divTerm)
        
        self.register_buffer("pe", pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) input tensor
            
        Returns:
            (B, T, D) with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class AgentEncoder(nn.Module):
    """Encodes agent trajectory using LSTM."""
    
    def __init__(self, inputDim: int = 8, hiddenDim: int = 256, numLayers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            inputDim, hiddenDim, numLayers,
            batch_first=True, bidirectional=False
        )
        self.typeEmbed = nn.Embedding(10, hiddenDim, padding_idx=0)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, T, 9) - trajectory with type in last feature
            
        Returns:
            (B, D) encoded agent features
        """
        trajectory, _ = self.lstm(inputs[:, :, :8])
        output = trajectory[:, -1]  # Last hidden state
        
        # Add type embedding
        agentType = inputs[:, -1, 8].int().clamp(0, 9)
        typeEmbed = self.typeEmbed(agentType)
        output = output + typeEmbed
        
        return output


class LaneEncoder(nn.Module):
    """Encodes lane features for GameFormer."""
    
    def __init__(self, dim: int = 256):
        super().__init__()
        
        # Line encoders
        self.selfLine = nn.Linear(3, 128)
        self.leftLine = nn.Linear(3, 128)
        self.rightLine = nn.Linear(3, 128)
        
        # Attribute encoders
        self.speedLimit = nn.Linear(1, 64)
        self.selfType = nn.Embedding(10, 64, padding_idx=0)
        self.leftType = nn.Embedding(15, 64, padding_idx=0)
        self.rightType = nn.Embedding(15, 64, padding_idx=0)
        self.trafficLight = nn.Embedding(10, 64, padding_idx=0)
        self.interpolating = nn.Embedding(3, 64, padding_idx=0)
        self.stopSign = nn.Embedding(3, 64, padding_idx=0)
        
        # Feature fusion
        self.pointnet = nn.Sequential(
            nn.Linear(512, 384),
            nn.ReLU(),
            nn.Linear(384, dim)
        )
        self.posEncode = PositionalEncoding(dim, maxLen=200)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, numLanes, lanePoints, 16) lane features
            
        Returns:
            (B, numLanes, lanePoints, D) encoded lane features
        """
        # Line features
        selfLine = self.selfLine(inputs[..., :3])
        leftLine = self.leftLine(inputs[..., 3:6])
        rightLine = self.rightLine(inputs[..., 6:9])
        
        # Attribute features
        speedLimit = self.speedLimit(inputs[..., 9:10])
        selfType = self.selfType(inputs[..., 10].int().clamp(0, 9))
        leftType = self.leftType(inputs[..., 11].int().clamp(0, 14))
        rightType = self.rightType(inputs[..., 12].int().clamp(0, 14))
        trafficLight = self.trafficLight(inputs[..., 13].int().clamp(0, 9))
        interpolating = self.interpolating(inputs[..., 14].int().clamp(0, 2))
        stopSign = self.stopSign(inputs[..., 15].int().clamp(0, 2))
        
        # Combine attributes
        laneAttr = selfType + leftType + rightType + trafficLight + interpolating + stopSign
        
        # Concatenate all features
        laneEmbed = torch.cat([
            selfLine, leftLine, rightLine, speedLimit, laneAttr
        ], dim=-1)
        
        # Process through pointnet with position encoding
        B, numLanes, numPoints, _ = laneEmbed.shape
        laneEmbed = laneEmbed.view(B * numLanes, numPoints, -1)
        output = self.pointnet(laneEmbed)
        output = self.posEncode(output)
        output = output.view(B, numLanes, numPoints, -1)
        
        return output


class CrosswalkEncoder(nn.Module):
    """Encodes crosswalk features."""
    
    def __init__(self, inputDim: int = 3, dim: int = 256):
        super().__init__()
        self.pointnet = nn.Sequential(
            nn.Linear(inputDim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, dim)
        )
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, numCrosswalks, crosswalkPoints, 3) crosswalk features
            
        Returns:
            (B, numCrosswalks, crosswalkPoints, D) encoded crosswalk features
        """
        return self.pointnet(inputs)


class FutureEncoder(nn.Module):
    """Encodes predicted future trajectories for interaction modeling."""
    
    def __init__(self, dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, dim)
        )
        self.typeEmbed = nn.Embedding(10, dim, padding_idx=0)
    
    def processState(
        self,
        trajs: torch.Tensor,
        currentStates: torch.Tensor
    ) -> torch.Tensor:
        """
        Process trajectories to include velocity and size.
        
        Args:
            trajs: (B, N, M, T, 2) predicted trajectories
            currentStates: (B, N, 9) current agent states
            
        Returns:
            (B, N, M, T, 8) processed trajectories
        """
        B, N, M, T, _ = trajs.shape
        
        # Expand current states for modes
        currentStates = currentStates.unsqueeze(2).expand(-1, -1, M, -1)
        
        # Compute velocities from displacements
        currentXy = currentStates[:, :, :, None, :2]
        xy = torch.cat([currentXy, trajs], dim=-2)
        dxy = torch.diff(xy, dim=-2)
        v = dxy / 0.1  # Assuming 10Hz
        
        # Compute heading
        theta = torch.atan2(
            dxy[..., 1],
            dxy[..., 0].clamp(min=1e-6)
        ).unsqueeze(-1)
        
        # Get size
        size = currentStates[:, :, :, None, 5:8].expand(-1, -1, -1, T, -1)
        
        # Combine: (x, y, heading, vx, vy, w, l, h)
        processed = torch.cat([trajs, theta, v, size], dim=-1)
        
        return processed
    
    def forward(
        self,
        trajs: torch.Tensor,
        currentStates: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            trajs: (B, N, M, T, 2) predicted trajectories (xy only)
            currentStates: (B, N, 9) current agent states
            
        Returns:
            (B, N, M, D) encoded future features
        """
        # Process trajectories
        trajs = self.processState(trajs, currentStates)
        
        # Encode
        B, N, M, T, _ = trajs.shape
        trajs = self.mlp(trajs.detach())
        
        # Max pool over time
        output = torch.max(trajs, dim=-2).values
        
        # Add type embedding
        agentType = currentStates[:, :, None, 8].int().clamp(0, 9)
        typeEmbed = self.typeEmbed(agentType)
        output = output + typeEmbed
        
        return output


class GmmPredictor(nn.Module):
    """GMM-based trajectory predictor."""
    
    def __init__(self, dim: int = 256, futureLen: int = 30):
        super().__init__()
        self.futureLen = futureLen
        
        # Gaussian parameters: mu_x, mu_y, log_sigma_x, log_sigma_y
        self.gaussian = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(512, futureLen * 4)
        )
        
        # Mode score
        self.score = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
    
    def forward(self, inputs: torch.Tensor) -> tuple:
        """
        Args:
            inputs: (B, M, D) mode features
            
        Returns:
            predictions: (B, M, T, 4) gaussian parameters
            scores: (B, M) mode scores
        """
        B, M, _ = inputs.shape
        
        predictions = self.gaussian(inputs)
        predictions = predictions.view(B, M, self.futureLen, 4)
        
        scores = self.score(inputs).squeeze(-1)
        
        return predictions, scores


class SelfTransformer(nn.Module):
    """Self-attention transformer layer."""
    
    def __init__(self, dim: int = 256, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.selfAttention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            inputs: (B, N, D) input features
            mask: (B, N) padding mask (True for padding)
            
        Returns:
            (B, N, D) output features
        """
        attnOut, _ = self.selfAttention(
            inputs, inputs, inputs,
            key_padding_mask=mask
        )
        attnOut = self.norm1(attnOut + inputs)
        output = self.norm2(self.ffn(attnOut) + attnOut)
        
        return output


class CrossTransformer(nn.Module):
    """Cross-attention transformer layer."""
    
    def __init__(self, dim: int = 256, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.crossAttention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            query: (B, M, D) query features
            key: (B, N, D) key features
            value: (B, N, D) value features
            mask: (B, N) padding mask (True for padding)
            
        Returns:
            (B, M, D) output features
        """
        attnOut, _ = self.crossAttention(
            query, key, value,
            key_padding_mask=mask
        )
        attnOut = self.norm1(attnOut)
        output = self.norm2(self.ffn(attnOut) + attnOut)
        
        return output


class InitialDecoder(nn.Module):
    """Initial decoder for level-0 predictions."""
    
    def __init__(
        self,
        numModes: int = 6,
        numAgents: int = 32,
        futureLen: int = 30,
        dim: int = 256
    ):
        super().__init__()
        self.numModes = numModes
        
        # Learnable queries
        self.modalQuery = nn.Embedding(numModes, dim)
        self.agentQuery = nn.Embedding(numAgents, dim)
        
        # Decoder
        self.queryEncoder = CrossTransformer(dim)
        self.predictor = GmmPredictor(dim, futureLen)
        
        # Register indices
        self.register_buffer("modalIdx", torch.arange(numModes))
        self.register_buffer("agentIdx", torch.arange(numAgents))
    
    def forward(
        self,
        agentId: int,
        currentState: torch.Tensor,
        encoding: torch.Tensor,
        mask: torch.Tensor
    ) -> tuple:
        """
        Args:
            agentId: Index of the agent to decode
            currentState: (B, 9) current state of the agent
            encoding: (B, N, D) context encoding
            mask: (B, N) padding mask
            
        Returns:
            queryContent: (B, M, D) decoded query features
            predictions: (B, M, T, 4) gaussian parameters
            scores: (B, M) mode scores
        """
        B = currentState.size(0)
        
        # Create query
        modalQuery = self.modalQuery(self.modalIdx)  # (M, D)
        agentQuery = self.agentQuery(
            self.agentIdx[min(agentId, len(self.agentIdx) - 1)]
        )  # (D,)
        
        query = modalQuery + agentQuery[None, :]  # (M, D)
        query = encoding[:, None, agentId] + query  # (B, M, D)
        
        # Decode
        queryContent = self.queryEncoder(query, encoding, encoding, mask)
        predictions, scores = self.predictor(queryContent)
        
        # Add current position to predictions
        predictions[..., :2] += currentState[:, None, None, :2]
        
        return queryContent, predictions, scores


class InteractionDecoder(nn.Module):
    """Interaction decoder for level-k reasoning."""
    
    def __init__(
        self,
        futureEncoder: nn.Module,
        futureLen: int = 30,
        dim: int = 256
    ):
        super().__init__()
        self.futureEncoder = futureEncoder
        self.interactionEncoder = SelfTransformer(dim)
        self.queryEncoder = CrossTransformer(dim)
        self.decoder = GmmPredictor(dim, futureLen)
    
    def forward(
        self,
        agentId: int,
        currentStates: torch.Tensor,
        actors: torch.Tensor,
        scores: torch.Tensor,
        lastContent: torch.Tensor,
        encoding: torch.Tensor,
        mask: torch.Tensor
    ) -> tuple:
        """
        Args:
            agentId: Index of agent to decode
            currentStates: (B, N, 9) current states of all agents
            actors: (B, N, M, T, 4) predictions from last level
            scores: (B, N, M) mode scores from last level
            lastContent: (B, M, D) query content from last level
            encoding: (B, N, D) context encoding
            mask: (B, N) padding mask
            
        Returns:
            queryContent: (B, M, D) decoded query features
            predictions: (B, M, T, 4) gaussian parameters
            scores: (B, M) mode scores
        """
        B, N, M, T, _ = actors.shape
        
        # Encode futures from last level
        multiFutures = self.futureEncoder(
            actors[..., :2], currentStates[:, :N]
        )  # (B, N, M, D)
        
        # Weighted average by scores
        scoreWeights = scores.softmax(-1).unsqueeze(-1)  # (B, N, M, 1)
        futures = (multiFutures * scoreWeights).mean(dim=2)  # (B, N, D)
        
        # Interaction encoding
        interaction = self.interactionEncoder(futures, mask[:, :N])
        
        # Append to context
        fullEncoding = torch.cat([interaction, encoding], dim=1)
        fullMask = torch.cat([mask[:, :N], mask], dim=1).clone()
        fullMask[:, agentId] = True  # Mask self from last level
        
        # Decode
        query = lastContent + multiFutures[:, agentId]
        queryContent = self.queryEncoder(query, fullEncoding, fullEncoding, fullMask)
        predictions, scores = self.decoder(queryContent)
        
        # Add current position
        predictions[..., :2] += currentStates[:, agentId, None, None, :2]
        
        return queryContent, predictions, scores
