"""
GameFormer model for PrayagGameFormer.

Implements the full GameFormer architecture with:
- Encoder: Agent, lane, crosswalk encoding with transformer fusion
- Decoder: Hierarchical level-k reasoning with GMM prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from .modules import (
    AgentEncoder,
    LaneEncoder,
    CrosswalkEncoder,
    FutureEncoder,
    InitialDecoder,
    InteractionDecoder
)


class Encoder(nn.Module):
    """GameFormer encoder with multi-modal fusion."""
    
    def __init__(
        self,
        neighborsToPredict: int = 31,
        layers: int = 6,
        dim: int = 256,
        heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.neighborsToPredict = neighborsToPredict
        
        # Encoders
        self.egoEncoder = AgentEncoder(hiddenDim=dim)
        self.agentEncoder = AgentEncoder(hiddenDim=dim)
        self.laneEncoder = LaneEncoder(dim=dim)
        self.crosswalkEncoder = CrosswalkEncoder(dim=dim)
        
        # Fusion transformer
        encoderLayer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            activation="gelu",
            dropout=dropout,
            batch_first=True
        )
        self.fusionEncoder = nn.TransformerEncoder(
            encoderLayer, layers,
            enable_nested_tensor=False
        )
        
        self.stride = 10  # Stride for map segment pooling
    
    def segmentMap(
        self,
        mapData: torch.Tensor,
        mapEncoding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Segment and pool map encodings.
        
        Args:
            mapData: (B, numElements, numPoints, D_in) raw map data
            mapEncoding: (B, numElements, numPoints, D) encoded map
            
        Returns:
            encoding: (B, N', D) pooled encoding
            mask: (B, N') padding mask
        """
        B, numElements, numPoints, D = mapEncoding.shape
        
        # Max pool to reduce point dimension
        encoding = mapEncoding.permute(0, 3, 1, 2)  # (B, D, E, P)
        encoding = F.max_pool2d(encoding, kernel_size=(1, self.stride))
        encoding = encoding.permute(0, 2, 3, 1)  # (B, E, P', D)
        encoding = encoding.reshape(B, -1, D)
        
        # Compute mask
        mask = torch.eq(mapData, 0)[:, :, :, 0]  # (B, E, P)
        mask = mask.reshape(B, numElements, numPoints // self.stride, self.stride)
        mask = torch.max(mask, dim=-1)[0]
        mask = mask.reshape(B, -1)
        
        return encoding, mask
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Encode inputs for GameFormer decoder.
        
        Args:
            inputs: Dictionary with ego_state, neighbors_state, map_lanes, map_crosswalks
            
        Returns:
            Dictionary with actors, encodings, masks
        """
        # Get inputs
        egoState = inputs["ego_state"]  # (B, T, 9)
        neighborsState = inputs["neighbors_state"]  # (B, N, T, 9)
        mapLanes = inputs["map_lanes"]  # (B, A, numLanes, lanePoints, 16)
        mapCrosswalks = inputs["map_crosswalks"]  # (B, A, numCW, cwPoints, 3)
        
        B = egoState.size(0)
        N = self.neighborsToPredict + 1
        
        # Encode actors — batched neighbor encoding
        encodedEgo = self.egoEncoder(egoState)  # (B, D)
        numNeighbors = min(neighborsState.size(1), N - 1)
        if numNeighbors > 0:
            neighborFlat = neighborsState[:, :numNeighbors].reshape(B * numNeighbors, -1, 9)
            encodedNeighborsFlat = self.agentEncoder(neighborFlat)  # (B*numNeighbors, D)
            encodedNeighbors = encodedNeighborsFlat.view(B, numNeighbors, -1)
            encodedActors = torch.cat([encodedEgo.unsqueeze(1), encodedNeighbors], dim=1)
        else:
            encodedActors = encodedEgo.unsqueeze(1)  # (B, 1, D)
        
        # Actor mask
        actors = torch.cat([
            egoState.unsqueeze(1), neighborsState[:, :N-1]
        ], dim=1)  # (B, N, T, 9)
        actorMask = torch.eq(actors[:, :, -1].sum(-1), 0)  # (B, N)
        
        device = egoState.device
        D = encodedActors.size(-1)
        
        # Encode maps — batched across all agents
        numAgents = min(mapLanes.size(1), N)
        encodedLanes = self.laneEncoder(
            mapLanes[:, :numAgents].reshape(B * numAgents, -1, mapLanes.size(3), 16)
        )
        encodedLanes = encodedLanes.view(
            B, numAgents, -1, encodedLanes.size(2), encodedLanes.size(3)
        )
        
        encodedCrosswalks = self.crosswalkEncoder(
            mapCrosswalks[:, :numAgents].reshape(B * numAgents, -1, mapCrosswalks.size(3), 3)
        )
        encodedCrosswalks = encodedCrosswalks.view(
            B, numAgents, -1, encodedCrosswalks.size(2), encodedCrosswalks.size(3)
        )
        
        # Batched segmentMap for all agents at once
        # Lanes: (B, A, numLanes, lanePoints, 16) → batch segment
        laneDataFlat = mapLanes[:, :numAgents].reshape(B * numAgents, -1, mapLanes.size(3), 16)
        laneEncFlat = encodedLanes.reshape(B * numAgents, -1, encodedLanes.size(3), encodedLanes.size(4))
        lanesFlat, lanesMaskFlat = self.segmentMap(laneDataFlat, laneEncFlat)
        L = lanesFlat.size(1)
        allLanes = lanesFlat.view(B, numAgents, L, D)
        allLanesMask = lanesMaskFlat.view(B, numAgents, L)
        
        cwDataFlat = mapCrosswalks[:, :numAgents].reshape(B * numAgents, -1, mapCrosswalks.size(3), 3)
        cwEncFlat = encodedCrosswalks.reshape(B * numAgents, -1, encodedCrosswalks.size(3), encodedCrosswalks.size(4))
        cwFlat, cwMaskFlat = self.segmentMap(cwDataFlat, cwEncFlat)
        C = cwFlat.size(1)
        allCW = cwFlat.view(B, numAgents, C, D)
        allCWMask = cwMaskFlat.view(B, numAgents, C)
        
        # Pad for agents beyond numAgents
        if N > numAgents:
            padN = N - numAgents
            allLanes = torch.cat([allLanes, torch.zeros(B, padN, L, D, device=device)], dim=1)
            allLanesMask = torch.cat([allLanesMask, torch.ones(B, padN, L, dtype=torch.bool, device=device)], dim=1)
            allCW = torch.cat([allCW, torch.zeros(B, padN, C, D, device=device)], dim=1)
            allCWMask = torch.cat([allCWMask, torch.ones(B, padN, C, dtype=torch.bool, device=device)], dim=1)
        
        # Build fusion inputs for all agents: (B, N, N+L+C, D)
        expActors = encodedActors.unsqueeze(1).expand(B, N, N, D)
        expActorMask = actorMask.unsqueeze(1).expand(B, N, N)
        fusionInput = torch.cat([expActors, allLanes, allCW], dim=2)
        fusionMask = torch.cat([expActorMask, allLanesMask, allCWMask], dim=2)
        S = fusionInput.size(2)
        
        # Run TransformerEncoder in agent groups to manage memory
        agentGroupSize = min(N, 8)
        encodedChunks = []
        for gStart in range(0, N, agentGroupSize):
            gEnd = min(gStart + agentGroupSize, N)
            gs = gEnd - gStart
            groupInput = fusionInput[:, gStart:gEnd].reshape(B * gs, S, D)
            groupMask = fusionMask[:, gStart:gEnd].reshape(B * gs, S)
            chunk = self.fusionEncoder(groupInput, src_key_padding_mask=groupMask)
            encodedChunks.append(chunk.view(B, gs, S, D))
        
        encodings = torch.cat(encodedChunks, dim=1)  # (B, N, S, D)
        masks = fusionMask  # (B, N, S)
        
        return {
            "actors": actors,
            "encodings": encodings,
            "masks": masks
        }


class Decoder(nn.Module):
    """GameFormer decoder with level-k reasoning."""
    
    def __init__(
        self,
        numModes: int = 6,
        futureLen: int = 30,
        neighborsToPredict: int = 31,
        levels: int = 3,
        dim: int = 256
    ):
        super().__init__()
        self.levels = levels
        self.neighborsToPredict = neighborsToPredict
        
        # Initial decoder
        self.initialStage = InitialDecoder(
            numModes, neighborsToPredict + 1, futureLen, dim
        )
        
        # Interaction decoders
        futureEncoder = FutureEncoder(dim)
        self.interactionStages = nn.ModuleList([
            InteractionDecoder(futureEncoder, futureLen, dim)
            for _ in range(levels)
        ])
    
    def forward(
        self,
        encoderOutputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Decode trajectories with level-k reasoning.
        
        Args:
            encoderOutputs: Output from encoder
            
        Returns:
            Dictionary with level_k_interactions and level_k_scores
        """
        outputs = {}
        
        actors = encoderOutputs["actors"]
        encodings = encoderOutputs["encodings"]
        masks = encoderOutputs["masks"]
        
        B, N = actors.shape[:2]
        N = min(N, self.neighborsToPredict + 1)
        device = actors.device
        
        currentStates = actors[:, :N, -1]  # (B, N, 9)
        S = encodings.size(2)
        D = encodings.size(3)
        M = self.initialStage.numModes
        T = self.initialStage.predictor.futureLen
        
        # ===== Batched Level 0 =====
        # Build queries for all agents at once
        modalEmbed = self.initialStage.modalQuery(self.initialStage.modalIdx)  # (M, D)
        agentEmbeds = self.initialStage.agentQuery(
            torch.arange(N, device=device).clamp(max=self.initialStage.agentIdx.size(0) - 1)
        )  # (N, D)
        # Diagonal tokens: encoding[b, i, i] for each agent
        diagTokens = encodings[:, torch.arange(N), torch.arange(N)]  # (B, N, D)
        
        queries = (
            diagTokens.unsqueeze(2) +           # (B, N, 1, D)
            modalEmbed.view(1, 1, M, D) +       # (1, 1, M, D)
            agentEmbeds.view(1, N, 1, D)         # (1, N, 1, D)
        )  # (B, N, M, D)
        
        # Flatten: (B*N, M, D), (B*N, S, D), (B*N, S)
        flatQ = queries.reshape(B * N, M, D)
        flatCtx = encodings[:, :N].reshape(B * N, S, D)
        flatMask = masks[:, :N].reshape(B * N, S)
        
        flatContent = self.initialStage.queryEncoder(flatQ, flatCtx, flatCtx, flatMask)
        flatPred, flatScores = self.initialStage.predictor(flatContent)
        
        # Add position offset
        posOffset = currentStates[:, :N, :2].reshape(B * N, 1, 1, 2)
        flatPred[..., :2] = flatPred[..., :2] + posOffset
        
        lastContent = flatContent.view(B, N, M, D)
        lastLevel = flatPred.view(B, N, M, T, 4)
        lastScores = flatScores.view(B, N, M)
        
        outputs["level_0_interactions"] = lastLevel
        outputs["level_0_scores"] = lastScores
        
        # ===== Batched Level-k Reasoning =====
        for k in range(1, self.levels + 1):
            interactionDecoder = self.interactionStages[k - 1]
            
            # Compute FutureEncoder ONCE (shared across all target agents)
            multiFutures = interactionDecoder.futureEncoder(
                lastLevel[..., :2], currentStates[:, :N]
            )  # (B, N, M, D)
            scoreWeights = lastScores.softmax(-1).unsqueeze(-1)  # (B, N, M, 1)
            futures = (multiFutures * scoreWeights).mean(dim=2)  # (B, N, D)
            
            # Self-attention for interaction (ONCE — no per-agent masking here)
            interMask = masks[:, 0, :N]  # (B, N) — actor mask from first agent's view
            interaction = interactionDecoder.interactionEncoder(futures, interMask)  # (B, N, D)
            
            # Build per-agent cross-attention contexts
            # Each agent i gets: cat(interaction, encodings[:, i]) with self-mask at i
            interExpanded = interaction.unsqueeze(1).expand(B, N, N, D)
            fullCtx = torch.cat([interExpanded, encodings[:, :N]], dim=2)  # (B, N, N+S, D)
            
            interMaskExp = interMask.unsqueeze(1).expand(B, N, N)
            fullMask = torch.cat([interMaskExp, masks[:, :N]], dim=2).clone()  # (B, N, N+S)
            # Self-mask: position i for target agent i
            fullMask[:, torch.arange(N), torch.arange(N)] = True
            
            # Queries: lastContent[:, i] + multiFutures[:, i]
            allQueries = lastContent + multiFutures  # (B, N, M, D)
            
            # Flatten and run batched cross-attention
            N_plus_S = fullCtx.size(2)
            flatQ = allQueries.reshape(B * N, M, D)
            flatCtx = fullCtx.reshape(B * N, N_plus_S, D)
            flatMask = fullMask.reshape(B * N, N_plus_S)
            
            flatContent = interactionDecoder.queryEncoder(flatQ, flatCtx, flatCtx, flatMask)
            flatPred, flatScores = interactionDecoder.decoder(flatContent)
            
            # Position offset
            posOffset = currentStates[:, :N, :2].reshape(B * N, 1, 1, 2)
            flatPred[..., :2] = flatPred[..., :2] + posOffset
            
            lastContent = flatContent.view(B, N, M, D)
            lastLevel = flatPred.view(B, N, M, T, 4)
            lastScores = flatScores.view(B, N, M)
            
            outputs[f"level_{k}_interactions"] = lastLevel
            outputs[f"level_{k}_scores"] = lastScores
        
        return outputs


class GameFormer(nn.Module):
    """Full GameFormer model."""
    
    def __init__(self, config: dict):
        super().__init__()
        
        self.numModes = config.get("numModes", 6)
        self.futureLen = config.get("predHorizon", 30)
        self.neighborsToPredict = config.get("neighborsToPredict", 31)
        
        dim = config.get("dim", 256)
        encoderLayers = config.get("encoderLayers", 6)
        decoderLevels = config.get("decoderLevels", 3)
        
        self.encoder = Encoder(
            neighborsToPredict=self.neighborsToPredict,
            layers=encoderLayers,
            dim=dim
        )
        self.decoder = Decoder(
            numModes=self.numModes,
            futureLen=self.futureLen,
            neighborsToPredict=self.neighborsToPredict,
            levels=decoderLevels,
            dim=dim
        )
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            inputs: Dictionary with input tensors
            
        Returns:
            Dictionary with predictions at each level
        """
        encoderOutputs = self.encoder(inputs)
        outputs = self.decoder(encoderOutputs)
        return outputs


class GameFormerLoss(nn.Module):
    """Loss function for GameFormer."""
    
    def __init__(self, config: dict):
        super().__init__()
        self.futureLen = config.get("predHorizon", 30)
        self.regWeight = config.get("regWeight", 1.0)
        self.clsWeight = config.get("clsWeight", 1.0)
        self.levels = config.get("decoderLevels", 3)
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss.
        
        Args:
            outputs: Model outputs
            batch: Batch data with ground truth
            
        Returns:
            Dictionary with loss components
        """
        # Get ground truth
        egoFuture = batch["ego_future"]  # (B, T, 2)
        neighborsFuture = batch["neighbors_future"]  # (B, N, T, 2)
        
        gtFuture = torch.cat([
            egoFuture.unsqueeze(1), neighborsFuture
        ], dim=1)  # (B, N+1, T, 2)
        
        totalLoss = 0.0
        regLoss = 0.0
        clsLoss = 0.0
        
        # Loss at each level
        for level in range(self.levels + 1):
            predictions = outputs[f"level_{level}_interactions"]  # (B, N, M, T, 4)
            scores = outputs[f"level_{level}_scores"]  # (B, N, M)
            
            B, N, M, T, _ = predictions.shape
            
            # Only compute loss for first N agents
            N = min(N, gtFuture.size(1))
            predictions = predictions[:, :N]
            scores = scores[:, :N]
            gt = gtFuture[:, :N]  # (B, N, T, 2)
            
            # Regression loss (best mode)
            predXy = predictions[..., :2]  # (B, N, M, T, 2)
            distances = torch.norm(
                predXy - gt.unsqueeze(2), dim=-1
            ).sum(dim=-1)  # (B, N, M)
            
            bestMode = distances.argmin(dim=-1)  # (B, N)
            bestPred = torch.gather(
                predXy,
                dim=2,
                index=bestMode[:, :, None, None, None].expand(-1, -1, -1, T, 2)
            ).squeeze(2)  # (B, N, T, 2)
            
            reg = F.smooth_l1_loss(bestPred, gt)
            regLoss += reg
            
            # Classification loss
            bestModeTarget = F.one_hot(bestMode, M).float()
            cls = F.cross_entropy(
                scores.reshape(-1, M),
                bestMode.reshape(-1)
            )
            clsLoss += cls
            
            totalLoss += self.regWeight * reg + self.clsWeight * cls
        
        return {
            "loss": totalLoss / (self.levels + 1),
            "reg_loss": regLoss / (self.levels + 1),
            "cls_loss": clsLoss / (self.levels + 1)
        }


class PostProcess(nn.Module):
    """Post-processor for computing metrics."""
    
    def __init__(self, config: dict):
        super().__init__()
        self.futureLen = config.get("predHorizon", 30)
        self.levels = config.get("decoderLevels", 3)
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Post-process outputs for metric computation.
        
        Returns ADE/FDE for the final level, ego agent, best mode.
        """
        # Get final level predictions
        level = self.levels
        predictions = outputs[f"level_{level}_interactions"]  # (B, N, M, T, 4)
        scores = outputs[f"level_{level}_scores"]  # (B, N, M)
        
        # Ego agent only (first agent)
        egoPred = predictions[:, 0]  # (B, M, T, 4)
        egoScores = scores[:, 0]  # (B, M)
        
        # Ground truth
        egoFuture = batch["ego_future"]  # (B, T, 2)
        
        # Best mode
        predXy = egoPred[..., :2]  # (B, M, T, 2)
        distances = torch.norm(
            predXy - egoFuture.unsqueeze(1), dim=-1
        )  # (B, M, T)
        
        ade = distances.mean(dim=-1)  # (B, M)
        fde = distances[:, :, -1]  # (B, M)
        
        # Best mode ADE/FDE
        bestMode = ade.argmin(dim=-1)  # (B,)
        bestAde = torch.gather(ade, 1, bestMode.unsqueeze(1)).squeeze(1)
        bestFde = torch.gather(fde, 1, bestMode.unsqueeze(1)).squeeze(1)
        
        return {
            "ade": bestAde.mean(),
            "fde": bestFde.mean(),
            "minAde": ade.min(dim=-1)[0].mean(),
            "minFde": fde.min(dim=-1)[0].mean()
        }
