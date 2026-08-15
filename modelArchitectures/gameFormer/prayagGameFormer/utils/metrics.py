"""
Comprehensive evaluation metrics for trajectory prediction.

Implements all metrics required for the comparison study:
- minADE@k, minFDE@k (k=1,4)
- Miss Rate @10px, @20px
- Normalized FDE
- APD (Average Pairwise Distance - Diversity)
- NLL (Negative Log-Likelihood)
- Collision Rate
- Off-Road Rate
- MSS (Model Selection Score)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy.stats import multivariate_normal


def computeADE(
    preds: np.ndarray,
    gt: np.ndarray,
    hasPreds: Optional[np.ndarray] = None
) -> float:
    """
    Compute Average Displacement Error.
    
    Args:
        preds: Predictions [N, numPreds, 2]
        gt: Ground truth [N, numPreds, 2]
        hasPreds: Valid mask [N, numPreds] or [N]
        
    Returns:
        ADE value
    """
    if hasPreds is not None:
        if hasPreds.ndim == 2:
            mask = hasPreds.any(axis=1)
        else:
            mask = hasPreds
        preds = preds[mask]
        gt = gt[mask]
    
    if len(preds) == 0:
        return 0.0
    
    err = np.sqrt(((preds - gt) ** 2).sum(axis=-1))
    return err.mean()


def computeFDE(
    preds: np.ndarray,
    gt: np.ndarray,
    hasPreds: Optional[np.ndarray] = None
) -> float:
    """
    Compute Final Displacement Error.
    
    Args:
        preds: Predictions [N, numPreds, 2]
        gt: Ground truth [N, numPreds, 2]
        hasPreds: Valid mask [N, numPreds] or [N]
        
    Returns:
        FDE value
    """
    if hasPreds is not None:
        if hasPreds.ndim == 2:
            mask = hasPreds.any(axis=1)
        else:
            mask = hasPreds
        preds = preds[mask]
        gt = gt[mask]
    
    if len(preds) == 0:
        return 0.0
    
    err = np.sqrt(((preds[:, -1] - gt[:, -1]) ** 2).sum(axis=-1))
    return err.mean()


def computeMinADEK(
    preds: np.ndarray,
    gt: np.ndarray,
    k: int = 1,
    hasPreds: Optional[np.ndarray] = None
) -> Tuple[float, np.ndarray]:
    """
    Compute minimum ADE across top-k modes.
    
    Args:
        preds: Multi-modal predictions [N, numModes, numPreds, 2]
        gt: Ground truth [N, numPreds, 2]
        k: Number of modes to consider
        hasPreds: Valid mask [N, numPreds] or [N]
        
    Returns:
        Tuple of (minADE@k, bestModeIndices)
    """
    preds = np.asarray(preds, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    
    if hasPreds is not None:
        if hasPreds.ndim == 2:
            mask = hasPreds.any(axis=1)
        else:
            mask = hasPreds.astype(bool)
        preds = preds[mask]
        gt = gt[mask]
    
    if len(preds) == 0:
        return 0.0, np.array([])
    
    # Consider only top-k modes
    preds = preds[:, :k]
    
    # Error per mode: [N, k]
    err = np.sqrt(((preds - gt[:, None, :, :]) ** 2).sum(axis=-1)).mean(axis=-1)
    
    # Min across modes
    bestModes = err.argmin(axis=1)
    minAde = err[np.arange(len(err)), bestModes].mean()
    
    return minAde, bestModes


def computeMinFDEK(
    preds: np.ndarray,
    gt: np.ndarray,
    k: int = 1,
    hasPreds: Optional[np.ndarray] = None
) -> Tuple[float, np.ndarray]:
    """
    Compute minimum FDE across top-k modes.
    
    Args:
        preds: Multi-modal predictions [N, numModes, numPreds, 2]
        gt: Ground truth [N, numPreds, 2]
        k: Number of modes to consider
        hasPreds: Valid mask [N, numPreds] or [N]
        
    Returns:
        Tuple of (minFDE@k, bestModeIndices)
    """
    preds = np.asarray(preds, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    
    if hasPreds is not None:
        if hasPreds.ndim == 2:
            mask = hasPreds.any(axis=1)
        else:
            mask = hasPreds.astype(bool)
        preds = preds[mask]
        gt = gt[mask]
    
    if len(preds) == 0:
        return 0.0, np.array([])
    
    # Consider only top-k modes
    preds = preds[:, :k]
    
    # FDE per mode: [N, k]
    err = np.sqrt(((preds[:, :, -1, :] - gt[:, None, -1, :]) ** 2).sum(axis=-1))
    
    # Min across modes
    bestModes = err.argmin(axis=1)
    minFde = err[np.arange(len(err)), bestModes].mean()
    
    return minFde, bestModes


def computeMissRate(
    preds: np.ndarray,
    gt: np.ndarray,
    threshold: float = 10.0,
    k: int = 1,
    hasPreds: Optional[np.ndarray] = None
) -> float:
    """
    Compute miss rate at given threshold.
    
    A prediction is a "miss" if the FDE > threshold for all top-k modes.
    
    Args:
        preds: Multi-modal predictions [N, numModes, numPreds, 2]
        gt: Ground truth [N, numPreds, 2]
        threshold: Distance threshold in pixels
        k: Number of modes to consider
        hasPreds: Valid mask
        
    Returns:
        Miss rate (0-1)
    """
    preds = np.asarray(preds, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    
    if hasPreds is not None:
        if hasPreds.ndim == 2:
            mask = hasPreds.any(axis=1)
        else:
            mask = hasPreds.astype(bool)
        preds = preds[mask]
        gt = gt[mask]
    
    if len(preds) == 0:
        return 0.0
    
    # Consider only top-k modes
    preds = preds[:, :k]
    
    # FDE per mode: [N, k]
    fde = np.sqrt(((preds[:, :, -1, :] - gt[:, None, -1, :]) ** 2).sum(axis=-1))
    
    # Miss if all modes exceed threshold
    minFde = fde.min(axis=1)
    missRate = (minFde > threshold).mean()
    
    return missRate


def computeNormFDE(
    preds: np.ndarray,
    gt: np.ndarray,
    obsLen: int,
    hasPreds: Optional[np.ndarray] = None
) -> float:
    """
    Compute normalized FDE (FDE / trajectory length).
    
    Args:
        preds: Predictions [N, numPreds, 2] or [N, numModes, numPreds, 2]
        gt: Ground truth [N, numPreds, 2]
        obsLen: Observation length (used to compute trajectory length)
        hasPreds: Valid mask
        
    Returns:
        Normalized FDE
    """
    preds = np.asarray(preds, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    
    if hasPreds is not None:
        if hasPreds.ndim == 2:
            mask = hasPreds.any(axis=1)
        else:
            mask = hasPreds.astype(bool)
        preds = preds[mask]
        gt = gt[mask]
    
    if len(preds) == 0:
        return 0.0
    
    # If multi-modal, use best mode
    if preds.ndim == 4:
        fde = np.sqrt(((preds[:, :, -1, :] - gt[:, None, -1, :]) ** 2).sum(axis=-1))
        bestModes = fde.argmin(axis=1)
        preds = preds[np.arange(len(preds)), bestModes]
    
    # Compute FDE
    fde = np.sqrt(((preds[:, -1] - gt[:, -1]) ** 2).sum(axis=-1))
    
    # Compute trajectory length (from GT)
    trajLen = np.sqrt((np.diff(gt, axis=1) ** 2).sum(axis=-1)).sum(axis=-1)
    trajLen = np.maximum(trajLen, 1e-6)  # Avoid division by zero
    
    normFde = (fde / trajLen).mean()
    return normFde


def computeAPD(
    preds: np.ndarray,
    k: int = 6
) -> float:
    """
    Compute Average Pairwise Distance (diversity metric).
    
    Measures the diversity of multi-modal predictions.
    
    Args:
        preds: Multi-modal predictions [N, numModes, numPreds, 2]
        k: Number of modes to consider
        
    Returns:
        APD value (average pairwise distance between modes)
    """
    preds = np.asarray(preds, dtype=np.float32)
    
    if len(preds) == 0:
        return 0.0
    
    # Consider only top-k modes
    preds = preds[:, :k]
    numModes = preds.shape[1]
    
    if numModes < 2:
        return 0.0
    
    # Compute pairwise distances between final positions
    finalPos = preds[:, :, -1, :]  # [N, k, 2]
    
    totalDist = 0.0
    numPairs = 0
    
    for i in range(numModes):
        for j in range(i + 1, numModes):
            dist = np.sqrt(((finalPos[:, i] - finalPos[:, j]) ** 2).sum(axis=-1))
            totalDist += dist.mean()
            numPairs += 1
    
    if numPairs == 0:
        return 0.0
    
    return totalDist / numPairs


def computeNLL(
    preds: np.ndarray,
    gt: np.ndarray,
    scores: Optional[np.ndarray] = None,
    sigma: float = 1.0,
    hasPreds: Optional[np.ndarray] = None
) -> float:
    """
    Compute Negative Log-Likelihood (probabilistic metric).
    
    Args:
        preds: Multi-modal predictions [N, numModes, numPreds, 2]
        gt: Ground truth [N, numPreds, 2]
        scores: Mode probabilities [N, numModes] (softmax applied if not normalized)
        sigma: Gaussian std for likelihood computation
        hasPreds: Valid mask
        
    Returns:
        NLL value
    """
    preds = np.asarray(preds, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    
    if hasPreds is not None:
        if hasPreds.ndim == 2:
            mask = hasPreds.any(axis=1)
        else:
            mask = hasPreds.astype(bool)
        preds = preds[mask]
        gt = gt[mask]
        if scores is not None:
            scores = scores[mask]
    
    if len(preds) == 0:
        return 0.0
    
    N, numModes, numPreds, _ = preds.shape
    
    # Default uniform scores if not provided
    if scores is None:
        scores = np.ones((N, numModes)) / numModes
    else:
        # Apply softmax if not normalized
        scores = np.exp(scores - scores.max(axis=1, keepdims=True))
        scores = scores / scores.sum(axis=1, keepdims=True)
    
    # Compute Gaussian log-likelihood for each mode
    logLik = np.zeros((N, numModes))
    
    for m in range(numModes):
        diff = preds[:, m] - gt  # [N, numPreds, 2]
        sqDist = (diff ** 2).sum(axis=-1)  # [N, numPreds]
        
        # Gaussian log-likelihood (sum over time)
        logLik[:, m] = -0.5 * sqDist.sum(axis=-1) / (sigma ** 2) - \
                       numPreds * np.log(2 * np.pi * sigma ** 2)
    
    # Log-sum-exp for mixture
    weightedLogLik = logLik + np.log(scores + 1e-10)
    maxLogLik = weightedLogLik.max(axis=1, keepdims=True)
    logSumExp = maxLogLik[:, 0] + np.log(np.exp(weightedLogLik - maxLogLik).sum(axis=1))
    
    nll = -logSumExp.mean()
    return nll


def computeCollisionRate(
    preds: np.ndarray,
    neighborPreds: np.ndarray,
    collisionThreshold: float = 5.0,
    hasPreds: Optional[np.ndarray] = None
) -> float:
    """
    Compute collision rate.
    
    Args:
        preds: Ego predictions [N, numModes, numPreds, 2] or [N, numPreds, 2]
        neighborPreds: Neighbor future positions [N, numNeighbors, numPreds, 2]
        collisionThreshold: Distance threshold for collision (pixels)
        hasPreds: Valid mask
        
    Returns:
        Collision rate (0-1)
    """
    preds = np.asarray(preds, dtype=np.float32)
    neighborPreds = np.asarray(neighborPreds, dtype=np.float32)
    
    if hasPreds is not None:
        if hasPreds.ndim == 2:
            mask = hasPreds.any(axis=1)
        else:
            mask = hasPreds.astype(bool)
        preds = preds[mask]
        neighborPreds = neighborPreds[mask]
    
    if len(preds) == 0 or len(neighborPreds) == 0:
        return 0.0
    
    # If multi-modal, use mode 0 (best mode)
    if preds.ndim == 4:
        preds = preds[:, 0]  # [N, numPreds, 2]
    
    N, numPreds, _ = preds.shape
    numNeighbors = neighborPreds.shape[1]
    
    # Check collision for each sample
    collisions = np.zeros(N, dtype=bool)
    
    for i in range(N):
        for t in range(numPreds):
            for n in range(numNeighbors):
                # Skip if neighbor not present
                if np.all(neighborPreds[i, n, t] == 0):
                    continue
                
                dist = np.sqrt(((preds[i, t] - neighborPreds[i, n, t]) ** 2).sum())
                if dist < collisionThreshold:
                    collisions[i] = True
                    break
            if collisions[i]:
                break
    
    return collisions.mean()


def computeOffRoadRate(
    preds: np.ndarray,
    roadMask: np.ndarray,
    hasPreds: Optional[np.ndarray] = None
) -> float:
    """
    Compute off-road rate.
    
    Args:
        preds: Predictions [N, numModes, numPreds, 2] or [N, numPreds, 2]
        roadMask: Binary road mask [H, W]
        hasPreds: Valid mask
        
    Returns:
        Off-road rate (0-1)
    """
    preds = np.asarray(preds, dtype=np.float32)
    
    if hasPreds is not None:
        if hasPreds.ndim == 2:
            mask = hasPreds.any(axis=1)
        else:
            mask = hasPreds.astype(bool)
        preds = preds[mask]
    
    if len(preds) == 0:
        return 0.0
    
    # If multi-modal, use mode 0 (best mode)
    if preds.ndim == 4:
        preds = preds[:, 0]  # [N, numPreds, 2]
    
    H, W = roadMask.shape
    N, numPreds, _ = preds.shape
    
    # Count off-road points
    offRoadCount = 0
    totalCount = 0
    
    for i in range(N):
        for t in range(numPreds):
            x, y = preds[i, t]
            
            # Clamp to valid range
            xi = int(np.clip(x, 0, W - 1))
            yi = int(np.clip(y, 0, H - 1))
            
            totalCount += 1
            if roadMask[yi, xi] == 0:  # Off road
                offRoadCount += 1
    
    if totalCount == 0:
        return 0.0
    
    return offRoadCount / totalCount


def computeMSS(
    minAde: float,
    minFde: float,
    collisionRate: float,
    offRoadRate: float,
    safetyWeight: float = 10.0
) -> float:
    """
    Compute Model Selection Score (MSS).
    
    MSS = minADE + minFDE + safetyWeight * (CR + ORR)
    
    Lower is better.
    
    Args:
        minAde: Minimum ADE
        minFde: Minimum FDE
        collisionRate: Collision rate (0-1)
        offRoadRate: Off-road rate (0-1)
        safetyWeight: Weight for safety metrics
        
    Returns:
        MSS value
    """
    return minAde + minFde + safetyWeight * (collisionRate + offRoadRate)


# ============================================================================
# Legacy functions for training compatibility
# ============================================================================

def predMetrics(
    preds: np.ndarray,
    gtPreds: np.ndarray,
    hasPreds: np.ndarray
) -> Tuple[float, float, float, float, np.ndarray]:
    """
    Compute prediction metrics (legacy).
    
    Args:
        preds: Predictions [N, numMods, numPreds, 2]
        gtPreds: Ground truth [N, numPreds, 2]
        hasPreds: Valid mask [N, numPreds]
        
    Returns:
        Tuple of (ade1, fde1, ade, fde, minIdcs)
    """
    ade1, _ = computeMinADEK(preds, gtPreds, k=1)
    fde1, _ = computeMinFDEK(preds, gtPreds, k=1)
    ade, minIdcs = computeMinADEK(preds, gtPreds, k=preds.shape[1])
    fde, _ = computeMinFDEK(preds, gtPreds, k=preds.shape[1])
    
    return ade1, fde1, ade, fde, minIdcs


def computeMetrics(metrics: dict) -> Dict[str, float]:
    """
    Compute final metrics from accumulated data.
    
    Args:
        metrics: Dictionary with accumulated predictions and losses
        
    Returns:
        Dictionary of computed metrics
    """
    result = {}
    
    # Count number of batches (for averaging)
    numBatches = 0
    for key in metrics:
        if isinstance(metrics[key], (int, float)):
            continue
        elif isinstance(metrics[key], list) and len(metrics[key]) > 0:
            numBatches = len(metrics[key])
            break
    
    if numBatches == 0:
        numBatches = 1
    
    # Loss metrics - handle different naming conventions
    if "numCls" in metrics and metrics["numCls"] > 0:
        result["clsLoss"] = metrics["clsLoss"] / metrics["numCls"]
    elif "cls_loss" in metrics:
        result["clsLoss"] = metrics["cls_loss"] / numBatches if isinstance(metrics["cls_loss"], (int, float)) else 0.0
    elif "clsLoss" in metrics:
        result["clsLoss"] = metrics["clsLoss"] / numBatches if isinstance(metrics["clsLoss"], (int, float)) else 0.0
    else:
        result["clsLoss"] = 0.0
        
    if "numReg" in metrics and metrics["numReg"] > 0:
        result["regLoss"] = metrics["regLoss"] / metrics["numReg"]
    elif "reg_loss" in metrics:
        result["regLoss"] = metrics["reg_loss"] / numBatches if isinstance(metrics["reg_loss"], (int, float)) else 0.0
    elif "regLoss" in metrics:
        result["regLoss"] = metrics["regLoss"] / numBatches if isinstance(metrics["regLoss"], (int, float)) else 0.0
    else:
        result["regLoss"] = 0.0
    
    result["loss"] = result["clsLoss"] + result["regLoss"]
    
    # Prediction metrics from postOut lists
    for key in ["ade", "fde", "ade1", "fde1", "minAde", "minFde"]:
        if key in metrics:
            vals = metrics[key]
            if isinstance(vals, list) and len(vals) > 0:
                # Average the scalar values
                if hasattr(vals[0], 'item'):
                    result[key] = sum(v.item() for v in vals) / len(vals)
                elif isinstance(vals[0], (int, float, np.floating)):
                    result[key] = sum(float(v) for v in vals) / len(vals)
                else:
                    # Try to convert
                    try:
                        result[key] = sum(float(v) for v in vals) / len(vals)
                    except:
                        pass
    
    # Legacy format with preds arrays
    if "preds" in metrics and len(metrics["preds"]) > 0:
        preds = np.concatenate(metrics["preds"], 0)
        gtPreds = np.concatenate(metrics["gtPreds"], 0)
        hasPreds = np.concatenate(metrics["hasPreds"], 0)
        
        if hasPreds.all() or len(preds) > 0:
            result["ade1"], _ = computeMinADEK(preds, gtPreds, k=1)
            result["fde1"], _ = computeMinFDEK(preds, gtPreds, k=1)
            result["ade"], _ = computeMinADEK(preds, gtPreds, k=preds.shape[1])
            result["fde"], _ = computeMinFDEK(preds, gtPreds, k=preds.shape[1])
    
    return result


def appendMetrics(
    metrics: dict,
    lossOut: dict,
    postOut: Optional[dict] = None
) -> dict:
    """
    Append batch metrics to accumulated metrics.
    
    Args:
        metrics: Accumulated metrics dictionary
        lossOut: Loss output from current batch
        postOut: Post-processed output from current batch
        
    Returns:
        Updated metrics dictionary
    """
    if len(metrics) == 0:
        for key in lossOut:
            if key != "loss":
                metrics[key] = 0.0
        
        if postOut is not None:
            for key in postOut:
                metrics[key] = []
    
    for key in lossOut:
        if key == "loss":
            continue
        if hasattr(lossOut[key], 'item'):
            metrics[key] += lossOut[key].item()
        else:
            metrics[key] += lossOut[key]
    
    if postOut is not None:
        for key in postOut:
            # Convert tensors to Python scalars/lists
            val = postOut[key]
            if hasattr(val, 'item'):
                # Scalar tensor
                metrics[key].append(val.item())
            elif hasattr(val, 'detach'):
                # Tensor - convert to numpy or list
                metrics[key].append(val.detach().cpu().numpy())
            elif isinstance(val, list):
                # List of items - extend the metrics list
                for item in val:
                    if hasattr(item, 'detach'):
                        metrics[key].append(item.detach().cpu().numpy())
                    else:
                        metrics[key].append(item)
            else:
                metrics[key].append(val)
    
    return metrics


class MetricsAccumulator:
    """Accumulator for comprehensive evaluation metrics."""
    
    def __init__(self, obsLen: int = 10, predLen: int = 30):
        """
        Initialize accumulator.
        
        Args:
            obsLen: Observation length
            predLen: Prediction length
        """
        self.obsLen = obsLen
        self.predLen = predLen
        self.reset()
    
    def reset(self) -> None:
        """Reset accumulated data."""
        self.preds = []
        self.gtPreds = []
        self.hasPreds = []
        self.scores = []
        self.neighborPreds = []
        self.roadMasks = []
        self.samples = 0
    
    def update(
        self,
        preds: np.ndarray,
        gtPreds: np.ndarray,
        hasPreds: Optional[np.ndarray] = None,
        scores: Optional[np.ndarray] = None,
        neighborPreds: Optional[np.ndarray] = None,
        roadMask: Optional[np.ndarray] = None
    ) -> None:
        """
        Update with batch data.
        
        Args:
            preds: Predictions [B, numModes, numPreds, 2]
            gtPreds: Ground truth [B, numPreds, 2]
            hasPreds: Valid mask [B, numPreds] or [B]
            scores: Mode probabilities [B, numModes]
            neighborPreds: Neighbor future positions [B, numNeighbors, numPreds, 2]
            roadMask: Road mask [H, W] (shared for batch or [B, H, W])
        """
        self.preds.append(preds)
        self.gtPreds.append(gtPreds)
        
        if hasPreds is not None:
            self.hasPreds.append(hasPreds)
        
        if scores is not None:
            self.scores.append(scores)
        
        if neighborPreds is not None:
            self.neighborPreds.append(neighborPreds)
        
        if roadMask is not None:
            self.roadMasks.append(roadMask)
        
        self.samples += len(preds)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary of metrics
        """
        if len(self.preds) == 0:
            return {}
        
        preds = np.concatenate(self.preds, axis=0)
        gtPreds = np.concatenate(self.gtPreds, axis=0)
        
        hasPreds = None
        if len(self.hasPreds) > 0:
            hasPreds = np.concatenate(self.hasPreds, axis=0)
        
        scores = None
        if len(self.scores) > 0:
            scores = np.concatenate(self.scores, axis=0)
        
        neighborPreds = None
        if len(self.neighborPreds) > 0:
            neighborPreds = np.concatenate(self.neighborPreds, axis=0)
        
        roadMask = None
        if len(self.roadMasks) > 0:
            roadMask = self.roadMasks[0]  # Assume same mask for all
        
        numModes = preds.shape[1] if preds.ndim == 4 else 1
        
        # Compute metrics
        result = {
            "samples": self.samples,
        }
        
        # minADE@k, minFDE@k
        result["minADE@1"], _ = computeMinADEK(preds, gtPreds, k=1, hasPreds=hasPreds)
        result["minADE@4"], _ = computeMinADEK(preds, gtPreds, k=min(4, numModes), hasPreds=hasPreds)
        result["minFDE@1"], _ = computeMinFDEK(preds, gtPreds, k=1, hasPreds=hasPreds)
        result["minFDE@4"], _ = computeMinFDEK(preds, gtPreds, k=min(4, numModes), hasPreds=hasPreds)
        
        # Miss rates
        result["MR@10px"] = computeMissRate(preds, gtPreds, threshold=10.0, k=numModes, hasPreds=hasPreds)
        result["MR@20px"] = computeMissRate(preds, gtPreds, threshold=20.0, k=numModes, hasPreds=hasPreds)
        
        # Normalized FDE
        result["NormFDE"] = computeNormFDE(preds, gtPreds, self.obsLen, hasPreds=hasPreds)
        
        # APD (diversity)
        result["APD"] = computeAPD(preds, k=numModes)
        
        # NLL
        result["NLL"] = computeNLL(preds, gtPreds, scores=scores, hasPreds=hasPreds)
        
        # Collision rate (if neighbor data available)
        if neighborPreds is not None:
            result["CR"] = computeCollisionRate(preds, neighborPreds, collisionThreshold=5.0, hasPreds=hasPreds)
        else:
            result["CR"] = 0.0
        
        # Off-road rate (if road mask available)
        if roadMask is not None:
            result["ORR"] = computeOffRoadRate(preds, roadMask, hasPreds=hasPreds)
        else:
            result["ORR"] = 0.0
        
        # MSS
        result["MSS"] = computeMSS(
            result["minADE@1"], result["minFDE@1"],
            result["CR"], result["ORR"]
        )
        
        return result
    
    def toDict(self) -> Dict[str, Any]:
        """Return raw data for saving."""
        return {
            "preds": [p.tolist() for p in self.preds],
            "gtPreds": [g.tolist() for g in self.gtPreds],
            "hasPreds": [h.tolist() for h in self.hasPreds] if self.hasPreds else [],
            "scores": [s.tolist() for s in self.scores] if self.scores else [],
            "samples": self.samples,
        }
