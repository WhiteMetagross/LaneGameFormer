"""
Evaluation Metrics Module
Provides standard trajectory prediction metrics (ADE, FDE, Collision Rate, Off-Road Rate).
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
from road_mask_manager import RoadMaskManager

class TrajectoryEvaluator:
    """
    Calculates evaluation metrics for trajectory prediction.
    """
    
    def __init__(self, road_mask: RoadMaskManager):
        """
        Initialize the evaluator.
        
        Args:
            road_mask: RoadMaskManager instance for off-road checks.
        """
        self.road_mask = road_mask

    def compute_ade(self, pred_path: List[Tuple[float, float]], gt_path: List[Tuple[float, float]]) -> float:
        """
        Compute Average Displacement Error (ADE).
        Mean Euclidean distance between predicted and ground truth points.
        """
        if not pred_path or not gt_path:
            return float('inf')
        
        # Truncate to shorter length to compare
        min_len = min(len(pred_path), len(gt_path))
        if min_len == 0:
            return float('inf')
            
        pred_arr = np.array(pred_path[:min_len])
        gt_arr = np.array(gt_path[:min_len])
        
        distances = np.linalg.norm(pred_arr - gt_arr, axis=1)
        return float(np.mean(distances))

    def compute_fde(self, pred_path: List[Tuple[float, float]], gt_path: List[Tuple[float, float]]) -> float:
        """
        Compute Final Displacement Error (FDE).
        Euclidean distance between the final predicted point and final ground truth point.
        """
        if not pred_path or not gt_path:
            return float('inf')
            
        # Compare the last point of the prediction horizon
        # Assuming paths are sampled at same rate and cover same horizon
        # If lengths differ, we compare the last available point of the shorter path
        # or strictly the point at the target horizon.
        # Here we compare the last point of the shorter sequence to be robust.
        min_len = min(len(pred_path), len(gt_path))
        if min_len == 0:
            return float('inf')
            
        pred_end = np.array(pred_path[min_len-1])
        gt_end = np.array(gt_path[min_len-1])
        
        return float(np.linalg.norm(pred_end - gt_end))

    def check_off_road(self, path: List[Tuple[float, float]]) -> bool:
        """
        Check if any point in the path is off-road.
        Returns True if off-road, False otherwise.
        """
        if not path:
            return False # Empty path is technically not off-road, or invalid.
            
        # We consider a path off-road if a significant portion or the end is off-road.
        # Strict metric: Any point off-road.
        for point in path:
            if not self.road_mask.is_point_on_road(point):
                return True
        return False

    def check_collision(self, pred_path: List[Tuple[float, float]], 
                        other_agents_futures: List[List[Tuple[float, float]]], 
                        collision_radius: float = 20.0) -> bool:
        """
        Check if the predicted path collides with any other agent's ground truth future.
        
        Args:
            pred_path: Predicted path (x, y) tuples.
            other_agents_futures: List of future paths for other agents.
            collision_radius: Distance threshold for collision.
            
        Returns:
            True if collision detected, False otherwise.
        """
        if not pred_path:
            return False
            
        # We assume time-synchronization. Point i in pred_path corresponds to Point i in other_agent_future
        for i, ego_pos in enumerate(pred_path):
            ego_pos_arr = np.array(ego_pos)
            
            for other_path in other_agents_futures:
                if i < len(other_path):
                    other_pos_arr = np.array(other_path[i])
                    dist = np.linalg.norm(ego_pos_arr - other_pos_arr)
                    if dist < collision_radius:
                        return True
        return False

    def evaluate_batch(self, predictions: List[Dict], gt_path: List[Tuple[float, float]], 
                       other_agents_futures: List[List[Tuple[float, float]]],
                       k_list: List[int] = [1, 4]) -> Dict:
        """
        Compute metrics for a single test case (one ego vehicle).
        
        Args:
            predictions: List of prediction dicts (must have 'path_points').
            gt_path: Ground truth future path.
            other_agents_futures: Ground truth futures of neighbors.
            k_list: List of K values to compute minADE/minFDE for.
            
        Returns:
            Dictionary of metrics.
        """
        if not predictions or not gt_path:
            return {}
            
        # Sort predictions by probability/score
        sorted_preds = sorted(predictions, key=lambda x: x.get('probability', 0), reverse=True)
        
        metrics = {}
        
        # 1. Displacement Metrics (minADE, minFDE) for each K
        max_k = max(k_list)
        top_max_k_preds = sorted_preds[:max_k]
        
        # Pre-compute ADEs and FDEs for all top predictions
        ades = []
        fdes = []
        paths = []
        probs = []
        
        for pred in top_max_k_preds:
            path = pred['path_points']
            paths.append(path)
            probs.append(pred.get('probability', 1.0))
            ades.append(self.compute_ade(path, gt_path))
            fdes.append(self.compute_fde(path, gt_path))
            
        for k in k_list:
            if k > len(ades):
                # If fewer predictions than K, use all available
                curr_ades = ades
                curr_fdes = fdes
            else:
                curr_ades = ades[:k]
                curr_fdes = fdes[:k]
                
            metrics[f'minADE@{k}'] = min(curr_ades) if curr_ades else float('inf')
            metrics[f'minFDE@{k}'] = min(curr_fdes) if curr_fdes else float('inf')

        # 2. Miss Rate (based on minFDE@max_k usually, or @4)
        # We'll use the largest K for "system capability" miss rate
        best_fde = metrics[f'minFDE@{max(k_list)}']
        metrics['miss_rate_10'] = 1.0 if best_fde > 10.0 else 0.0
        metrics['miss_rate_20'] = 1.0 if best_fde > 20.0 else 0.0
        
        # 3. Normalized FDE
        # Calculate GT length
        gt_arr = np.array(gt_path)
        if len(gt_arr) > 1:
            # Arc length
            diffs = np.linalg.norm(gt_arr[1:] - gt_arr[:-1], axis=1)
            gt_len = np.sum(diffs)
        else:
            gt_len = 0.0
            
        if gt_len > 1.0: # Avoid division by zero or tiny lengths
            metrics['norm_fde'] = best_fde / gt_len
        else:
            metrics['norm_fde'] = best_fde # Fallback or 0?
            
        # 4. Collision and Off-Road (based on best matching trajectory)
        # Find index of best FDE among the top K
        best_idx = np.argmin(fdes) if fdes else 0
        best_path = paths[best_idx] if paths else []
        
        metrics['collision'] = 1.0 if self.check_collision(best_path, other_agents_futures) else 0.0
        metrics['off_road'] = 1.0 if self.check_off_road(best_path) else 0.0
        
        # 5. Diversity: APD (Average Pairwise Distance)
        # Computed on top K=4 (or max_k)
        if len(paths) > 1:
            pairwise_dists = []
            for i in range(len(paths)):
                for j in range(i + 1, len(paths)):
                    # Compute ADE between path i and path j
                    # Truncate to min len
                    p1 = paths[i]
                    p2 = paths[j]
                    min_len = min(len(p1), len(p2))
                    if min_len > 0:
                        d = np.mean(np.linalg.norm(np.array(p1[:min_len]) - np.array(p2[:min_len]), axis=1))
                        pairwise_dists.append(d)
            metrics['apd'] = np.mean(pairwise_dists) if pairwise_dists else 0.0
        else:
            metrics['apd'] = 0.0
            
        # 6. Probabilistic: NLL (Negative Log Likelihood)
        # Using Gaussian Mixture Model assumption
        # P(y) = sum(w_i * N(y | y_i, sigma))
        # We compute average NLL over time steps
        sigma = 20.0 # Assumed standard deviation in pixels
        
        # Normalize probabilities
        probs = np.array(probs)
        if np.sum(probs) > 0:
            probs = probs / np.sum(probs)
        else:
            probs = np.ones_like(probs) / len(probs)
            
        nll_steps = []
        min_len = min(len(gt_path), min([len(p) for p in paths])) if paths else 0
        
        if min_len > 0:
            for t in range(min_len):
                gt_pt = np.array(gt_path[t])
                
                # Compute likelihood for this point
                likelihood = 0.0
                for i, path in enumerate(paths):
                    pred_pt = np.array(path[t])
                    dist_sq = np.sum((gt_pt - pred_pt)**2)
                    
                    # Gaussian PDF (ignoring constants that cancel out or are fixed)
                    # PDF = (1 / (2*pi*sigma^2)) * exp(-dist^2 / (2*sigma^2))
                    # We can just use the exp part and log later, but need constants for "true" NLL
                    # Let's use a simplified proportional NLL or full one.
                    # Full 2D Gaussian:
                    norm_const = 1.0 / (2 * np.pi * sigma**2)
                    prob_density = norm_const * np.exp(-dist_sq / (2 * sigma**2))
                    
                    likelihood += probs[i] * prob_density
                
                # Avoid log(0)
                likelihood = max(likelihood, 1e-10)
                nll_steps.append(-np.log(likelihood))
            
            metrics['nll'] = np.mean(nll_steps)
        else:
            metrics['nll'] = 0.0 # Or NaN
            
        return metrics
