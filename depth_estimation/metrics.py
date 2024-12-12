import torch
import numpy as np
from typing import Dict, Union, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> float:
    """
    Compute Mean Absolute Error.
    
    Args:
        pred: Predicted depth map
        target: Ground truth depth map
        mask: Optional mask for valid depth values
        
    Returns:
        MAE value
    """
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    
    return torch.mean(torch.abs(pred - target)).item()

def compute_rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> float:
    """
    Compute Root Mean Square Error.
    
    Args:
        pred: Predicted depth map
        target: Ground truth depth map
        mask: Optional mask for valid depth values
        
    Returns:
        RMSE value
    """
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()

def compute_rel(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> float:
    """
    Compute mean relative error.
    
    Args:
        pred: Predicted depth map
        target: Ground truth depth map
        mask: Optional mask for valid depth values
        
    Returns:
        Relative error value
    """
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    
    return torch.mean(torch.abs(pred - target) / target).item()

def compute_threshold_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 1.25,
    mask: torch.Tensor = None
) -> float:
    """
    Compute accuracy under threshold (δ < threshold).
    
    Args:
        pred: Predicted depth map
        target: Ground truth depth map
        threshold: Accuracy threshold
        mask: Optional mask for valid depth values
        
    Returns:
        Accuracy value
    """
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    
    ratio = torch.maximum(pred / target, target / pred)
    return torch.mean((ratio < threshold).float()).item()

def evaluate_depth_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor = None
) -> Dict[str, float]:
    """
    Compute all depth estimation metrics.
    
    Args:
        pred: Predicted depth map
        target: Ground truth depth map
        mask: Optional mask for valid depth values
        
    Returns:
        Dictionary containing all metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['mae'] = compute_mae(pred, target, mask)
    metrics['rmse'] = compute_rmse(pred, target, mask)
    metrics['rel'] = compute_rel(pred, target, mask)
    
    # Threshold accuracies
    thresholds = [1.25, 1.25**2, 1.25**3]
    for t in thresholds:
        metrics[f'delta_{t}'] = compute_threshold_accuracy(pred, target, t, mask)
    
    return metrics

def log_metrics(metrics: Dict[str, float]) -> None:
    """Print metrics in a formatted way."""
    logger.info("\nDepth Estimation Metrics:")
    logger.info("-" * 30)
    logger.info(f"MAE:      {metrics['mae']:.4f}")
    logger.info(f"RMSE:     {metrics['rmse']:.4f}")
    logger.info(f"Rel:      {metrics['rel']:.4f}")
    logger.info(f"δ < 1.25: {metrics['delta_1.25']:.4f}")
    logger.info(f"δ < 1.25²: {metrics['delta_1.5625']:.4f}")
    logger.info(f"δ < 1.25³: {metrics['delta_1.953125']:.4f}")