import torch
import torch.nn.functional as F
import numpy as np

def cosine_similarity_metric(vector1, vector2):
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vector1 (torch.Tensor): First vector.
        vector2 (torch.Tensor): Second vector.
    
    Returns:
        float: Cosine similarity.
    """
    return F.cosine_similarity(vector1, vector2, dim=-1).mean().item()

def reconstruction_error(original, reconstructed):
    """
    Compute mean squared error (MSE) for reconstruction.
    
    Args:
        original (torch.Tensor): Original tensor.
        reconstructed (torch.Tensor): Reconstructed tensor.
    
    Returns:
        float: Reconstruction error.
    """
    return F.mse_loss(reconstructed, original).item()

def mean_absolute_error(predictions, targets):
    """
    Compute Mean Absolute Error (MAE).

    Args:
        predictions (np.ndarray): Predicted depth maps.
        targets (np.ndarray): Ground truth depth maps.

    Returns:
        float: Mean Absolute Error.
    """
    return np.mean(np.abs(predictions - targets))

def rmse(predictions, targets):
    """
    Compute Root Mean Squared Error (RMSE).

    Args:
        predictions (np.ndarray): Predicted depth maps.
        targets (np.ndarray): Ground truth depth maps.

    Returns:
        float: RMSE value.
    """
    return np.sqrt(np.mean((predictions - targets) ** 2))