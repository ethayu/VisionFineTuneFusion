import torch
import torch.nn.functional as F

def reconstruction_loss(original, reconstructed):
    """
    MSE loss for reconstruction.

    Args:
        original (torch.Tensor): Original CLS token.
        reconstructed (torch.Tensor): Reconstructed CLS token.

    Returns:
        torch.Tensor: Reconstruction loss.
    """
    return F.mse_loss(reconstructed, original)

def clip_loss(latent_vector, clip_vector):
    """
    Cosine similarity loss for CLIP embeddings.

    Args:
        latent_vector (torch.Tensor): Latent vector from autoencoder.
        clip_vector (torch.Tensor): Vector from CLIP encoder.

    Returns:
        torch.Tensor: CLIP loss.
    """
    return 1 - F.cosine_similarity(latent_vector, clip_vector, dim=-1).mean()
