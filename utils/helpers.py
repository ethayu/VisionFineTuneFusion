import os
import torch

def save_checkpoint(model, path):
    """
    Save the model checkpoint.

    Args:
        model (torch.nn.Module): Model to save.
        path (str): Path to save the checkpoint.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(model, path, device="cpu"):
    """
    Load the model checkpoint.

    Args:
        model (torch.nn.Module): Model to load the checkpoint into.
        path (str): Path of the checkpoint.
        device (str): Device to map the model to.
    
    Returns:
        torch.nn.Module: Model with loaded weights.
    """
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"Checkpoint loaded from {path}")
    return model

def ensure_dir(directory):
    """
    Ensure a directory exists. If not, create it.
    
    Args:
        directory (str): Path to the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
