import torch
from transformers import AutoModel, AutoImageProcessor

def load_dino_model(model_name="facebook/dinov2-base"):
    """
    Load the DiNOv2 model.

    Args:
        model_name (str): Name of the pre-trained DiNO model.

    Returns:
        torch.nn.Module: Loaded DiNO model.
    """
    try:
        model = AutoModel.from_pretrained(model_name)
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading DINO model: {e}")

def extract_cls_token(model, images):
    """
    Extract the class (cls) token from the DiNO model.

    Args:
        model (torch.nn.Module): Loaded DiNO model.
        images (torch.Tensor): Batch of images.

    Returns:
        torch.Tensor: CLS tokens for the batch of images.
    """
    outputs = model(images, output_hidden_states=True)
    return outputs.last_hidden_state[:, 0, :]  # CLS token

def extract_cls_and_patches(model, images):
    """
    Extract CLS and patch tokens from the DiNO model.

    Args:
        model (torch.nn.Module): Loaded DiNO model.
        images (torch.Tensor): Batch of images.

    Returns:
        torch.Tensor: CLS tokens for the batch of images.
        torch.Tensor: Patch embeddings for the batch of images.
    """
    outputs = model(images, output_hidden_states=True)
    last_hidden_state = outputs.last_hidden_state
    cls_token = last_hidden_state[:, 0, :]  # CLS token
    patch_tokens = last_hidden_state[:, 1:, :]  # Patch embeddings
    return cls_token, patch_tokens