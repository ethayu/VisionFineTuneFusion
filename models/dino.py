import torch
from transformers import DINOModel

def load_dino_model(model_name="facebook/dino-v2-large"):
    """
    Load the DiNOv2 model.

    Args:
        model_name (str): Name of the pre-trained DiNO model.

    Returns:
        torch.nn.Module: Loaded DiNO model.
    """
    model = DINOModel.from_pretrained(model_name)
    return model

def extract_cls_token(model, images):
    """
    Extract the class (cls) token from the DiNO model.

    Args:
        model (torch.nn.Module): Loaded DiNO model.
        images (torch.Tensor): Batch of images.

    Returns:
        torch.Tensor: CLS tokens for the batch of images.
    """
    outputs = model(images)
    return outputs.last_hidden_state[:, 0, :]  # CLS token

def extract_cls_and_patches(dino_model, images):
    """
    Extract CLS and patch tokens from the DiNO model.

    Args:
        dino_model (torch.nn.Module): Loaded DiNO model.
        images (torch.Tensor): Batch of images.

    Returns:
        torch.Tensor: CLS tokens for the batch of images.
        torch.Tensor: Patch embeddings for the batch of images.
    """
    outputs = dino_model(images)  # Assuming this returns token embeddings
    cls_token = outputs[:, 0, :]  # CLS token
    patch_tokens = outputs[:, 1:, :]  # Patch embeddings
    return cls_token, patch_tokens
