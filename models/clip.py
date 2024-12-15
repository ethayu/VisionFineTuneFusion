import torch
from transformers import CLIPProcessor, CLIPModel

def load_clip_model(model_name="openai/clip-vit-base-patch32"):
    """
    Load the CLIP model.

    Args:
        model_name (str): Name of the pre-trained CLIP model.

    Returns:
        CLIPModel: Loaded CLIP model.
        CLIPProcessor: Preprocessor for image-text pairs.
    """
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor

def get_clip_embeddings(model, processor, images, texts):
    """
    Get image and text embeddings from the CLIP model.

    Args:
        model (CLIPModel): Loaded CLIP model.
        processor (CLIPProcessor): Preprocessor for images and text.
        images (list): List of PIL images.
        texts (list): List of text descriptions.

    Returns:
        torch.Tensor, torch.Tensor: Image embeddings, text embeddings.
    """
    if texts:
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        return outputs.image_embeds, outputs.text_embeds
    else:
        inputs = processor(images=images, return_tensors="pt", padding=True)
        outputs = model.vision_model(**inputs)
        return outputs.last_hidden_state
