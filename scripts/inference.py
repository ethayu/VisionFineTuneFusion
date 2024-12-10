import torch
from PIL import Image
from models.autoencoder import Autoencoder
from models.dino_model import load_dino_model, extract_cls_token
from models.clip_model import load_clip_model, get_clip_embeddings

def run_inference(autoencoder, dino, clip, processor, image_path, text):
    """
    Perform inference on an image-text pair.
    
    Args:
        autoencoder (Autoencoder): Trained autoencoder.
        dino (torch.nn.Module): DiNO model.
        clip (torch.nn.Module): CLIP model.
        processor (CLIPProcessor): CLIP processor.
        image_path (str): Path to the image.
        text (str): Input text.

    Returns:
        dict: Inference results including latent vector and CLIP similarities.
    """
    autoencoder.eval()
    dino.eval()
    clip.eval()

    # Process image
    image = Image.open(image_path).convert("RGB")
    image = processor(images=[image], return_tensors="pt")["pixel_values"]

    # Extract CLS token and latent space
    with torch.no_grad():
        cls_token = extract_cls_token(dino, image)
        _, latent = autoencoder(cls_token)

    # Compute CLIP embeddings
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    clip_outputs = clip(**inputs)
    image_embeds, text_embeds = clip_outputs.image_embeds, clip_outputs.text_embeds

    # Compute similarities
    image_similarity = torch.cosine_similarity(latent, image_embeds, dim=-1).item()
    text_similarity = torch.cosine_similarity(latent, text_embeds, dim=-1).item()

    return {
        "latent_vector": latent.cpu().numpy(),
        "image_similarity": image_similarity,
        "text_similarity": text_similarity,
    }

if __name__ == "__main__":
    autoencoder = Autoencoder(input_dim=1024, latent_dim=768)
    autoencoder.load_state_dict(torch.load("results/checkpoints/autoencoder.pth"))
    dino = load_dino_model("facebook/dino-v2-large")
    clip, processor = load_clip_model("openai/clip-vit-base-patch32")

    result = run_inference(autoencoder, dino, clip, processor, "sample.jpg", "A description of the sample image.")
    print(result)
