import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from models.dino import load_dino_model
from models.autoencoder import Autoencoder
from utils.helpers import load_checkpoint

def extract_features(
    image_dir, output_file, model_type="dino", dino_model_name="facebook/dino-v2-large",
    autoencoder_path=None, device="cuda"
):
    """
    Extract features using DiNO or DiNO + Autoencoder.

    Args:
        image_dir (str): Directory containing the images.
        output_file (str): Path to save the extracted features.
        model_type (str): Model type ("dino" or "custom").
        dino_model_name (str): Pre-trained DiNO model name.
        autoencoder_path (str): Path to the trained autoencoder checkpoint (if using custom model).
        device (str): Device to run the model on ("cuda" or "cpu").
    """
    dino = load_dino_model(dino_model_name).to(device).eval()

    if model_type == "custom":
        assert autoencoder_path, "Autoencoder path must be provided for custom model."
        autoencoder = Autoencoder(input_dim=1024, latent_dim=768).to(device)
        load_checkpoint(autoencoder, autoencoder_path, device)
        autoencoder.eval()
    else:
        autoencoder = None

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5))
    ])

    features = []
    image_paths = []
    for img_name in os.listdir(image_dir):
        if img_name.endswith((".jpg", ".png")):
            img_path = os.path.join(image_dir, img_name)
            image = Image.open(img_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                cls_token = dino(image).last_hidden_state[:, 0, :]
                if autoencoder:
                    _, latent = autoencoder(cls_token)
                    features.append(latent.cpu().numpy())
                else:
                    features.append(cls_token.cpu().numpy())
                image_paths.append(img_path)

    features = np.vstack(features)
    np.savez(output_file, features=features, image_paths=image_paths)
    print(f"Features saved to {output_file}")

if __name__ == "__main__":
    # Example usage
    extract_features(
        image_dir="data/raw",
        output_file="instance_retrieval/features.npz",
        model_type="custom",  # Use "dino" or "custom"
        autoencoder_path="results/checkpoints/autoencoder.pth"
    )
