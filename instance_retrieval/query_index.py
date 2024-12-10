import os
import faiss
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from models.dino_model import load_dino_model
from models.autoencoder import Autoencoder
from utils.helpers import load_checkpoint

def query_index(
    query_image_path, features_file, index_file, top_k=5, model_type="dino",
    dino_model_name="facebook/dino-v2-large", autoencoder_path=None, device="cuda"
):
    """
    Query the FAISS index with a given image.

    Args:
        query_image_path (str): Path to the query image.
        features_file (str): Path to the .npz file with features and image paths.
        index_file (str): Path to the FAISS index file.
        top_k (int): Number of nearest neighbors to return.
        model_type (str): Model type ("dino" or "custom").
        dino_model_name (str): Pre-trained DiNO model name.
        autoencoder_path (str): Path to the trained autoencoder checkpoint (if using custom model).
        device (str): Device to run the model on ("cuda" or "cpu").
    """
    # Load DiNO
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

    # Process query image
    query_image = Image.open(query_image_path).convert("RGB")
    query_image = transform(query_image).unsqueeze(0).to(device)

    # Extract query features
    with torch.no_grad():
        query_feature = dino(query_image).last_hidden_state[:, 0, :]
        if autoencoder:
            _, query_feature = autoencoder(query_feature)
        query_feature = query_feature.cpu().numpy()

    # Load FAISS index
    index = faiss.read_index(index_file)
    distances, indices = index.search(query_feature, top_k)

    # Retrieve corresponding image paths
    data = np.load(features_file)
    image_paths = data["image_paths"]

    print(f"Query results for {query_image_path}:")
    for i in range(top_k):
        print(f"{i + 1}: {image_paths[indices[0][i]]} (Distance: {distances[0][i]:.4f})")

if __name__ == "__main__":
    # Example usage
    query_index(
        query_image_path="data/query/query_image.jpg",
        features_file="instance_retrieval/features.npz",
        index_file="instance_retrieval/faiss_index.idx",
        top_k=5,
        model_type="custom",  # Use "dino" or "custom"
        autoencoder_path="results/checkpoints/autoencoder.pth"
    )
