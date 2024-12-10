import os
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from data.dataset_loader import get_data_loader
from models.autoencoder import Autoencoder
from models.dino_model import load_dino_model, extract_cls_token

def visualize_latent_space(autoencoder, dino, dataloader, method="tsne", output_path="results/plots/latent_space.png"):
    """
    Visualize the learned latent space.
    
    Args:
        autoencoder (Autoencoder): Trained autoencoder model.
        dino (torch.nn.Module): DiNO model.
        dataloader (DataLoader): DataLoader for the dataset.
        method (str): Dimensionality reduction method ("tsne" or "pca").
        output_path (str): Path to save the visualization.
    """
    autoencoder.eval()
    dino.eval()
    latent_vectors = []
    labels = []

    with torch.no_grad():
        for images, texts in dataloader:
            cls_tokens = extract_cls_token(dino, images)
            _, latent = autoencoder(cls_tokens)
            latent_vectors.append(latent.cpu().numpy())
            labels.extend(texts)

    # Flatten latent vectors
    latent_vectors = torch.tensor(latent_vectors).reshape(-1, latent.shape[-1])

    # Dimensionality reduction
    if method == "tsne":
        reduced = TSNE(n_components=2).fit_transform(latent_vectors)
    elif method == "pca":
        reduced = PCA(n_components=2).fit_transform(latent_vectors)
    else:
        raise ValueError("Invalid method. Use 'tsne' or 'pca'.")

    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced[:, 0], reduced[:, 1], c="blue", alpha=0.6)
    plt.title(f"Latent Space Visualization ({method.upper()})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.savefig(output_path)
    print(f"Latent space visualization saved to {output_path}")

if __name__ == "__main__":
    autoencoder = Autoencoder(input_dim=1024, latent_dim=768)
    autoencoder.load_state_dict(torch.load("results/checkpoints/autoencoder.pth"))
    dino = load_dino_model("facebook/dino-v2-large")

    dataloader = get_data_loader("data/processed", batch_size=32)
    visualize_latent_space(autoencoder, dino, dataloader)
