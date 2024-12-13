import os
import faiss
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from pathlib import Path
from typing import List, Tuple
import logging

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Standard library imports
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Project imports
from models import load_dino_model, Autoencoder  # Import from models/__init__.py
from utils import load_checkpoint, compute_cosine_similarity  # Import from utils/__init__.py



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_models(
    model_type: str,
    dino_model_name: str,
    autoencoder_path: str = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """
    Load the required models based on model type.
    
    Args:
        model_type: Type of model to use ("dino" or "custom")
        dino_model_name: Name of the DINO model to load
        autoencoder_path: Path to autoencoder checkpoint
        device: Device to load models on
        
    Returns:
        Tuple of (dino_model, autoencoder)
    """
    logger.info(f"Loading DINO model: {dino_model_name}")
    dino = load_dino_model(dino_model_name).to(device).eval()

    if model_type == "custom":
        if not autoencoder_path or not os.path.exists(autoencoder_path):
            raise ValueError("Valid autoencoder path must be provided for custom model")
        logger.info(f"Loading autoencoder from: {autoencoder_path}")
        autoencoder = Autoencoder(input_dim=1024, latent_dim=768).to(device)
        load_checkpoint(autoencoder, autoencoder_path, device)
        autoencoder.eval()
    else:
        autoencoder = None

    return dino, autoencoder

def get_image_transform() -> transforms.Compose:
    """Create the image transformation pipeline."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def extract_query_features(
    image_path: str,
    dino: torch.nn.Module,
    autoencoder: torch.nn.Module = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> np.ndarray:
    """
    Extract features from a query image.
    
    Args:
        image_path: Path to the query image
        dino: DINO model
        autoencoder: Optional autoencoder model
        device: Device to run inference on
        
    Returns:
        numpy array of extracted features
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Query image not found: {image_path}")

    transform = get_image_transform()
    
    try:
        query_image = Image.open(image_path).convert("RGB")
        query_image = transform(query_image).unsqueeze(0).to(device)
    except Exception as e:
        raise ValueError(f"Error processing query image: {e}")

    with torch.no_grad():
        query_feature = dino(query_image).last_hidden_state[:, 0, :]
        if autoencoder:
            _, query_feature = autoencoder(query_feature)
        return query_feature.cpu().numpy()

def query_index(
    query_image_path: str,
    features_file: str,
    index_file: str,
    top_k: int = 5,
    model_type: str = "dino",
    dino_model_name: str = "facebook/dinov2-large",
    autoencoder_path: str = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> List[Tuple[str, float]]:
    """
    Query the FAISS index with a given image.
    
    Args:
        query_image_path: Path to the query image
        features_file: Path to the .npz file with features and image paths
        index_file: Path to the FAISS index file
        top_k: Number of nearest neighbors to return
        model_type: Model type ("dino" or "custom")
        dino_model_name: Pre-trained DINO model name
        autoencoder_path: Path to the trained autoencoder checkpoint
        device: Device to run the model on
        
    Returns:
        List of tuples containing (image_path, distance) for top_k matches
    """
    # Validate inputs
    if not os.path.exists(features_file):
        raise FileNotFoundError(f"Features file not found: {features_file}")
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"Index file not found: {index_file}")

    # Load models
    dino, autoencoder = load_models(model_type, dino_model_name, autoencoder_path, device)

    # Extract query features
    query_feature = extract_query_features(query_image_path, dino, autoencoder, device)

    # Load FAISS index
    logger.info("Loading FAISS index...")
    index = faiss.read_index(index_file)
    
    # Load feature metadata
    logger.info("Loading feature metadata...")
    data = np.load(features_file)
    image_paths = data["image_paths"]
    
    # Verify index and features compatibility
    if len(image_paths) != index.ntotal:
        raise ValueError(f"Mismatch between index size ({index.ntotal}) and number of image paths ({len(image_paths)})")

    # Perform search
    logger.info("Searching index...")
    distances, indices = index.search(query_feature, min(top_k, len(image_paths)))
    
    # Format results
    results = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        distance = distances[0][i]
        if idx < len(image_paths):  # Safety check
            results.append((image_paths[idx], distance))
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Query image retrieval index")
    parser.add_argument("--query_image", required=True, help="Path to query image")
    parser.add_argument("--features_file", required=True, help="Path to features file")
    parser.add_argument("--index_file", required=True, help="Path to FAISS index")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--model_type", default="dino", choices=["dino", "custom"], help="Model type")
    parser.add_argument("--autoencoder_path", help="Path to autoencoder checkpoint")
    
    args = parser.parse_args()
    
    try:
        results = query_index(
            query_image_path=args.query_image,
            features_file=args.features_file,
            index_file=args.index_file,
            top_k=args.top_k,
            model_type=args.model_type,
            autoencoder_path=args.autoencoder_path
        )
        
        print(f"\nTop {len(results)} matches for {args.query_image}:")
        for i, (path, distance) in enumerate(results, 1):
            print(f"{i}. {path} (Distance: {distance:.4f})")
            
    except Exception as e:
        logger.error(f"Error during query: {e}")
        raise