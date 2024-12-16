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
from models import load_dino_model, CLSAutoencoder, PatchAutoencoder
from utils import load_checkpoint, compute_cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_models(
    model_type: str,
    dino_model_name: str,
    cls_autoencoder_path: str = None,
    patch_autoencoder_path: str = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    """
    Load the required models based on model type.
    
    Args:
        model_type: Type of model to use ("dino", "custom_cls", or "custom_patch")
        dino_model_name: Name of the DINO model to load
        cls_autoencoder_path: Path to CLS autoencoder checkpoint
        patch_autoencoder_path: Path to patch autoencoder checkpoint
        device: Device to load models on
        
    Returns:
        Tuple of (dino_model, cls_autoencoder, patch_autoencoder)
    """
    logger.info(f"Loading DINO model: {dino_model_name}")
    dino = load_dino_model(dino_model_name).to(device).eval()

    cls_autoencoder = None
    patch_autoencoder = None

    if model_type == "custom_cls":
        if not cls_autoencoder_path or not os.path.exists(cls_autoencoder_path):
            raise ValueError("Valid CLS autoencoder path must be provided for custom_cls model")
        logger.info(f"Loading CLS autoencoder from: {cls_autoencoder_path}")
        cls_autoencoder = CLSAutoencoder(input_dim=1024, latent_dim=512).to(device)
        state_dict = torch.load(cls_autoencoder_path, map_location=device)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        cls_autoencoder.load_state_dict(state_dict)
        cls_autoencoder.eval()
    
    elif model_type == "custom_patch":
        if not patch_autoencoder_path or not os.path.exists(patch_autoencoder_path):
            raise ValueError("Valid patch autoencoder path must be provided for custom_patch model")
        logger.info(f"Loading patch autoencoder from: {patch_autoencoder_path}")
        patch_autoencoder = PatchAutoencoder(input_dim=1024, latent_dim=512).to(device)
        state_dict = torch.load(patch_autoencoder_path, map_location=device)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        patch_autoencoder.load_state_dict(state_dict)
        patch_autoencoder.eval()

    return dino, cls_autoencoder, patch_autoencoder

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
    cls_autoencoder: torch.nn.Module = None,
    patch_autoencoder: torch.nn.Module = None,
    model_type: str = "dino",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> np.ndarray:
    """
    Extract features from a query image.
    
    Args:
        image_path: Path to the query image
        dino: DINO model
        cls_autoencoder: Optional CLS autoencoder model
        patch_autoencoder: Optional patch autoencoder model
        model_type: Model type to use
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
        outputs = dino(query_image)
        
        if model_type == "custom_cls" and cls_autoencoder:
            cls_features = outputs.last_hidden_state[:, 0, :]
            _, query_feature = cls_autoencoder(cls_features)
        
        elif model_type == "custom_patch" and patch_autoencoder:
            patch_features = outputs.last_hidden_state[:, 1:, :]
            patch_reconstructed = []
            for patches in patch_features:
                patch_latents = []
                for patch in patches:
                    _, latent = patch_autoencoder(patch.unsqueeze(0))
                    patch_latents.append(latent)
                patch_reconstructed.append(torch.mean(torch.cat(patch_latents), dim=0))
            query_feature = torch.stack(patch_reconstructed)
        
        else:  # dino
            query_feature = outputs.last_hidden_state[:, 0, :]
        
        return query_feature.cpu().numpy()

def query_index(
    query_image_path: str,
    features_file: str,
    index_file: str,
    top_k: int = 5,
    model_type: str = "dino",
    dino_model_name: str = "facebook/dinov2-large",
    cls_autoencoder_path: str = None,
    patch_autoencoder_path: str = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> List[Tuple[str, float]]:
    """
    Query the FAISS index with a given image.
    
    Args:
        query_image_path: Path to the query image
        features_file: Path to the .npz file with features and image paths
        index_file: Path to the FAISS index file
        top_k: Number of nearest neighbors to return
        model_type: Model type ("dino", "custom_cls", or "custom_patch")
        dino_model_name: Pre-trained DINO model name
        cls_autoencoder_path: Path to the CLS autoencoder checkpoint
        patch_autoencoder_path: Path to the patch autoencoder checkpoint
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
    dino, cls_autoencoder, patch_autoencoder = load_models(
        model_type, dino_model_name, cls_autoencoder_path, patch_autoencoder_path, device
    )

    # Extract query features
    query_feature = extract_query_features(
        query_image_path, dino, cls_autoencoder, patch_autoencoder, model_type, device
    )

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

def get_image_label(image_path: str, labels_dir: str) -> str:
    """
    Get label for an image by reading its corresponding label file.
    
    Args:
        image_path: Path to image file (e.g., 'image_00001.jpg')
        labels_dir: Directory containing label files
    """
    # Extract image number and create label filename
    image_num = int(image_path.split('_')[-1].split('.')[0])
    label_filename = f"image_{image_num-1:05d}.txt"
    label_path = os.path.join(labels_dir, label_filename)
    
    try:
        with open(label_path, 'r') as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Error reading label file {label_path}: {e}")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Query image retrieval index")
    parser.add_argument("--query_image", required=True, help="Path to query image")
    parser.add_argument("--features_file", required=True, help="Path to features file")
    parser.add_argument("--index_file", required=True, help="Path to FAISS index")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--model_type", default="dino", 
                       choices=["dino", "custom_cls", "custom_patch"], help="Model type")
    parser.add_argument("--cls_autoencoder_path", help="Path to CLS autoencoder checkpoint")
    parser.add_argument("--patch_autoencoder_path", help="Path to patch autoencoder checkpoint")
    parser.add_argument("--labels_dir", required=True, help="Directory containing image labels")
    
    args = parser.parse_args()
    
    try:
        results = query_index(
            query_image_path=args.query_image,
            features_file=args.features_file,
            index_file=args.index_file,
            top_k=args.top_k,
            model_type=args.model_type,
            cls_autoencoder_path=args.cls_autoencoder_path,
            patch_autoencoder_path=args.patch_autoencoder_path
        )
        
        # Get query image label
        query_label = get_image_label(os.path.basename(args.query_image), args.labels_dir)
        print(f"\nQuery image: {args.query_image} (Class: {query_label})")
        print(f"\nTop {len(results)} matches:")
        
        for i, (path, distance) in enumerate(results, 1):
            retrieved_label = get_image_label(os.path.basename(path), args.labels_dir)
            print(f"{i}. {path} (Class: {retrieved_label}, Distance: {distance:.4f})")
            
    except Exception as e:
        logger.error(f"Error during query: {e}")
        raise
