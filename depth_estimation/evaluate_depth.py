import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import DepthEstimationModel
from train_depth import DepthDataset
from  metrics import evaluate_depth_metrics, log_metrics
from torchvision import transforms
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Dict, List
import json
from tqdm import tqdm
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
from models import load_dino_model, CLSAutoencoder, PatchAutoencoder  # Import from models/__init__.py
from utils import load_checkpoint, compute_cosine_similarity  # Import from utils/__init__.py

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(
    checkpoint_path: str,
    model_type: str = "custom_patch",
    dino_model_name: str = "facebook/dinov2-large",
    cls_autoencoder_path: Optional[str] = None,
    patch_autoencoder_path: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> nn.Module:
    """Load the trained depth estimation model."""
    
    # Initialize model
    model = DepthEstimationModel(
        model_type=model_type,
        dino_model_name=dino_model_name,
        cls_autoencoder_path=cls_autoencoder_path,
        patch_autoencoder_path=patch_autoencoder_path,
        device=device
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model

def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    save_predictions: bool = False,
    output_dir: Optional[str] = None
) -> Dict[str, float]:
    """
    Evaluate the depth estimation model.
    
    Args:
        model: Trained depth estimation model
        data_loader: DataLoader for evaluation
        device: Device to run evaluation on
        save_predictions: Whether to save predicted depth maps
        output_dir: Directory to save predictions if save_predictions is True
        
    Returns:
        Dictionary containing evaluation metrics
    """
    if save_predictions:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    all_metrics = []
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(tqdm(data_loader, desc="Evaluating")):
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            predictions = model(images)
            
            # Compute metrics for batch
            metrics = evaluate_depth_metrics(predictions, targets)
            all_metrics.append(metrics)
            
            # Save predictions if requested
            if save_predictions:
                for j, pred in enumerate(predictions):
                    pred_np = pred.cpu().numpy()
                    np.save(
                        Path(output_dir) / f"pred_{i*data_loader.batch_size + j}.npy",
                        pred_np
                    )
    
    # Average metrics across all batches
    final_metrics = {}
    for key in all_metrics[0].keys():
        final_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    return final_metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate depth estimation model")
    parser.add_argument("--image_dir", required=True, help="Path to image directory")
    parser.add_argument("--depth_dir", required=True, help="Path to depth maps directory")
    parser.add_argument("--checkpoint_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--model_type", default="custom_patch",
                       choices=["dino", "custom_cls", "custom_patch"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_predictions", action="store_true",
                       help="Save predicted depth maps")
    parser.add_argument("--output_dir", default="depth_estimation/predictions",
                       help="Directory to save predictions")
    parser.add_argument("--cls_autoencoder_path")
    parser.add_argument("--patch_autoencoder_path")
    parser.add_argument("--results_file", help="Path to save evaluation results")
    
    args = parser.parse_args()
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    target_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Create dataset and dataloader
    dataset = DepthDataset(
        image_dir=args.image_dir,
        depth_dir=args.depth_dir,
        transform=transform
    )
        
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Load model
    model = load_model(
        checkpoint_path=args.checkpoint_path,
        model_type=args.model_type,
        cls_autoencoder_path=args.cls_autoencoder_path,
        patch_autoencoder_path=args.patch_autoencoder_path,
        device=args.device
    )
    
    # Evaluate model
    metrics = evaluate_model(
        model,
        data_loader,
        args.device,
        args.save_predictions,
        args.output_dir
    )
    
    # Log metrics
    log_metrics(metrics)
    
    # Save results if requested
    if args.results_file:
        with open(args.results_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"\nResults saved to {args.results_file}")
