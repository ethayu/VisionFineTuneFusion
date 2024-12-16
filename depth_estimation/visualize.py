import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pathlib import Path
import argparse
import logging
from typing import Optional, Tuple
import sys
import os

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from model import DepthEstimationModel
from models import load_dino_model, CLSAutoencoder, PatchAutoencoder

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
    
    model = DepthEstimationModel(
        model_type=model_type,
        dino_model_name=dino_model_name,
        cls_autoencoder_path=cls_autoencoder_path,
        patch_autoencoder_path=patch_autoencoder_path,
        device=device
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model

def predict_depth(
    model: nn.Module,
    image_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Predict depth map for a single image."""
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        depth_map = model(image_tensor)
    
    return image_tensor, depth_map

def load_ground_truth_depth(depth_path: str) -> torch.Tensor:
    """Load and preprocess ground truth depth map."""
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    depth_map = Image.open(depth_path)
    depth_tensor = transform(depth_map)
    
    # Normalize to [0, 1]
    if depth_tensor.max() > 0:
        depth_tensor = depth_tensor / depth_tensor.max()
    
    return depth_tensor

def get_depth_path_from_image(image_path: str, depth_dir: str) -> str:
    """Get corresponding depth map path from image path."""
    image_name = os.path.basename(image_path)
    depth_name = image_name.replace('.jpg', '.png')
    return os.path.join(depth_dir, depth_name)

def visualize_depth_comparison(
    image_tensor: torch.Tensor,
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    output_path: Optional[str] = None,
    show_plot: bool = True
):
    """Visualize input image, predicted depth, and ground truth depth."""
    
    # Convert tensors to numpy arrays
    image = image_tensor.squeeze().cpu().numpy()
    pred_depth = pred_depth.squeeze().cpu().numpy()
    gt_depth = gt_depth.squeeze().cpu().numpy()
    
    # Denormalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std.reshape(3, 1, 1) * image + mean.reshape(3, 1, 1)
    image = np.clip(image, 0, 1)
    
    # Create figure
    plt.figure(figsize=(15, 5))
    
    # Plot original image
    plt.subplot(131)
    plt.imshow(np.transpose(image, (1, 2, 0)))
    plt.title('Input Image')
    plt.axis('off')
    
    # Plot predicted depth map
    plt.subplot(132)
    plt.imshow(pred_depth, cmap='gray')
    plt.colorbar(label='Depth')
    plt.title('Predicted Depth Map')
    plt.axis('off')
    
    # Plot ground truth depth map
    plt.subplot(133)
    plt.imshow(gt_depth, cmap='gray')
    plt.colorbar(label='Depth')
    plt.title('Ground Truth Depth Map')
    plt.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {output_path}")
    
    if show_plot:
        plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize depth estimation predictions")
    parser.add_argument("--image_path", required=True, help="Path to input image")
    parser.add_argument("--depth_dir", required=True, help="Directory containing ground truth depth maps")
    parser.add_argument("--checkpoint_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--model_type", default="custom_patch",
                       choices=["dino", "custom_cls", "custom_patch"])
    parser.add_argument("--cls_autoencoder_path", help="Path to CLS autoencoder checkpoint")
    parser.add_argument("--patch_autoencoder_path", help="Path to patch autoencoder checkpoint")
    parser.add_argument("--output_path", help="Path to save visualization")
    parser.add_argument("--no_show", action="store_true", help="Don't display the plot")
    
    args = parser.parse_args()
    
    try:
        # Load model
        model = load_model(
            checkpoint_path=args.checkpoint_path,
            model_type=args.model_type,
            cls_autoencoder_path=args.cls_autoencoder_path,
            patch_autoencoder_path=args.patch_autoencoder_path
        )
        
        # Predict depth
        image_tensor, pred_depth = predict_depth(model, args.image_path)
        
        # Load ground truth depth
        depth_path = get_depth_path_from_image(args.image_path, args.depth_dir)
        gt_depth = load_ground_truth_depth(depth_path)
        
        # Visualize results
        visualize_depth_comparison(
            image_tensor,
            pred_depth,
            gt_depth,
            output_path=args.output_path,
            show_plot=not args.no_show
        )
        
    except Exception as e:
        logger.error(f"Error during visualization: {e}")
        raise

if __name__ == "__main__":
    main()
