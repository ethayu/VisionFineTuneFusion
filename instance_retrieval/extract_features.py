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

def extract_features(
    image_dir,
    output_file,
    model_type="dino",
    dino_model_name="facebook/dinov2-large",
    cls_autoencoder_path=None,
    patch_autoencoder_path=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    batch_size=32
):
    """
    Extract features using DiNO, DiNO + CLS Autoencoder, or DiNO + Patch Autoencoder.
    
    Args:
        image_dir (str): Directory containing the images.
        output_file (str): Path to save the extracted features.
        model_type (str): Model type ("dino", "custom_cls", or "custom_patch").
        dino_model_name (str): Pre-trained DiNO model name.
        cls_autoencoder_path (str): Path to the CLS autoencoder checkpoint.
        patch_autoencoder_path (str): Path to the patch autoencoder checkpoint.
        device (str): Device to run the model on ("cuda" or "cpu").
        batch_size (int): Number of images to process at once.
    """
    # Create output directory if it doesn't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Validate inputs
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    if model_type == "custom_cls" and (not cls_autoencoder_path or not os.path.exists(cls_autoencoder_path)):
        raise ValueError("Valid CLS autoencoder path must be provided for custom_cls model.")
    
    if model_type == "custom_patch" and (not patch_autoencoder_path or not os.path.exists(patch_autoencoder_path)):
        raise ValueError("Valid patch autoencoder path must be provided for custom_patch model.")

    print(f"Loading DINO model: {dino_model_name}")
    dino = load_dino_model(dino_model_name).to(device).eval()

    cls_autoencoder = None
    patch_autoencoder = None

    if model_type == "custom_cls":
        print(f"Loading CLS autoencoder from: {cls_autoencoder_path}")
        cls_autoencoder = CLSAutoencoder(input_dim=1024, latent_dim=512).to(device)
        state_dict = torch.load(cls_autoencoder_path, map_location=device)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        cls_autoencoder.load_state_dict(state_dict)
        cls_autoencoder.eval()
    
    elif model_type == "custom_patch":
        print(f"Loading Patch autoencoder from: {patch_autoencoder_path}")
        patch_autoencoder = PatchAutoencoder(input_dim=1024, latent_dim=512).to(device)
        state_dict = torch.load(patch_autoencoder_path, map_location=device)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        patch_autoencoder.load_state_dict(state_dict)
        patch_autoencoder.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Get list of valid image files
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_paths = [
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in valid_extensions
    ]

    if not image_paths:
        raise ValueError(f"No valid images found in {image_dir}")

    features = []
    processed_paths = []
    
    # Process images in batches
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        valid_paths = []

        # Prepare batch
        for img_path in batch_paths:
            try:
                image = Image.open(img_path).convert("RGB")
                batch_images.append(transform(image))
                valid_paths.append(img_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

        if not batch_images:
            continue

        # Process batch
        with torch.no_grad():
            batch_tensor = torch.stack(batch_images).to(device)
            outputs = dino(batch_tensor)
            
            if model_type == "custom_cls":
                cls_features = outputs.last_hidden_state[:, 0, :]
                _, batch_features = cls_autoencoder(cls_features)
            
            elif model_type == "custom_patch":
                patch_features = outputs.last_hidden_state[:, 1:, :]
                patch_reconstructed = []
                for patches in patch_features:
                    patch_latents = []
                    for patch in patches:
                        _, latent = patch_autoencoder(patch.unsqueeze(0))
                        patch_latents.append(latent)
                    patch_reconstructed.append(torch.mean(torch.cat(patch_latents), dim=0))
                batch_features = torch.stack(patch_reconstructed)
            
            else:  # dino
                batch_features = outputs.last_hidden_state[:, 0, :]
            
            features.append(batch_features.cpu().numpy())
            processed_paths.extend(valid_paths)

    # Concatenate all features
    features = np.vstack(features)

    # Save features and paths
    print(f"Saving {len(processed_paths)} features to {output_file}")
    np.savez_compressed(
        output_file,
        features=features,
        image_paths=processed_paths,
        model_type=model_type,
        dino_model=dino_model_name
    )
    
    return features, processed_paths

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract features from images using DINO")
    parser.add_argument("--image_dir", required=True, help="Directory containing images")
    parser.add_argument("--output_file", required=True, help="Path to save features")
    parser.add_argument("--model_type", default="dino", choices=["dino", "custom_cls", "custom_patch"], help="Model type")
    parser.add_argument("--cls_autoencoder_path", help="Path to CLS autoencoder checkpoint")
    parser.add_argument("--patch_autoencoder_path", help="Path to patch autoencoder checkpoint")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    
    args = parser.parse_args()
    
    extract_features(
        image_dir=args.image_dir,
        output_file=args.output_file,
        model_type=args.model_type,
        cls_autoencoder_path=args.cls_autoencoder_path,
        patch_autoencoder_path=args.patch_autoencoder_path,
        batch_size=args.batch_size
    )
