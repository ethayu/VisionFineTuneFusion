import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from tqdm import tqdm  # Added for progress tracking
from pathlib import Path  # Added for better path handling

def extract_features(
    image_dir,
    output_file,
    model_type="dino",
    dino_model_name="facebook/dino-v2-large",
    autoencoder_path=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    batch_size=32  # Added batch processing
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
        batch_size (int): Number of images to process at once.
    """
    # Create output directory if it doesn't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Validate inputs
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    if model_type == "custom" and (not autoencoder_path or not os.path.exists(autoencoder_path)):
        raise ValueError("Valid autoencoder path must be provided for custom model.")

    print(f"Loading DINO model: {dino_model_name}")
    dino = load_dino_model(dino_model_name).to(device).eval()

    if model_type == "custom":
        print(f"Loading autoencoder from: {autoencoder_path}")
        autoencoder = Autoencoder(input_dim=1024, latent_dim=768).to(device)
        load_checkpoint(autoencoder, autoencoder_path, device)
        autoencoder.eval()
    else:
        autoencoder = None

    transform = transforms.Compose([
        transforms.Resize(256),  # Slightly larger size for better quality
        transforms.CenterCrop(224),  # Consistent center crop
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
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
            batch_features = dino(batch_tensor).last_hidden_state[:, 0, :]
            
            if autoencoder:
                _, batch_features = autoencoder(batch_features)
            
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
    parser.add_argument("--model_type", default="dino", choices=["dino", "custom"], help="Model type")
    parser.add_argument("--autoencoder_path", help="Path to autoencoder checkpoint")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    
    args = parser.parse_args()
    
    extract_features(
        image_dir=args.image_dir,
        output_file=args.output_file,
        model_type=args.model_type,
        autoencoder_path=args.autoencoder_path,
        batch_size=args.batch_size
    )