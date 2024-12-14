import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import DepthEstimationModel
from torchvision import transforms
from tqdm import tqdm
import os
from PIL import Image
import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DepthDataset(Dataset):
    """Dataset for depth estimation training."""
    
    def __init__(
        self,
        image_dir: str,
        depth_dir: str,
        transform: Optional[transforms.Compose] = None
    ):
        self.image_dir = Path(image_dir)
        self.depth_dir = Path(depth_dir)
        
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not self.depth_dir.exists():
            raise FileNotFoundError(f"Depth directory not found: {depth_dir}")
        
        # Get matching files only
        self.image_files = sorted(list(self.image_dir.glob("*.jpg")))
        self.depth_files = []
        
        for img_file in self.image_files:
            depth_file = self.depth_dir / f"{img_file.stem}.png"
            if not depth_file.exists():
                logger.warning(f"No matching depth map for {img_file}")
                continue
            self.depth_files.append(depth_file)
        
        if not self.depth_files:
            raise RuntimeError("No valid image-depth pairs found")
        
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.depth_files)
    
    def process_depth_map(self, depth_map: Image.Image) -> torch.Tensor:
        # Convert to numpy array
        depth_array = np.array(depth_map)
        
        # Convert to float32
        depth_array = depth_array.astype(np.float32)
        
        # Normalize to [0, 1]
        if depth_array.max() > 0:
            depth_array = depth_array / depth_array.max()
        
        # Resize
        depth_pil = Image.fromarray(depth_array)
        depth_pil = depth_pil.resize((224, 224), Image.Resampling.BILINEAR)
        
        # Convert to tensor
        depth_tensor = torch.from_numpy(np.array(depth_pil)).float()
        
        # Add channel dimension if needed
        if depth_tensor.dim() == 2:
            depth_tensor = depth_tensor.unsqueeze(0)
        
        return depth_tensor
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            # Load and process image
            image = Image.open(self.image_files[idx]).convert("RGB")
            if self.transform:
                image = self.transform(image)
            
            # Load and process depth map
            depth = Image.open(self.depth_files[idx])
            depth = self.process_depth_map(depth)
            
            return image, depth
        
        except Exception as e:
            logger.error(f"Error loading item {idx}: {e}")
            # Return a default tensor pair in case of error
            return torch.zeros((3, 224, 224)), torch.zeros((1, 224, 224))

def train_depth_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    epochs: int,
    device: str,
    lr: float = 0.0001,
    checkpoint_dir: str = "depth_estimation/checkpoints",
    patience: int = 5
) -> None:
    """Train the depth estimation model."""
    
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for images, depth_maps in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            images = images.to(device)
            depth_maps = depth_maps.to(device)
            
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, depth_maps)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, depth_maps in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                images = images.to(device)
                depth_maps = depth_maps.to(device)
                
                predictions = model(images)
                loss = criterion(predictions, depth_maps)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        logger.info(f"Epoch {epoch+1}")
        logger.info(f"Train Loss: {avg_train_loss:.4f}")
        logger.info(f"Val Loss: {avg_val_loss:.4f}")
        logger.info(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            checkpoint_path = Path(checkpoint_dir) / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'train_loss': avg_train_loss
            }, checkpoint_path)
            logger.info(f"Saved best model to {checkpoint_path}")
        else:
            patience_counter += 1
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch+1}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                    'train_loss': avg_train_loss
                }, checkpoint_path)
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping after {epoch+1} epochs")
            break

if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Train depth estimation model")
    parser.add_argument("--image_dir", required=True, help="Path to image directory")
    parser.add_argument("--depth_dir", required=True, help="Path to depth maps directory")
    parser.add_argument("--model_type", default="custom_patch", 
                       choices=["dino", "custom_cls", "custom_patch"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint_dir", default="depth_estimation/checkpoints")
    parser.add_argument("--cls_autoencoder_path", help="Path to CLS token autoencoder weights")
    parser.add_argument("--patch_autoencoder_path", help="Path to patch autoencoder weights")
    parser.add_argument("--val_split", type=float, default=0.1, 
                       help="Validation set split ratio (default: 0.1)")
    
    args = parser.parse_args()
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    try:
        dataset = DepthDataset(
            image_dir=args.image_dir,
            depth_dir=args.depth_dir,
            transform=transform
        )
    except Exception as e:
        logger.error(f"Failed to create dataset: {e}")
        sys.exit(1)
    
    # Split dataset
    val_size = int(args.val_split * len(dataset))
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if args.device == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if args.device == "cuda" else False
    )
    
    # Initialize model
    try:
        model = DepthEstimationModel(
            model_type=args.model_type,
            cls_autoencoder_path=args.cls_autoencoder_path,
            patch_autoencoder_path=args.patch_autoencoder_path,
            device=args.device
        )
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        sys.exit(1)
    
    # Train model
    try:
        train_depth_model(
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            epochs=args.epochs,
            device=args.device,
            lr=args.lr,
            checkpoint_dir=args.checkpoint_dir
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)