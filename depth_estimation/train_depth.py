import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from depth_estimation.model import DepthEstimationModel
from torchvision import transforms
from tqdm import tqdm
import os
from PIL import Image
import logging
from pathlib import Path
from typing import Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DepthDataset(Dataset):
    """Dataset for depth estimation training."""
    
    def __init__(
        self,
        image_dir: str,
        depth_dir: str,
        transform: Optional[transforms.Compose] = None,
        target_transform: Optional[transforms.Compose] = None
    ):
        self.image_dir = Path(image_dir)
        self.depth_dir = Path(depth_dir)
        
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not self.depth_dir.exists():
            raise FileNotFoundError(f"Depth directory not found: {depth_dir}")
        
        # Get matching files only
        self.image_files = sorted(self.image_dir.glob("*.jpg"))
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
        self.target_transform = target_transform
    
    def __len__(self) -> int:
        return len(self.depth_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            image = Image.open(self.image_files[idx]).convert("RGB")
            depth = Image.open(self.depth_files[idx])
            
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                depth = self.target_transform(depth)
            
            return image, depth
        
        except Exception as e:
            logger.error(f"Error loading item {idx}: {e}")
            raise

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
        
        for images, depth_maps in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = images.to(device)
            depth_maps = depth_maps.to(device)
            
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, depth_maps)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, depth_maps in val_loader:
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
            }, checkpoint_path)
            logger.info(f"Saved best model to {checkpoint_path}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping after {epoch+1} epochs")
            break

if __name__ == "__main__":
    import argparse
    
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
    parser.add_argument("--cls_autoencoder_path")
    parser.add_argument("--patch_autoencoder_path")
    
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
    
    # Create dataset
    dataset = DepthDataset(
        image_dir=args.image_dir,
        depth_dir=args.depth_dir,
        transform=transform,
        target_transform=target_transform
    )
    
    # Split dataset
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset)