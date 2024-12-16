import os
import shutil
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

def clean_directory(directory):
    """Remove and recreate a directory"""
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

def extract_and_save_dataset(mat_file_path, output_dir="data/depth"):
    """
    Extracts images and depth maps from .mat file and saves them to specified directories.
    """
    # Clean and create directory structure
    images_dir = os.path.join(output_dir, "images")
    depth_dir = os.path.join(output_dir, "depth_maps")
    
    clean_directory(images_dir)
    clean_directory(depth_dir)
    
    print(f"Extracting data from {mat_file_path}...")
    try:
        with h5py.File(mat_file_path, 'r') as f:
            num_samples = f['images'].shape[0]
            
            for i in tqdm(range(num_samples), desc="Extracting"):
                # Extract and process image
                image = np.array(f['images'][i])
                image = np.transpose(image, (2, 1, 0))  # Fixed transpose order
                image = Image.fromarray(image.astype('uint8'))
                image_path = os.path.join(images_dir, f"{i:05d}.jpg")  # Changed to .jpg
                image.save(image_path, 'JPEG')  # Explicitly save as JPEG
                
                # Extract and process depth map
                depth = np.array(f['depths'][i])
                depth = np.transpose(depth, (1, 0))  # Fixed transpose order
                
                # Save depth map as uint16 PNG
                depth_normalized = ((depth - np.min(depth)) * (65535.0 / (np.max(depth) - np.min(depth)))).astype('uint16')
                depth_img = Image.fromarray(depth_normalized)
                depth_path = os.path.join(depth_dir, f"{i:05d}.png")
                depth_img.save(depth_path)
        
        print(f"Extraction complete. Saved {num_samples} pairs.")
        return True
    except Exception as e:
        print(f"Error during extraction: {str(e)}")
        return False

class NYUDepthDataset(Dataset):
    def __init__(self, data_dir="data/depth", transform=None, depth_transform=None):
        """
        Dataset for loading extracted NYU Depth images and depth maps.
        """
        self.images_dir = os.path.join(data_dir, "images")
        self.depth_dir = os.path.join(data_dir, "depth_maps")
        
        # Verify directories exist
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Images directory not found at {self.images_dir}")
        if not os.path.exists(self.depth_dir):
            raise FileNotFoundError(f"Depth maps directory not found at {self.depth_dir}")
        
        # Get sorted list of image files - changed to .jpg
        self.image_files = sorted([f for f in os.listdir(self.images_dir) if f.endswith('.jpg')])
        self.depth_files = sorted([f for f in os.listdir(self.depth_dir) if f.endswith('.png')])
        
        if not self.image_files or not self.depth_files:
            raise RuntimeError("No image-depth pairs found")
            
        print(f"Found {len(self.image_files)} image-depth pairs")
        
        self.transform = transform
        self.depth_transform = depth_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')
        
        # Load depth map
        depth_path = os.path.join(self.depth_dir, self.depth_files[idx])
        depth = Image.open(depth_path)
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        if self.depth_transform:
            depth = self.depth_transform(depth)
        
        return image, depth

def get_dataloader(data_dir="data/depth", batch_size=32, shuffle=True, num_workers=4):
    """
    Creates a DataLoader for the extracted NYU Depth dataset.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    depth_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = NYUDepthDataset(
        data_dir=data_dir,
        transform=transform,
        depth_transform=depth_transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader

if __name__ == "__main__":
    mat_file_path = "./data/nyu_depth_v2_labeled.mat"
    output_dir = "data/depth"
    
    # Force re-extraction of the dataset
    print("Extracting dataset from .mat file...")
    if extract_and_save_dataset(mat_file_path, output_dir):
        # Create and test the dataloader
        dataloader = get_dataloader(output_dir, batch_size=16)
        
        # Test the dataloader
        for batch_idx, (images, depths) in enumerate(dataloader):
            print(f"Batch {batch_idx + 1}:")
            print(f" - Images shape: {images.shape}")
            print(f" - Depths shape: {depths.shape}")
            break
    else:
        print("Dataset extraction failed. Please check the .mat file and try again.")
