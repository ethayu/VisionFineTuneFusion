import os
import shutil
import h5py
import scipy.io
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import torch
from pathlib import Path

def clean_directory(directory):
    """Remove and recreate a directory"""
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

def extract_and_save_labels(mat_file_path, output_dir="data/flowers"):
    """
    Extracts labels from .mat file and saves them to specified directory.
    """
    # Create labels directory
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)
    
    print(f"Checking file existence at {mat_file_path}...")
    if not os.path.exists(mat_file_path):
        print(f"Error: File not found at {mat_file_path}")
        return False
    
    print(f"Attempting to extract labels from {mat_file_path}...")
    try:
        # Try loading with scipy.io first (for older .mat files)
        try:
            print("Attempting to load with scipy.io...")
            mat_contents = scipy.io.loadmat(mat_file_path)
            print(f"Available keys in mat file: {mat_contents.keys()}")
            labels = mat_contents['labels']
            
        except Exception as scipy_error:
            print(f"scipy.io failed: {scipy_error}")
            print("Attempting to load with h5py...")
            
            with h5py.File(mat_file_path, 'r') as f:
                print(f"Available keys in mat file: {list(f.keys())}")
                labels = np.array(f['labels'])
        
        # Ensure labels are in the correct shape
        if labels.ndim > 1:
            labels = np.squeeze(labels)
        
        # Save labels as numpy array
        labels_path = os.path.join(labels_dir, "labels.npy")
        np.save(labels_path, labels)
        
        # Save as text file for easy viewing
        labels_txt_path = os.path.join(labels_dir, "labels.txt")
        np.savetxt(labels_txt_path, labels, fmt='%d')
        
        print(f"Extraction complete. Saved {len(labels)} labels.")
        
        # Save basic statistics
        stats_path = os.path.join(labels_dir, "stats.txt")
        with open(stats_path, 'w') as stats_file:
            stats_file.write(f"Total labels: {len(labels)}\n")
            stats_file.write(f"Unique classes: {len(np.unique(labels))}\n")
            stats_file.write(f"Class distribution:\n")
            unique, counts = np.unique(labels, return_counts=True)
            for u, c in zip(unique, counts):
                stats_file.write(f"Class {u}: {c} samples\n")
        
        return True
        
    except Exception as e:
        print(f"Error during extraction: {str(e)}")
        print("Please ensure:")
        print("1. The .mat file exists and is not corrupted")
        print("2. The file contains a 'labels' key (or similar)")
        print("3. You have read permissions for the file")
        return False

class FlowerDataset(Dataset):
    def __init__(self, data_dir="data/flowers", transform=None):
        """
        Dataset for loading flower images and their labels.
        """
        self.images_dir = os.path.join(data_dir, "images")
        self.labels_path = os.path.join(data_dir, "labels", "labels.npy")
        
        # Verify directories and files exist
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Images directory not found at {self.images_dir}")
        if not os.path.exists(self.labels_path):
            raise FileNotFoundError(f"Labels file not found at {self.labels_path}")
        
        # Get sorted list of image files
        self.image_files = sorted([f for f in os.listdir(self.images_dir) 
                                 if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        if not self.image_files:
            raise RuntimeError("No images found")
        
        # Load labels
        self.labels = np.load(self.labels_path)
        
        if len(self.image_files) != len(self.labels):
            raise RuntimeError(f"Number of images ({len(self.image_files)}) "
                             f"doesn't match number of labels ({len(self.labels)})")
            
        print(f"Found {len(self.image_files)} image-label pairs")
        
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')
        
        # Get label
        label = self.labels[idx]
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_dataloader(data_dir="data/flowers", batch_size=32, shuffle=True, num_workers=4):
    """
    Creates a DataLoader for the flower dataset.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = FlowerDataset(
        data_dir=data_dir,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader
  
def create_individual_labels(labels_txt_path, output_dir):
    """
    Creates individual label files for each image from the labels.txt file.
    
    Args:
        labels_txt_path (str): Path to the labels.txt file
        output_dir (str): Directory to save individual label files
    """
    # Create output directory
    instance_labels_dir = Path(output_dir) / "image_labels"
    instance_labels_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading labels from {labels_txt_path}")
    
    # Read all labels
    labels = np.loadtxt(labels_txt_path, dtype=int)
    
    print(f"Processing {len(labels)} labels...")
    
    # Create individual files for each label
    for idx, label in enumerate(labels):
        # Create filename that matches image filename pattern
        label_filename = f"image_{idx:05d}.txt"
        label_path = instance_labels_dir / label_filename
        
        # Save individual label
        with open(label_path, 'w') as f:
            f.write(str(label))
    
    print(f"Created {len(labels)} individual label files in {instance_labels_dir}")
    
    # Create a summary file
    summary_path = instance_labels_dir / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Total images: {len(labels)}\n")
        unique_labels = np.unique(labels)
        f.write(f"Unique labels: {len(unique_labels)}\n")
        f.write("\nLabel distribution:\n")
        for label in unique_labels:
            count = np.sum(labels == label)
            f.write(f"Label {label}: {count} images\n")


if __name__ == "__main__":
    mat_file_path = "./data/flowers/labels.mat"
    output_dir = "data/flowers"
    
    # Extract the labels
    print("Extracting labels from .mat file...")
    success = extract_and_save_labels(mat_file_path, output_dir)
    
    if success:
        try:
            # Create and test the dataloader
            dataloader = get_dataloader(output_dir, batch_size=16)
            
            # Test the dataloader
            for batch_idx, (images, labels) in enumerate(dataloader):
                print(f"Batch {batch_idx + 1}:")
                print(f" - Images shape: {images.shape}")
                print(f" - Labels shape: {labels.shape}")
                print(f" - Unique labels in batch: {torch.unique(labels).numpy()}")
                break
        except Exception as e:
            print(f"Error testing dataloader: {str(e)}")
      
      
        labels_txt_path = "data/flowers/labels/labels.txt"
        output_dir = "data/flowers"
        
        try:
            create_individual_labels(labels_txt_path, output_dir)
        except Exception as e:
            print(f"Error processing labels: {e}")
        labels_dir = "data/flowers/labels"
        try:
            if os.path.exists(labels_dir):
                shutil.rmtree(labels_dir)
                print(f"Deleted directory: {labels_dir}")
            else:
                print(f"Directory does not exist: {labels_dir}")
        except Exception as e:
           print(f"Error deleting directory {labels_dir}: {str(e)}")
    else:
        print("Label extraction failed. Please check the messages above and try again.")