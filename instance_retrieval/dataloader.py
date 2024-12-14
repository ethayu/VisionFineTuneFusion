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

def extract_and_save_labels(mat_file_path, output_dir="data/flowers"):
    """
    Extracts labels from .mat file and saves them to specified directory.
    """
    # Create labels directory
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)
    
    print(f"Extracting labels from {mat_file_path}...")
    try:
        with h5py.File(mat_file_path, 'r') as f:
            # Extract labels array
            # Assuming labels are stored in the .mat file under 'labels' key
            # Modify this key based on your .mat file structure
            labels = np.array(f['labels'])
            
            # Save labels as numpy array
            labels_path = os.path.join(labels_dir, "labels.npy")
            np.save(labels_path, labels)
            
            # Also save as text file for easy viewing
            labels_txt_path = os.path.join(labels_dir, "labels.txt")
            np.savetxt(labels_txt_path, labels, fmt='%d')
            
            print(f"Extraction complete. Saved {len(labels)} labels.")
            
            # Save some basic statistics
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

if __name__ == "__main__":
    mat_file_path = "./data/flowers/labels.mat"  # Update this path to your labels.mat file
    output_dir = "data/flowers"
    
    # Extract the labels
    print("Extracting labels from .mat file...")
    if extract_and_save_labels(mat_file_path, output_dir):
        # Create and test the dataloader
        dataloader = get_dataloader(output_dir, batch_size=16)
        
        # Test the dataloader
        for batch_idx, (images, labels) in enumerate(dataloader):
            print(f"Batch {batch_idx + 1}:")
            print(f" - Images shape: {images.shape}")
            print(f" - Labels shape: {labels.shape}")
            print(f" - Unique labels in batch: {torch.unique(labels).numpy()}")
            break
    else:
        print("Label extraction failed. Please check the .mat file and try again.")