from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import os
import torch


class COCODataset(Dataset):
    """
    A PyTorch Dataset class for loading the COCO dataset.
    """

    def __init__(self, image_dir, annotation_file, transform=None, task="retrieval"):
        self.image_dir = image_dir
        self.annotation_file = annotation_file
        self.transform = transform
        self.task = task
        self.annotations = self._load_annotations()

    def _load_annotations(self):
        """
        Load the COCO annotation file and filter the data based on the task.
        """
        with open(self.annotation_file, "r") as f:
            data = json.load(f)

        annotations = []
        for img in data["images"]:
            entry = {"image_id": img["id"], "file_name": img["file_name"]}
            if self.task == "retrieval":
                entry["caption"] = next(
                    (ann["caption"] for ann in data["annotations"] if ann["image_id"] == img["id"]),
                    None,
                )
            annotations.append(entry)

        return annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        Retrieve an item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: (image, text) where image is the transformed image tensor and text is the caption.
        """
        item = self.annotations[idx]
        image_path = os.path.join(self.image_dir, item["file_name"])
        
        # Load and convert the image to RGB
        image = Image.open(image_path).convert("RGB")

        # Apply transformation if specified
        if self.transform:
            image = self.transform(image)

        text = item["caption"] if self.task == "retrieval" and "caption" in item else None

        return image, text


def get_data_loader(
    image_dir, annotation_file, batch_size=32, shuffle=True, task="retrieval"
):
    """
    Create a PyTorch DataLoader for the COCO dataset.

    Args:
        image_dir (str): Directory containing the COCO images.
        annotation_file (str): Path to the COCO annotation JSON file.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset.
        transform (callable, optional): Transformations to apply to the images.
        task (str): Task type ('retrieval', 'depth', etc.).

    Returns:
        DataLoader: A PyTorch DataLoader for the dataset.
    """
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize images to a fixed size
            transforms.ToTensor(),         # Convert PIL Image to PyTorch Tensor
            # transforms.Normalize(          # Normalize the image
            #     mean=(0.5, 0.5, 0.5), 
            #     std=(0.5, 0.5, 0.5)
            # ),
        ]
    )

    dataset = COCODataset(
        image_dir=image_dir,
        annotation_file=annotation_file,
        transform=transform,
        task=task,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    # Define the transformation pipeline
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize images to a fixed size
            transforms.ToTensor(),         # Convert PIL Image to PyTorch Tensor
            transforms.Normalize(          # Normalize the image
                mean=(0.5, 0.5, 0.5), 
                std=(0.5, 0.5, 0.5)
            ),
        ]
    )

    # Set paths
    image_dir = "data/coco/train2017/"
    annotation_file = "data/coco/annotations/captions_train2017.json"

    # Initialize DataLoader
    loader = get_data_loader(
        image_dir=image_dir,
        annotation_file=annotation_file,
        batch_size=8,
        shuffle=True,
        transform=transform,
        task="retrieval",
    )

    # Iterate through the DataLoader
    for images, captions in loader:
        print(f"Image Batch Shape: {images.shape}")
        print(f"Captions: {captions}")
